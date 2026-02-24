# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e2_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-17 16:47:16 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1775 | 0.2597 | 0.3419 | 0.4230 | 0.5052 | 0.5880 | 0.6697 | 0.7527 | 0.8334 | 0.9161 | 0.9975 |
| QAT+Prune only | 0.9274 | 0.8979 | 0.8682 | 0.8389 | 0.8087 | 0.7785 | 0.7494 | 0.7199 | 0.6898 | 0.6606 | 0.6313 |
| QAT+PTQ | 0.9274 | 0.8982 | 0.8685 | 0.8394 | 0.8092 | 0.7790 | 0.7503 | 0.7208 | 0.6908 | 0.6618 | 0.6326 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9274 | 0.8982 | 0.8685 | 0.8394 | 0.8092 | 0.7790 | 0.7503 | 0.7208 | 0.6908 | 0.6618 | 0.6326 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2123 | 0.3775 | 0.5092 | 0.6173 | 0.7077 | 0.7838 | 0.8496 | 0.9055 | 0.9554 | 0.9988 |
| QAT+Prune only | 0.0000 | 0.5528 | 0.6570 | 0.7017 | 0.7253 | 0.7403 | 0.7515 | 0.7594 | 0.7650 | 0.7700 | 0.7740 |
| QAT+PTQ | 0.0000 | 0.5539 | 0.6581 | 0.7026 | 0.7263 | 0.7411 | 0.7524 | 0.7603 | 0.7660 | 0.7710 | 0.7750 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5539 | 0.6581 | 0.7026 | 0.7263 | 0.7411 | 0.7524 | 0.7603 | 0.7660 | 0.7710 | 0.7750 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1775 | 0.1777 | 0.1780 | 0.1768 | 0.1769 | 0.1784 | 0.1780 | 0.1814 | 0.1768 | 0.1835 | 0.0000 |
| QAT+Prune only | 0.9274 | 0.9276 | 0.9274 | 0.9279 | 0.9269 | 0.9256 | 0.9266 | 0.9267 | 0.9235 | 0.9238 | 0.0000 |
| QAT+PTQ | 0.9274 | 0.9277 | 0.9275 | 0.9280 | 0.9270 | 0.9255 | 0.9268 | 0.9267 | 0.9238 | 0.9245 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9274 | 0.9277 | 0.9275 | 0.9280 | 0.9270 | 0.9255 | 0.9268 | 0.9267 | 0.9238 | 0.9245 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1775 | 0.0000 | 0.0000 | 0.0000 | 0.1775 | 1.0000 |
| 90 | 10 | 299,940 | 0.2597 | 0.1188 | 0.9978 | 0.2123 | 0.1777 | 0.9986 |
| 80 | 20 | 291,350 | 0.3419 | 0.2328 | 0.9975 | 0.3775 | 0.1780 | 0.9966 |
| 70 | 30 | 194,230 | 0.4230 | 0.3418 | 0.9975 | 0.5092 | 0.1768 | 0.9941 |
| 60 | 40 | 145,675 | 0.5052 | 0.4469 | 0.9975 | 0.6173 | 0.1769 | 0.9908 |
| 50 | 50 | 116,540 | 0.5880 | 0.5484 | 0.9975 | 0.7077 | 0.1784 | 0.9864 |
| 40 | 60 | 97,115 | 0.6697 | 0.6454 | 0.9975 | 0.7838 | 0.1780 | 0.9797 |
| 30 | 70 | 83,240 | 0.7527 | 0.7398 | 0.9975 | 0.8496 | 0.1814 | 0.9694 |
| 20 | 80 | 72,835 | 0.8334 | 0.8290 | 0.9975 | 0.9055 | 0.1768 | 0.9474 |
| 10 | 90 | 64,740 | 0.9161 | 0.9166 | 0.9975 | 0.9554 | 0.1835 | 0.8926 |
| 0 | 100 | 58,270 | 0.9975 | 1.0000 | 0.9975 | 0.9988 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9274 | 0.0000 | 0.0000 | 0.0000 | 0.9274 | 1.0000 |
| 90 | 10 | 299,940 | 0.8979 | 0.4919 | 0.6309 | 0.5528 | 0.9276 | 0.9577 |
| 80 | 20 | 291,350 | 0.8682 | 0.6848 | 0.6313 | 0.6570 | 0.9274 | 0.9096 |
| 70 | 30 | 194,230 | 0.8389 | 0.7896 | 0.6313 | 0.7017 | 0.9279 | 0.8545 |
| 60 | 40 | 145,675 | 0.8087 | 0.8521 | 0.6313 | 0.7253 | 0.9269 | 0.7904 |
| 50 | 50 | 116,540 | 0.7785 | 0.8946 | 0.6313 | 0.7403 | 0.9256 | 0.7152 |
| 40 | 60 | 97,115 | 0.7494 | 0.9281 | 0.6313 | 0.7515 | 0.9266 | 0.6263 |
| 30 | 70 | 83,240 | 0.7199 | 0.9526 | 0.6313 | 0.7594 | 0.9267 | 0.5186 |
| 20 | 80 | 72,835 | 0.6898 | 0.9706 | 0.6313 | 0.7650 | 0.9235 | 0.3851 |
| 10 | 90 | 64,740 | 0.6606 | 0.9868 | 0.6313 | 0.7700 | 0.9238 | 0.2178 |
| 0 | 100 | 58,270 | 0.6313 | 1.0000 | 0.6313 | 0.7740 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9274 | 0.0000 | 0.0000 | 0.0000 | 0.9274 | 1.0000 |
| 90 | 10 | 299,940 | 0.8982 | 0.4929 | 0.6321 | 0.5539 | 0.9277 | 0.9578 |
| 80 | 20 | 291,350 | 0.8685 | 0.6857 | 0.6326 | 0.6581 | 0.9275 | 0.9099 |
| 70 | 30 | 194,230 | 0.8394 | 0.7901 | 0.6326 | 0.7026 | 0.9280 | 0.8549 |
| 60 | 40 | 145,675 | 0.8092 | 0.8525 | 0.6326 | 0.7263 | 0.9270 | 0.7910 |
| 50 | 50 | 116,540 | 0.7790 | 0.8946 | 0.6326 | 0.7411 | 0.9255 | 0.7158 |
| 40 | 60 | 97,115 | 0.7503 | 0.9283 | 0.6326 | 0.7524 | 0.9268 | 0.6271 |
| 30 | 70 | 83,240 | 0.7208 | 0.9527 | 0.6326 | 0.7603 | 0.9267 | 0.5195 |
| 20 | 80 | 72,835 | 0.6908 | 0.9708 | 0.6326 | 0.7660 | 0.9238 | 0.3860 |
| 10 | 90 | 64,740 | 0.6618 | 0.9869 | 0.6326 | 0.7710 | 0.9245 | 0.2185 |
| 0 | 100 | 58,270 | 0.6326 | 1.0000 | 0.6326 | 0.7750 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9274 | 0.0000 | 0.0000 | 0.0000 | 0.9274 | 1.0000 |
| 90 | 10 | 299,940 | 0.8982 | 0.4929 | 0.6321 | 0.5539 | 0.9277 | 0.9578 |
| 80 | 20 | 291,350 | 0.8685 | 0.6857 | 0.6326 | 0.6581 | 0.9275 | 0.9099 |
| 70 | 30 | 194,230 | 0.8394 | 0.7901 | 0.6326 | 0.7026 | 0.9280 | 0.8549 |
| 60 | 40 | 145,675 | 0.8092 | 0.8525 | 0.6326 | 0.7263 | 0.9270 | 0.7910 |
| 50 | 50 | 116,540 | 0.7790 | 0.8946 | 0.6326 | 0.7411 | 0.9255 | 0.7158 |
| 40 | 60 | 97,115 | 0.7503 | 0.9283 | 0.6326 | 0.7524 | 0.9268 | 0.6271 |
| 30 | 70 | 83,240 | 0.7208 | 0.9527 | 0.6326 | 0.7603 | 0.9267 | 0.5195 |
| 20 | 80 | 72,835 | 0.6908 | 0.9708 | 0.6326 | 0.7660 | 0.9238 | 0.3860 |
| 10 | 90 | 64,740 | 0.6618 | 0.9869 | 0.6326 | 0.7710 | 0.9245 | 0.2185 |
| 0 | 100 | 58,270 | 0.6326 | 1.0000 | 0.6326 | 0.7750 | 0.0000 | 0.0000 |


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
0.15       0.2597   0.2123   0.1777   0.9984   0.9975   0.1188   <--
0.20       0.2597   0.2123   0.1777   0.9984   0.9975   0.1188  
0.25       0.2597   0.2123   0.1777   0.9984   0.9975   0.1188  
0.30       0.2597   0.2123   0.1777   0.9984   0.9975   0.1188  
0.35       0.2597   0.2123   0.1777   0.9984   0.9975   0.1188  
0.40       0.2597   0.2123   0.1777   0.9984   0.9975   0.1188  
0.45       0.2597   0.2123   0.1777   0.9984   0.9975   0.1188  
0.50       0.2597   0.2123   0.1777   0.9984   0.9975   0.1188  
0.55       0.2597   0.2123   0.1777   0.9984   0.9975   0.1188  
0.60       0.2597   0.2123   0.1777   0.9984   0.9975   0.1188  
0.65       0.2597   0.2123   0.1777   0.9984   0.9975   0.1188  
0.70       0.2597   0.2123   0.1777   0.9984   0.9975   0.1188  
0.75       0.2597   0.2123   0.1777   0.9984   0.9975   0.1188  
0.80       0.2597   0.2123   0.1777   0.9984   0.9975   0.1188  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2597, F1=0.2123, Normal Recall=0.1777, Normal Precision=0.9984, Attack Recall=0.9975, Attack Precision=0.1188

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
0.15       0.3414   0.3773   0.1774   0.9966   0.9975   0.2326   <--
0.20       0.3414   0.3773   0.1774   0.9966   0.9975   0.2326  
0.25       0.3414   0.3773   0.1774   0.9966   0.9975   0.2326  
0.30       0.3414   0.3773   0.1774   0.9966   0.9975   0.2326  
0.35       0.3414   0.3773   0.1774   0.9966   0.9975   0.2326  
0.40       0.3414   0.3773   0.1774   0.9966   0.9975   0.2326  
0.45       0.3414   0.3773   0.1774   0.9966   0.9975   0.2326  
0.50       0.3414   0.3773   0.1774   0.9966   0.9975   0.2326  
0.55       0.3414   0.3773   0.1774   0.9966   0.9975   0.2326  
0.60       0.3414   0.3773   0.1774   0.9966   0.9975   0.2326  
0.65       0.3414   0.3773   0.1774   0.9966   0.9975   0.2326  
0.70       0.3414   0.3773   0.1774   0.9966   0.9975   0.2326  
0.75       0.3414   0.3773   0.1774   0.9966   0.9975   0.2326  
0.80       0.3414   0.3773   0.1774   0.9966   0.9975   0.2326  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3414, F1=0.3773, Normal Recall=0.1774, Normal Precision=0.9966, Attack Recall=0.9975, Attack Precision=0.2326

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
0.15       0.4238   0.5095   0.1779   0.9941   0.9975   0.3421   <--
0.20       0.4238   0.5095   0.1779   0.9941   0.9975   0.3421  
0.25       0.4238   0.5095   0.1779   0.9941   0.9975   0.3421  
0.30       0.4238   0.5095   0.1779   0.9941   0.9975   0.3421  
0.35       0.4238   0.5095   0.1779   0.9941   0.9975   0.3421  
0.40       0.4238   0.5095   0.1779   0.9941   0.9975   0.3421  
0.45       0.4238   0.5095   0.1779   0.9941   0.9975   0.3421  
0.50       0.4238   0.5095   0.1779   0.9941   0.9975   0.3421  
0.55       0.4238   0.5095   0.1779   0.9941   0.9975   0.3421  
0.60       0.4238   0.5095   0.1779   0.9941   0.9975   0.3421  
0.65       0.4238   0.5095   0.1779   0.9941   0.9975   0.3421  
0.70       0.4238   0.5095   0.1779   0.9941   0.9975   0.3421  
0.75       0.4238   0.5095   0.1779   0.9941   0.9975   0.3421  
0.80       0.4238   0.5095   0.1779   0.9941   0.9975   0.3421  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4238, F1=0.5095, Normal Recall=0.1779, Normal Precision=0.9941, Attack Recall=0.9975, Attack Precision=0.3421

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
0.15       0.5057   0.6175   0.1779   0.9909   0.9975   0.4472   <--
0.20       0.5057   0.6175   0.1779   0.9909   0.9975   0.4472  
0.25       0.5057   0.6175   0.1779   0.9909   0.9975   0.4472  
0.30       0.5057   0.6175   0.1779   0.9909   0.9975   0.4472  
0.35       0.5057   0.6175   0.1779   0.9909   0.9975   0.4472  
0.40       0.5057   0.6175   0.1779   0.9909   0.9975   0.4472  
0.45       0.5057   0.6175   0.1779   0.9909   0.9975   0.4472  
0.50       0.5057   0.6175   0.1779   0.9909   0.9975   0.4472  
0.55       0.5057   0.6175   0.1779   0.9909   0.9975   0.4472  
0.60       0.5057   0.6175   0.1779   0.9909   0.9975   0.4472  
0.65       0.5057   0.6175   0.1779   0.9909   0.9975   0.4472  
0.70       0.5057   0.6175   0.1779   0.9909   0.9975   0.4472  
0.75       0.5057   0.6175   0.1779   0.9909   0.9975   0.4472  
0.80       0.5057   0.6175   0.1779   0.9909   0.9975   0.4472  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5057, F1=0.6175, Normal Recall=0.1779, Normal Precision=0.9909, Attack Recall=0.9975, Attack Precision=0.4472

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
0.15       0.5882   0.7078   0.1788   0.9865   0.9975   0.5485   <--
0.20       0.5882   0.7078   0.1788   0.9865   0.9975   0.5485  
0.25       0.5882   0.7078   0.1788   0.9865   0.9975   0.5485  
0.30       0.5882   0.7078   0.1788   0.9865   0.9975   0.5485  
0.35       0.5882   0.7078   0.1788   0.9865   0.9975   0.5485  
0.40       0.5882   0.7078   0.1788   0.9865   0.9975   0.5485  
0.45       0.5882   0.7078   0.1788   0.9865   0.9975   0.5485  
0.50       0.5882   0.7078   0.1788   0.9865   0.9975   0.5485  
0.55       0.5882   0.7078   0.1788   0.9865   0.9975   0.5485  
0.60       0.5882   0.7078   0.1788   0.9865   0.9975   0.5485  
0.65       0.5882   0.7078   0.1788   0.9865   0.9975   0.5485  
0.70       0.5882   0.7078   0.1788   0.9865   0.9975   0.5485  
0.75       0.5882   0.7078   0.1788   0.9865   0.9975   0.5485  
0.80       0.5882   0.7078   0.1788   0.9865   0.9975   0.5485  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5882, F1=0.7078, Normal Recall=0.1788, Normal Precision=0.9865, Attack Recall=0.9975, Attack Precision=0.5485

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
0.15       0.8977   0.5516   0.9276   0.9575   0.6290   0.4912   <--
0.20       0.8977   0.5516   0.9276   0.9575   0.6290   0.4912  
0.25       0.8977   0.5516   0.9276   0.9575   0.6290   0.4912  
0.30       0.8977   0.5516   0.9276   0.9575   0.6290   0.4912  
0.35       0.8977   0.5516   0.9276   0.9575   0.6290   0.4912  
0.40       0.8977   0.5516   0.9276   0.9575   0.6290   0.4912  
0.45       0.8977   0.5516   0.9276   0.9575   0.6290   0.4912  
0.50       0.8977   0.5516   0.9276   0.9575   0.6290   0.4912  
0.55       0.8977   0.5516   0.9276   0.9575   0.6290   0.4912  
0.60       0.8977   0.5516   0.9276   0.9575   0.6290   0.4912  
0.65       0.8977   0.5516   0.9276   0.9575   0.6290   0.4912  
0.70       0.8977   0.5516   0.9276   0.9575   0.6290   0.4912  
0.75       0.8977   0.5516   0.9276   0.9575   0.6290   0.4912  
0.80       0.8977   0.5516   0.9276   0.9575   0.6290   0.4912  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8977, F1=0.5516, Normal Recall=0.9276, Normal Precision=0.9575, Attack Recall=0.6290, Attack Precision=0.4912

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
0.15       0.8682   0.6571   0.9274   0.9096   0.6313   0.6851   <--
0.20       0.8682   0.6571   0.9274   0.9096   0.6313   0.6851  
0.25       0.8682   0.6571   0.9274   0.9096   0.6313   0.6851  
0.30       0.8682   0.6571   0.9274   0.9096   0.6313   0.6851  
0.35       0.8682   0.6571   0.9274   0.9096   0.6313   0.6851  
0.40       0.8682   0.6571   0.9274   0.9096   0.6313   0.6851  
0.45       0.8682   0.6571   0.9274   0.9096   0.6313   0.6851  
0.50       0.8682   0.6571   0.9274   0.9096   0.6313   0.6851  
0.55       0.8682   0.6571   0.9274   0.9096   0.6313   0.6851  
0.60       0.8682   0.6571   0.9274   0.9096   0.6313   0.6851  
0.65       0.8682   0.6571   0.9274   0.9096   0.6313   0.6851  
0.70       0.8682   0.6571   0.9274   0.9096   0.6313   0.6851  
0.75       0.8682   0.6571   0.9274   0.9096   0.6313   0.6851  
0.80       0.8682   0.6571   0.9274   0.9096   0.6313   0.6851  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8682, F1=0.6571, Normal Recall=0.9274, Normal Precision=0.9096, Attack Recall=0.6313, Attack Precision=0.6851

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
0.15       0.8385   0.7010   0.9272   0.8544   0.6313   0.7881   <--
0.20       0.8385   0.7010   0.9272   0.8544   0.6313   0.7881  
0.25       0.8385   0.7010   0.9272   0.8544   0.6313   0.7881  
0.30       0.8385   0.7010   0.9272   0.8544   0.6313   0.7881  
0.35       0.8385   0.7010   0.9272   0.8544   0.6313   0.7881  
0.40       0.8385   0.7010   0.9272   0.8544   0.6313   0.7881  
0.45       0.8385   0.7010   0.9272   0.8544   0.6313   0.7881  
0.50       0.8385   0.7010   0.9272   0.8544   0.6313   0.7881  
0.55       0.8385   0.7010   0.9272   0.8544   0.6313   0.7881  
0.60       0.8385   0.7010   0.9272   0.8544   0.6313   0.7881  
0.65       0.8385   0.7010   0.9272   0.8544   0.6313   0.7881  
0.70       0.8385   0.7010   0.9272   0.8544   0.6313   0.7881  
0.75       0.8385   0.7010   0.9272   0.8544   0.6313   0.7881  
0.80       0.8385   0.7010   0.9272   0.8544   0.6313   0.7881  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8385, F1=0.7010, Normal Recall=0.9272, Normal Precision=0.8544, Attack Recall=0.6313, Attack Precision=0.7881

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
0.15       0.8088   0.7254   0.9271   0.7905   0.6313   0.8524   <--
0.20       0.8088   0.7254   0.9271   0.7905   0.6313   0.8524  
0.25       0.8088   0.7254   0.9271   0.7905   0.6313   0.8524  
0.30       0.8088   0.7254   0.9271   0.7905   0.6313   0.8524  
0.35       0.8088   0.7254   0.9271   0.7905   0.6313   0.8524  
0.40       0.8088   0.7254   0.9271   0.7905   0.6313   0.8524  
0.45       0.8088   0.7254   0.9271   0.7905   0.6313   0.8524  
0.50       0.8088   0.7254   0.9271   0.7905   0.6313   0.8524  
0.55       0.8088   0.7254   0.9271   0.7905   0.6313   0.8524  
0.60       0.8088   0.7254   0.9271   0.7905   0.6313   0.8524  
0.65       0.8088   0.7254   0.9271   0.7905   0.6313   0.8524  
0.70       0.8088   0.7254   0.9271   0.7905   0.6313   0.8524  
0.75       0.8088   0.7254   0.9271   0.7905   0.6313   0.8524  
0.80       0.8088   0.7254   0.9271   0.7905   0.6313   0.8524  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8088, F1=0.7254, Normal Recall=0.9271, Normal Precision=0.7905, Attack Recall=0.6313, Attack Precision=0.8524

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
0.15       0.7788   0.7406   0.9263   0.7153   0.6313   0.8955   <--
0.20       0.7788   0.7406   0.9263   0.7153   0.6313   0.8955  
0.25       0.7788   0.7406   0.9263   0.7153   0.6313   0.8955  
0.30       0.7788   0.7406   0.9263   0.7153   0.6313   0.8955  
0.35       0.7788   0.7406   0.9263   0.7153   0.6313   0.8955  
0.40       0.7788   0.7406   0.9263   0.7153   0.6313   0.8955  
0.45       0.7788   0.7406   0.9263   0.7153   0.6313   0.8955  
0.50       0.7788   0.7406   0.9263   0.7153   0.6313   0.8955  
0.55       0.7788   0.7406   0.9263   0.7153   0.6313   0.8955  
0.60       0.7788   0.7406   0.9263   0.7153   0.6313   0.8955  
0.65       0.7788   0.7406   0.9263   0.7153   0.6313   0.8955  
0.70       0.7788   0.7406   0.9263   0.7153   0.6313   0.8955  
0.75       0.7788   0.7406   0.9263   0.7153   0.6313   0.8955  
0.80       0.7788   0.7406   0.9263   0.7153   0.6313   0.8955  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7788, F1=0.7406, Normal Recall=0.9263, Normal Precision=0.7153, Attack Recall=0.6313, Attack Precision=0.8955

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
0.15       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922   <--
0.20       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.25       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.30       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.35       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.40       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.45       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.50       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.55       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.60       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.65       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.70       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.75       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.80       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8980, F1=0.5527, Normal Recall=0.9277, Normal Precision=0.9576, Attack Recall=0.6303, Attack Precision=0.4922

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
0.15       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859   <--
0.20       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.25       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.30       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.35       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.40       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.45       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.50       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.55       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.60       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.65       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.70       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.75       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.80       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8686, F1=0.6582, Normal Recall=0.9276, Normal Precision=0.9099, Attack Recall=0.6326, Attack Precision=0.6859

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
0.15       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887   <--
0.20       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.25       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.30       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.35       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.40       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.45       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.50       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.55       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.60       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.65       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.70       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.75       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.80       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8389, F1=0.7020, Normal Recall=0.9273, Normal Precision=0.8548, Attack Recall=0.6326, Attack Precision=0.7887

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
0.15       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527   <--
0.20       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.25       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.30       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.35       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.40       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.45       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.50       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.55       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.60       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.65       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.70       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.75       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.80       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8093, F1=0.7263, Normal Recall=0.9271, Normal Precision=0.7910, Attack Recall=0.6326, Attack Precision=0.8527

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
0.15       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958   <--
0.20       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.25       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.30       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.35       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.40       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.45       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.50       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.55       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.60       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.65       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.70       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.75       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.80       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7795, F1=0.7415, Normal Recall=0.9264, Normal Precision=0.7160, Attack Recall=0.6326, Attack Precision=0.8958

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
0.15       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922   <--
0.20       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.25       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.30       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.35       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.40       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.45       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.50       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.55       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.60       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.65       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.70       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.75       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
0.80       0.8980   0.5527   0.9277   0.9576   0.6303   0.4922  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8980, F1=0.5527, Normal Recall=0.9277, Normal Precision=0.9576, Attack Recall=0.6303, Attack Precision=0.4922

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
0.15       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859   <--
0.20       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.25       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.30       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.35       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.40       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.45       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.50       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.55       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.60       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.65       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.70       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.75       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
0.80       0.8686   0.6582   0.9276   0.9099   0.6326   0.6859  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8686, F1=0.6582, Normal Recall=0.9276, Normal Precision=0.9099, Attack Recall=0.6326, Attack Precision=0.6859

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
0.15       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887   <--
0.20       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.25       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.30       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.35       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.40       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.45       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.50       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.55       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.60       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.65       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.70       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.75       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
0.80       0.8389   0.7020   0.9273   0.8548   0.6326   0.7887  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8389, F1=0.7020, Normal Recall=0.9273, Normal Precision=0.8548, Attack Recall=0.6326, Attack Precision=0.7887

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
0.15       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527   <--
0.20       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.25       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.30       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.35       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.40       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.45       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.50       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.55       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.60       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.65       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.70       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.75       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
0.80       0.8093   0.7263   0.9271   0.7910   0.6326   0.8527  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8093, F1=0.7263, Normal Recall=0.9271, Normal Precision=0.7910, Attack Recall=0.6326, Attack Precision=0.8527

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
0.15       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958   <--
0.20       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.25       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.30       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.35       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.40       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.45       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.50       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.55       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.60       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.65       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.70       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.75       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
0.80       0.7795   0.7415   0.9264   0.7160   0.6326   0.8958  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7795, F1=0.7415, Normal Recall=0.9264, Normal Precision=0.7160, Attack Recall=0.6326, Attack Precision=0.8958

```

