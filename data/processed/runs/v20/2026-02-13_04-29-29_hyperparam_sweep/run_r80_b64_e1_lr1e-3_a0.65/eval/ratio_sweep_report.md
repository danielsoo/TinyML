# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e1_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-16 10:35:35 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2635 | 0.3303 | 0.3959 | 0.4616 | 0.5274 | 0.5938 | 0.6596 | 0.7244 | 0.7911 | 0.8570 | 0.9220 |
| QAT+Prune only | 0.8204 | 0.8386 | 0.8556 | 0.8734 | 0.8908 | 0.9073 | 0.9271 | 0.9435 | 0.9607 | 0.9782 | 0.9958 |
| QAT+PTQ | 0.8200 | 0.8383 | 0.8554 | 0.8732 | 0.8906 | 0.9071 | 0.9270 | 0.9434 | 0.9606 | 0.9783 | 0.9958 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8200 | 0.8383 | 0.8554 | 0.8732 | 0.8906 | 0.9071 | 0.9270 | 0.9434 | 0.9606 | 0.9783 | 0.9958 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2162 | 0.3791 | 0.5068 | 0.6095 | 0.6942 | 0.7647 | 0.8241 | 0.8760 | 0.9207 | 0.9594 |
| QAT+Prune only | 0.0000 | 0.5524 | 0.7340 | 0.8252 | 0.8794 | 0.9148 | 0.9425 | 0.9610 | 0.9760 | 0.9880 | 0.9979 |
| QAT+PTQ | 0.0000 | 0.5519 | 0.7336 | 0.8249 | 0.8793 | 0.9146 | 0.9424 | 0.9610 | 0.9759 | 0.9880 | 0.9979 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5519 | 0.7336 | 0.8249 | 0.8793 | 0.9146 | 0.9424 | 0.9610 | 0.9759 | 0.9880 | 0.9979 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2635 | 0.2644 | 0.2643 | 0.2643 | 0.2643 | 0.2656 | 0.2659 | 0.2632 | 0.2675 | 0.2712 | 0.0000 |
| QAT+Prune only | 0.8204 | 0.8211 | 0.8206 | 0.8210 | 0.8208 | 0.8188 | 0.8241 | 0.8215 | 0.8206 | 0.8197 | 0.0000 |
| QAT+PTQ | 0.8200 | 0.8207 | 0.8203 | 0.8207 | 0.8205 | 0.8183 | 0.8238 | 0.8212 | 0.8197 | 0.8204 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8200 | 0.8207 | 0.8203 | 0.8207 | 0.8205 | 0.8183 | 0.8238 | 0.8212 | 0.8197 | 0.8204 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2635 | 0.0000 | 0.0000 | 0.0000 | 0.2635 | 1.0000 |
| 90 | 10 | 299,940 | 0.3303 | 0.1224 | 0.9235 | 0.2162 | 0.2644 | 0.9688 |
| 80 | 20 | 291,350 | 0.3959 | 0.2386 | 0.9220 | 0.3791 | 0.2643 | 0.9313 |
| 70 | 30 | 194,230 | 0.4616 | 0.3494 | 0.9220 | 0.5068 | 0.2643 | 0.8878 |
| 60 | 40 | 145,675 | 0.5274 | 0.4552 | 0.9220 | 0.6095 | 0.2643 | 0.8357 |
| 50 | 50 | 116,540 | 0.5938 | 0.5567 | 0.9220 | 0.6942 | 0.2656 | 0.7731 |
| 40 | 60 | 97,115 | 0.6596 | 0.6533 | 0.9220 | 0.7647 | 0.2659 | 0.6945 |
| 30 | 70 | 83,240 | 0.7244 | 0.7449 | 0.9220 | 0.8241 | 0.2632 | 0.5913 |
| 20 | 80 | 72,835 | 0.7911 | 0.8343 | 0.9220 | 0.8760 | 0.2675 | 0.4617 |
| 10 | 90 | 64,740 | 0.8570 | 0.9193 | 0.9220 | 0.9207 | 0.2712 | 0.2788 |
| 0 | 100 | 58,270 | 0.9220 | 1.0000 | 0.9220 | 0.9594 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8204 | 0.0000 | 0.0000 | 0.0000 | 0.8204 | 1.0000 |
| 90 | 10 | 299,940 | 0.8386 | 0.3822 | 0.9961 | 0.5524 | 0.8211 | 0.9995 |
| 80 | 20 | 291,350 | 0.8556 | 0.5812 | 0.9958 | 0.7340 | 0.8206 | 0.9987 |
| 70 | 30 | 194,230 | 0.8734 | 0.7045 | 0.9958 | 0.8252 | 0.8210 | 0.9978 |
| 60 | 40 | 145,675 | 0.8908 | 0.7874 | 0.9958 | 0.8794 | 0.8208 | 0.9966 |
| 50 | 50 | 116,540 | 0.9073 | 0.8461 | 0.9958 | 0.9148 | 0.8188 | 0.9949 |
| 40 | 60 | 97,115 | 0.9271 | 0.8947 | 0.9958 | 0.9425 | 0.8241 | 0.9924 |
| 30 | 70 | 83,240 | 0.9435 | 0.9286 | 0.9958 | 0.9610 | 0.8215 | 0.9882 |
| 20 | 80 | 72,835 | 0.9607 | 0.9569 | 0.9958 | 0.9760 | 0.8206 | 0.9798 |
| 10 | 90 | 64,740 | 0.9782 | 0.9803 | 0.9958 | 0.9880 | 0.8197 | 0.9557 |
| 0 | 100 | 58,270 | 0.9958 | 1.0000 | 0.9958 | 0.9979 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8200 | 0.0000 | 0.0000 | 0.0000 | 0.8200 | 1.0000 |
| 90 | 10 | 299,940 | 0.8383 | 0.3817 | 0.9961 | 0.5519 | 0.8207 | 0.9995 |
| 80 | 20 | 291,350 | 0.8554 | 0.5807 | 0.9958 | 0.7336 | 0.8203 | 0.9987 |
| 70 | 30 | 194,230 | 0.8732 | 0.7041 | 0.9958 | 0.8249 | 0.8207 | 0.9978 |
| 60 | 40 | 145,675 | 0.8906 | 0.7872 | 0.9958 | 0.8793 | 0.8205 | 0.9966 |
| 50 | 50 | 116,540 | 0.9071 | 0.8457 | 0.9958 | 0.9146 | 0.8183 | 0.9949 |
| 40 | 60 | 97,115 | 0.9270 | 0.8945 | 0.9958 | 0.9424 | 0.8238 | 0.9924 |
| 30 | 70 | 83,240 | 0.9434 | 0.9285 | 0.9958 | 0.9610 | 0.8212 | 0.9882 |
| 20 | 80 | 72,835 | 0.9606 | 0.9567 | 0.9958 | 0.9759 | 0.8197 | 0.9799 |
| 10 | 90 | 64,740 | 0.9783 | 0.9803 | 0.9958 | 0.9880 | 0.8204 | 0.9559 |
| 0 | 100 | 58,270 | 0.9958 | 1.0000 | 0.9958 | 0.9979 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8200 | 0.0000 | 0.0000 | 0.0000 | 0.8200 | 1.0000 |
| 90 | 10 | 299,940 | 0.8383 | 0.3817 | 0.9961 | 0.5519 | 0.8207 | 0.9995 |
| 80 | 20 | 291,350 | 0.8554 | 0.5807 | 0.9958 | 0.7336 | 0.8203 | 0.9987 |
| 70 | 30 | 194,230 | 0.8732 | 0.7041 | 0.9958 | 0.8249 | 0.8207 | 0.9978 |
| 60 | 40 | 145,675 | 0.8906 | 0.7872 | 0.9958 | 0.8793 | 0.8205 | 0.9966 |
| 50 | 50 | 116,540 | 0.9071 | 0.8457 | 0.9958 | 0.9146 | 0.8183 | 0.9949 |
| 40 | 60 | 97,115 | 0.9270 | 0.8945 | 0.9958 | 0.9424 | 0.8238 | 0.9924 |
| 30 | 70 | 83,240 | 0.9434 | 0.9285 | 0.9958 | 0.9610 | 0.8212 | 0.9882 |
| 20 | 80 | 72,835 | 0.9606 | 0.9567 | 0.9958 | 0.9759 | 0.8197 | 0.9799 |
| 10 | 90 | 64,740 | 0.9783 | 0.9803 | 0.9958 | 0.9880 | 0.8204 | 0.9559 |
| 0 | 100 | 58,270 | 0.9958 | 1.0000 | 0.9958 | 0.9979 | 0.0000 | 0.0000 |


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
0.15       0.3299   0.2154   0.2644   0.9674   0.9199   0.1220   <--
0.20       0.3299   0.2154   0.2644   0.9674   0.9199   0.1220  
0.25       0.3299   0.2154   0.2644   0.9674   0.9199   0.1220  
0.30       0.3299   0.2154   0.2644   0.9674   0.9199   0.1220  
0.35       0.3299   0.2154   0.2644   0.9674   0.9199   0.1220  
0.40       0.3299   0.2154   0.2644   0.9674   0.9199   0.1220  
0.45       0.3299   0.2154   0.2644   0.9674   0.9199   0.1220  
0.50       0.3299   0.2154   0.2644   0.9674   0.9199   0.1220  
0.55       0.3299   0.2154   0.2644   0.9674   0.9199   0.1220  
0.60       0.3299   0.2154   0.2644   0.9674   0.9199   0.1220  
0.65       0.3299   0.2154   0.2644   0.9674   0.9199   0.1220  
0.70       0.3299   0.2154   0.2644   0.9674   0.9199   0.1220  
0.75       0.3299   0.2154   0.2644   0.9674   0.9199   0.1220  
0.80       0.3299   0.2154   0.2644   0.9674   0.9199   0.1220  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3299, F1=0.2154, Normal Recall=0.2644, Normal Precision=0.9674, Attack Recall=0.9199, Attack Precision=0.1220

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
0.15       0.3954   0.3789   0.2638   0.9312   0.9220   0.2384   <--
0.20       0.3954   0.3789   0.2638   0.9312   0.9220   0.2384  
0.25       0.3954   0.3789   0.2638   0.9312   0.9220   0.2384  
0.30       0.3954   0.3789   0.2638   0.9312   0.9220   0.2384  
0.35       0.3954   0.3789   0.2638   0.9312   0.9220   0.2384  
0.40       0.3954   0.3789   0.2638   0.9312   0.9220   0.2384  
0.45       0.3954   0.3789   0.2638   0.9312   0.9220   0.2384  
0.50       0.3954   0.3789   0.2638   0.9312   0.9220   0.2384  
0.55       0.3954   0.3789   0.2638   0.9312   0.9220   0.2384  
0.60       0.3954   0.3789   0.2638   0.9312   0.9220   0.2384  
0.65       0.3954   0.3789   0.2638   0.9312   0.9220   0.2384  
0.70       0.3954   0.3789   0.2638   0.9312   0.9220   0.2384  
0.75       0.3954   0.3789   0.2638   0.9312   0.9220   0.2384  
0.80       0.3954   0.3789   0.2638   0.9312   0.9220   0.2384  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3954, F1=0.3789, Normal Recall=0.2638, Normal Precision=0.9312, Attack Recall=0.9220, Attack Precision=0.2384

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
0.15       0.4620   0.5070   0.2649   0.8880   0.9220   0.3496   <--
0.20       0.4620   0.5070   0.2649   0.8880   0.9220   0.3496  
0.25       0.4620   0.5070   0.2649   0.8880   0.9220   0.3496  
0.30       0.4620   0.5070   0.2649   0.8880   0.9220   0.3496  
0.35       0.4620   0.5070   0.2649   0.8880   0.9220   0.3496  
0.40       0.4620   0.5070   0.2649   0.8880   0.9220   0.3496  
0.45       0.4620   0.5070   0.2649   0.8880   0.9220   0.3496  
0.50       0.4620   0.5070   0.2649   0.8880   0.9220   0.3496  
0.55       0.4620   0.5070   0.2649   0.8880   0.9220   0.3496  
0.60       0.4620   0.5070   0.2649   0.8880   0.9220   0.3496  
0.65       0.4620   0.5070   0.2649   0.8880   0.9220   0.3496  
0.70       0.4620   0.5070   0.2649   0.8880   0.9220   0.3496  
0.75       0.4620   0.5070   0.2649   0.8880   0.9220   0.3496  
0.80       0.4620   0.5070   0.2649   0.8880   0.9220   0.3496  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4620, F1=0.5070, Normal Recall=0.2649, Normal Precision=0.8880, Attack Recall=0.9220, Attack Precision=0.3496

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
0.15       0.5274   0.6095   0.2643   0.8356   0.9220   0.4552   <--
0.20       0.5274   0.6095   0.2643   0.8356   0.9220   0.4552  
0.25       0.5274   0.6095   0.2643   0.8356   0.9220   0.4552  
0.30       0.5274   0.6095   0.2643   0.8356   0.9220   0.4552  
0.35       0.5274   0.6095   0.2643   0.8356   0.9220   0.4552  
0.40       0.5274   0.6095   0.2643   0.8356   0.9220   0.4552  
0.45       0.5274   0.6095   0.2643   0.8356   0.9220   0.4552  
0.50       0.5274   0.6095   0.2643   0.8356   0.9220   0.4552  
0.55       0.5274   0.6095   0.2643   0.8356   0.9220   0.4552  
0.60       0.5274   0.6095   0.2643   0.8356   0.9220   0.4552  
0.65       0.5274   0.6095   0.2643   0.8356   0.9220   0.4552  
0.70       0.5274   0.6095   0.2643   0.8356   0.9220   0.4552  
0.75       0.5274   0.6095   0.2643   0.8356   0.9220   0.4552  
0.80       0.5274   0.6095   0.2643   0.8356   0.9220   0.4552  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5274, F1=0.6095, Normal Recall=0.2643, Normal Precision=0.8356, Attack Recall=0.9220, Attack Precision=0.4552

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
0.15       0.5943   0.6944   0.2665   0.7737   0.9220   0.5569   <--
0.20       0.5943   0.6944   0.2665   0.7737   0.9220   0.5569  
0.25       0.5943   0.6944   0.2665   0.7737   0.9220   0.5569  
0.30       0.5943   0.6944   0.2665   0.7737   0.9220   0.5569  
0.35       0.5943   0.6944   0.2665   0.7737   0.9220   0.5569  
0.40       0.5943   0.6944   0.2665   0.7737   0.9220   0.5569  
0.45       0.5943   0.6944   0.2665   0.7737   0.9220   0.5569  
0.50       0.5943   0.6944   0.2665   0.7737   0.9220   0.5569  
0.55       0.5943   0.6944   0.2665   0.7737   0.9220   0.5569  
0.60       0.5943   0.6944   0.2665   0.7737   0.9220   0.5569  
0.65       0.5943   0.6944   0.2665   0.7737   0.9220   0.5569  
0.70       0.5943   0.6944   0.2665   0.7737   0.9220   0.5569  
0.75       0.5943   0.6944   0.2665   0.7737   0.9220   0.5569  
0.80       0.5943   0.6944   0.2665   0.7737   0.9220   0.5569  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5943, F1=0.6944, Normal Recall=0.2665, Normal Precision=0.7737, Attack Recall=0.9220, Attack Precision=0.5569

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
0.15       0.8386   0.5524   0.8211   0.9995   0.9960   0.3822   <--
0.20       0.8386   0.5524   0.8211   0.9995   0.9960   0.3822  
0.25       0.8386   0.5524   0.8211   0.9995   0.9960   0.3822  
0.30       0.8386   0.5524   0.8211   0.9995   0.9960   0.3822  
0.35       0.8386   0.5524   0.8211   0.9995   0.9960   0.3822  
0.40       0.8386   0.5524   0.8211   0.9995   0.9960   0.3822  
0.45       0.8386   0.5524   0.8211   0.9995   0.9960   0.3822  
0.50       0.8386   0.5524   0.8211   0.9995   0.9960   0.3822  
0.55       0.8386   0.5524   0.8211   0.9995   0.9960   0.3822  
0.60       0.8386   0.5524   0.8211   0.9995   0.9960   0.3822  
0.65       0.8386   0.5524   0.8211   0.9995   0.9960   0.3822  
0.70       0.8386   0.5524   0.8211   0.9995   0.9960   0.3822  
0.75       0.8386   0.5524   0.8211   0.9995   0.9960   0.3822  
0.80       0.8386   0.5524   0.8211   0.9995   0.9960   0.3822  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8386, F1=0.5524, Normal Recall=0.8211, Normal Precision=0.9995, Attack Recall=0.9960, Attack Precision=0.3822

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
0.15       0.8563   0.7349   0.8215   0.9987   0.9958   0.5823   <--
0.20       0.8563   0.7349   0.8215   0.9987   0.9958   0.5823  
0.25       0.8563   0.7349   0.8215   0.9987   0.9958   0.5823  
0.30       0.8563   0.7349   0.8215   0.9987   0.9958   0.5823  
0.35       0.8563   0.7349   0.8215   0.9987   0.9958   0.5823  
0.40       0.8563   0.7349   0.8215   0.9987   0.9958   0.5823  
0.45       0.8563   0.7349   0.8215   0.9987   0.9958   0.5823  
0.50       0.8563   0.7349   0.8215   0.9987   0.9958   0.5823  
0.55       0.8563   0.7349   0.8215   0.9987   0.9958   0.5823  
0.60       0.8563   0.7349   0.8215   0.9987   0.9958   0.5823  
0.65       0.8563   0.7349   0.8215   0.9987   0.9958   0.5823  
0.70       0.8563   0.7349   0.8215   0.9987   0.9958   0.5823  
0.75       0.8563   0.7349   0.8215   0.9987   0.9958   0.5823  
0.80       0.8563   0.7349   0.8215   0.9987   0.9958   0.5823  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8563, F1=0.7349, Normal Recall=0.8215, Normal Precision=0.9987, Attack Recall=0.9958, Attack Precision=0.5823

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
0.15       0.8733   0.8251   0.8208   0.9978   0.9958   0.7043   <--
0.20       0.8733   0.8251   0.8208   0.9978   0.9958   0.7043  
0.25       0.8733   0.8251   0.8208   0.9978   0.9958   0.7043  
0.30       0.8733   0.8251   0.8208   0.9978   0.9958   0.7043  
0.35       0.8733   0.8251   0.8208   0.9978   0.9958   0.7043  
0.40       0.8733   0.8251   0.8208   0.9978   0.9958   0.7043  
0.45       0.8733   0.8251   0.8208   0.9978   0.9958   0.7043  
0.50       0.8733   0.8251   0.8208   0.9978   0.9958   0.7043  
0.55       0.8733   0.8251   0.8208   0.9978   0.9958   0.7043  
0.60       0.8733   0.8251   0.8208   0.9978   0.9958   0.7043  
0.65       0.8733   0.8251   0.8208   0.9978   0.9958   0.7043  
0.70       0.8733   0.8251   0.8208   0.9978   0.9958   0.7043  
0.75       0.8733   0.8251   0.8208   0.9978   0.9958   0.7043  
0.80       0.8733   0.8251   0.8208   0.9978   0.9958   0.7043  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8733, F1=0.8251, Normal Recall=0.8208, Normal Precision=0.9978, Attack Recall=0.9958, Attack Precision=0.7043

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
0.15       0.8905   0.8792   0.8203   0.9966   0.9958   0.7870   <--
0.20       0.8905   0.8792   0.8203   0.9966   0.9958   0.7870  
0.25       0.8905   0.8792   0.8203   0.9966   0.9958   0.7870  
0.30       0.8905   0.8792   0.8203   0.9966   0.9958   0.7870  
0.35       0.8905   0.8792   0.8203   0.9966   0.9958   0.7870  
0.40       0.8905   0.8792   0.8203   0.9966   0.9958   0.7870  
0.45       0.8905   0.8792   0.8203   0.9966   0.9958   0.7870  
0.50       0.8905   0.8792   0.8203   0.9966   0.9958   0.7870  
0.55       0.8905   0.8792   0.8203   0.9966   0.9958   0.7870  
0.60       0.8905   0.8792   0.8203   0.9966   0.9958   0.7870  
0.65       0.8905   0.8792   0.8203   0.9966   0.9958   0.7870  
0.70       0.8905   0.8792   0.8203   0.9966   0.9958   0.7870  
0.75       0.8905   0.8792   0.8203   0.9966   0.9958   0.7870  
0.80       0.8905   0.8792   0.8203   0.9966   0.9958   0.7870  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8905, F1=0.8792, Normal Recall=0.8203, Normal Precision=0.9966, Attack Recall=0.9958, Attack Precision=0.7870

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
0.15       0.9076   0.9151   0.8194   0.9949   0.9958   0.8464   <--
0.20       0.9076   0.9151   0.8194   0.9949   0.9958   0.8464  
0.25       0.9076   0.9151   0.8194   0.9949   0.9958   0.8464  
0.30       0.9076   0.9151   0.8194   0.9949   0.9958   0.8464  
0.35       0.9076   0.9151   0.8194   0.9949   0.9958   0.8464  
0.40       0.9076   0.9151   0.8194   0.9949   0.9958   0.8464  
0.45       0.9076   0.9151   0.8194   0.9949   0.9958   0.8464  
0.50       0.9076   0.9151   0.8194   0.9949   0.9958   0.8464  
0.55       0.9076   0.9151   0.8194   0.9949   0.9958   0.8464  
0.60       0.9076   0.9151   0.8194   0.9949   0.9958   0.8464  
0.65       0.9076   0.9151   0.8194   0.9949   0.9958   0.8464  
0.70       0.9076   0.9151   0.8194   0.9949   0.9958   0.8464  
0.75       0.9076   0.9151   0.8194   0.9949   0.9958   0.8464  
0.80       0.9076   0.9151   0.8194   0.9949   0.9958   0.8464  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9076, F1=0.9151, Normal Recall=0.8194, Normal Precision=0.9949, Attack Recall=0.9958, Attack Precision=0.8464

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
0.15       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817   <--
0.20       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.25       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.30       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.35       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.40       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.45       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.50       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.55       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.60       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.65       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.70       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.75       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.80       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8383, F1=0.5519, Normal Recall=0.8207, Normal Precision=0.9995, Attack Recall=0.9961, Attack Precision=0.3817

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
0.15       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819   <--
0.20       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.25       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.30       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.35       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.40       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.45       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.50       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.55       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.60       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.65       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.70       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.75       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.80       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8561, F1=0.7346, Normal Recall=0.8211, Normal Precision=0.9987, Attack Recall=0.9958, Attack Precision=0.5819

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
0.15       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039   <--
0.20       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.25       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.30       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.35       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.40       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.45       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.50       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.55       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.60       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.65       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.70       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.75       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.80       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8731, F1=0.8248, Normal Recall=0.8205, Normal Precision=0.9978, Attack Recall=0.9958, Attack Precision=0.7039

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
0.15       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866   <--
0.20       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.25       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.30       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.35       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.40       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.45       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.50       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.55       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.60       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.65       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.70       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.75       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.80       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8903, F1=0.8789, Normal Recall=0.8199, Normal Precision=0.9966, Attack Recall=0.9958, Attack Precision=0.7866

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
0.15       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460   <--
0.20       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.25       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.30       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.35       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.40       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.45       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.50       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.55       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.60       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.65       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.70       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.75       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.80       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9073, F1=0.9148, Normal Recall=0.8188, Normal Precision=0.9949, Attack Recall=0.9958, Attack Precision=0.8460

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
0.15       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817   <--
0.20       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.25       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.30       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.35       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.40       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.45       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.50       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.55       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.60       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.65       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.70       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.75       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
0.80       0.8383   0.5519   0.8207   0.9995   0.9961   0.3817  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8383, F1=0.5519, Normal Recall=0.8207, Normal Precision=0.9995, Attack Recall=0.9961, Attack Precision=0.3817

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
0.15       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819   <--
0.20       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.25       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.30       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.35       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.40       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.45       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.50       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.55       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.60       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.65       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.70       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.75       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
0.80       0.8561   0.7346   0.8211   0.9987   0.9958   0.5819  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8561, F1=0.7346, Normal Recall=0.8211, Normal Precision=0.9987, Attack Recall=0.9958, Attack Precision=0.5819

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
0.15       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039   <--
0.20       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.25       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.30       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.35       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.40       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.45       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.50       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.55       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.60       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.65       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.70       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.75       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
0.80       0.8731   0.8248   0.8205   0.9978   0.9958   0.7039  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8731, F1=0.8248, Normal Recall=0.8205, Normal Precision=0.9978, Attack Recall=0.9958, Attack Precision=0.7039

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
0.15       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866   <--
0.20       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.25       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.30       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.35       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.40       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.45       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.50       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.55       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.60       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.65       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.70       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.75       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
0.80       0.8903   0.8789   0.8199   0.9966   0.9958   0.7866  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8903, F1=0.8789, Normal Recall=0.8199, Normal Precision=0.9966, Attack Recall=0.9958, Attack Precision=0.7866

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
0.15       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460   <--
0.20       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.25       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.30       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.35       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.40       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.45       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.50       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.55       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.60       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.65       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.70       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.75       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
0.80       0.9073   0.9148   0.8188   0.9949   0.9958   0.8460  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9073, F1=0.9148, Normal Recall=0.8188, Normal Precision=0.9949, Attack Recall=0.9958, Attack Precision=0.8460

```

