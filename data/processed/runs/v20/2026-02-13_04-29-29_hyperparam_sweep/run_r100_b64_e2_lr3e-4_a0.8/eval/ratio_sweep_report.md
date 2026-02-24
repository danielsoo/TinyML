# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e2_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-20 04:04:39 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9341 | 0.8668 | 0.7996 | 0.7335 | 0.6662 | 0.5992 | 0.5328 | 0.4664 | 0.3986 | 0.3319 | 0.2651 |
| QAT+Prune only | 0.9147 | 0.9190 | 0.9236 | 0.9289 | 0.9329 | 0.9371 | 0.9420 | 0.9457 | 0.9510 | 0.9550 | 0.9600 |
| QAT+PTQ | 0.9147 | 0.9191 | 0.9234 | 0.9287 | 0.9324 | 0.9364 | 0.9413 | 0.9450 | 0.9501 | 0.9540 | 0.9588 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9147 | 0.9191 | 0.9234 | 0.9287 | 0.9324 | 0.9364 | 0.9413 | 0.9450 | 0.9501 | 0.9540 | 0.9588 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2843 | 0.3460 | 0.3737 | 0.3885 | 0.3981 | 0.4051 | 0.4102 | 0.4136 | 0.4167 | 0.4191 |
| QAT+Prune only | 0.0000 | 0.7035 | 0.8340 | 0.8901 | 0.9196 | 0.9385 | 0.9521 | 0.9612 | 0.9691 | 0.9746 | 0.9796 |
| QAT+PTQ | 0.0000 | 0.7033 | 0.8335 | 0.8897 | 0.9190 | 0.9378 | 0.9515 | 0.9607 | 0.9685 | 0.9740 | 0.9790 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.7033 | 0.8335 | 0.8897 | 0.9190 | 0.9378 | 0.9515 | 0.9607 | 0.9685 | 0.9740 | 0.9790 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9341 | 0.9337 | 0.9332 | 0.9342 | 0.9337 | 0.9333 | 0.9345 | 0.9363 | 0.9326 | 0.9336 | 0.0000 |
| QAT+Prune only | 0.9147 | 0.9144 | 0.9144 | 0.9156 | 0.9148 | 0.9142 | 0.9149 | 0.9124 | 0.9147 | 0.9098 | 0.0000 |
| QAT+PTQ | 0.9147 | 0.9146 | 0.9145 | 0.9157 | 0.9147 | 0.9140 | 0.9151 | 0.9128 | 0.9151 | 0.9099 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9147 | 0.9146 | 0.9145 | 0.9157 | 0.9147 | 0.9140 | 0.9151 | 0.9128 | 0.9151 | 0.9099 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9341 | 0.0000 | 0.0000 | 0.0000 | 0.9341 | 1.0000 |
| 90 | 10 | 299,940 | 0.8668 | 0.3072 | 0.2646 | 0.2843 | 0.9337 | 0.9195 |
| 80 | 20 | 291,350 | 0.7996 | 0.4980 | 0.2651 | 0.3460 | 0.9332 | 0.8355 |
| 70 | 30 | 194,230 | 0.7335 | 0.6333 | 0.2651 | 0.3737 | 0.9342 | 0.7479 |
| 60 | 40 | 145,675 | 0.6662 | 0.7271 | 0.2651 | 0.3885 | 0.9337 | 0.6558 |
| 50 | 50 | 116,540 | 0.5992 | 0.7990 | 0.2651 | 0.3981 | 0.9333 | 0.5595 |
| 40 | 60 | 97,115 | 0.5328 | 0.8585 | 0.2651 | 0.4051 | 0.9345 | 0.4588 |
| 30 | 70 | 83,240 | 0.4664 | 0.9066 | 0.2651 | 0.4102 | 0.9363 | 0.3532 |
| 20 | 80 | 72,835 | 0.3986 | 0.9402 | 0.2651 | 0.4136 | 0.9326 | 0.2408 |
| 10 | 90 | 64,740 | 0.3319 | 0.9729 | 0.2651 | 0.4167 | 0.9336 | 0.1237 |
| 0 | 100 | 58,270 | 0.2651 | 1.0000 | 0.2651 | 0.4191 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9147 | 0.0000 | 0.0000 | 0.0000 | 0.9147 | 1.0000 |
| 90 | 10 | 299,940 | 0.9190 | 0.5550 | 0.9606 | 0.7035 | 0.9144 | 0.9952 |
| 80 | 20 | 291,350 | 0.9236 | 0.7372 | 0.9600 | 0.8340 | 0.9144 | 0.9892 |
| 70 | 30 | 194,230 | 0.9289 | 0.8297 | 0.9600 | 0.8901 | 0.9156 | 0.9816 |
| 60 | 40 | 145,675 | 0.9329 | 0.8825 | 0.9600 | 0.9196 | 0.9148 | 0.9717 |
| 50 | 50 | 116,540 | 0.9371 | 0.9180 | 0.9600 | 0.9385 | 0.9142 | 0.9581 |
| 40 | 60 | 97,115 | 0.9420 | 0.9442 | 0.9600 | 0.9521 | 0.9149 | 0.9385 |
| 30 | 70 | 83,240 | 0.9457 | 0.9624 | 0.9600 | 0.9612 | 0.9124 | 0.9072 |
| 20 | 80 | 72,835 | 0.9510 | 0.9783 | 0.9600 | 0.9691 | 0.9147 | 0.8512 |
| 10 | 90 | 64,740 | 0.9550 | 0.9897 | 0.9600 | 0.9746 | 0.9098 | 0.7165 |
| 0 | 100 | 58,270 | 0.9600 | 1.0000 | 0.9600 | 0.9796 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9147 | 0.0000 | 0.0000 | 0.0000 | 0.9147 | 1.0000 |
| 90 | 10 | 299,940 | 0.9191 | 0.5551 | 0.9594 | 0.7033 | 0.9146 | 0.9951 |
| 80 | 20 | 291,350 | 0.9234 | 0.7372 | 0.9588 | 0.8335 | 0.9145 | 0.9889 |
| 70 | 30 | 194,230 | 0.9287 | 0.8299 | 0.9588 | 0.8897 | 0.9157 | 0.9811 |
| 60 | 40 | 145,675 | 0.9324 | 0.8823 | 0.9588 | 0.9190 | 0.9147 | 0.9709 |
| 50 | 50 | 116,540 | 0.9364 | 0.9177 | 0.9588 | 0.9378 | 0.9140 | 0.9569 |
| 40 | 60 | 97,115 | 0.9413 | 0.9442 | 0.9588 | 0.9515 | 0.9151 | 0.9368 |
| 30 | 70 | 83,240 | 0.9450 | 0.9625 | 0.9588 | 0.9607 | 0.9128 | 0.9048 |
| 20 | 80 | 72,835 | 0.9501 | 0.9783 | 0.9589 | 0.9685 | 0.9151 | 0.8476 |
| 10 | 90 | 64,740 | 0.9540 | 0.9897 | 0.9588 | 0.9740 | 0.9099 | 0.7107 |
| 0 | 100 | 58,270 | 0.9588 | 1.0000 | 0.9588 | 0.9790 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9147 | 0.0000 | 0.0000 | 0.0000 | 0.9147 | 1.0000 |
| 90 | 10 | 299,940 | 0.9191 | 0.5551 | 0.9594 | 0.7033 | 0.9146 | 0.9951 |
| 80 | 20 | 291,350 | 0.9234 | 0.7372 | 0.9588 | 0.8335 | 0.9145 | 0.9889 |
| 70 | 30 | 194,230 | 0.9287 | 0.8299 | 0.9588 | 0.8897 | 0.9157 | 0.9811 |
| 60 | 40 | 145,675 | 0.9324 | 0.8823 | 0.9588 | 0.9190 | 0.9147 | 0.9709 |
| 50 | 50 | 116,540 | 0.9364 | 0.9177 | 0.9588 | 0.9378 | 0.9140 | 0.9569 |
| 40 | 60 | 97,115 | 0.9413 | 0.9442 | 0.9588 | 0.9515 | 0.9151 | 0.9368 |
| 30 | 70 | 83,240 | 0.9450 | 0.9625 | 0.9588 | 0.9607 | 0.9128 | 0.9048 |
| 20 | 80 | 72,835 | 0.9501 | 0.9783 | 0.9589 | 0.9685 | 0.9151 | 0.8476 |
| 10 | 90 | 64,740 | 0.9540 | 0.9897 | 0.9588 | 0.9740 | 0.9099 | 0.7107 |
| 0 | 100 | 58,270 | 0.9588 | 1.0000 | 0.9588 | 0.9790 | 0.0000 | 0.0000 |


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
0.15       0.8672   0.2877   0.9337   0.9199   0.2683   0.3101   <--
0.20       0.8672   0.2877   0.9337   0.9199   0.2683   0.3101  
0.25       0.8672   0.2877   0.9337   0.9199   0.2683   0.3101  
0.30       0.8672   0.2877   0.9337   0.9199   0.2683   0.3101  
0.35       0.8672   0.2877   0.9337   0.9199   0.2683   0.3101  
0.40       0.8672   0.2877   0.9337   0.9199   0.2683   0.3101  
0.45       0.8672   0.2877   0.9337   0.9199   0.2683   0.3101  
0.50       0.8672   0.2877   0.9337   0.9199   0.2683   0.3101  
0.55       0.8672   0.2877   0.9337   0.9199   0.2683   0.3101  
0.60       0.8672   0.2877   0.9337   0.9199   0.2683   0.3101  
0.65       0.8672   0.2877   0.9337   0.9199   0.2683   0.3101  
0.70       0.8672   0.2877   0.9337   0.9199   0.2683   0.3101  
0.75       0.8672   0.2877   0.9337   0.9199   0.2683   0.3101  
0.80       0.8672   0.2877   0.9337   0.9199   0.2683   0.3101  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8672, F1=0.2877, Normal Recall=0.9337, Normal Precision=0.9199, Attack Recall=0.2683, Attack Precision=0.3101

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
0.15       0.7998   0.3463   0.9335   0.8356   0.2651   0.4992   <--
0.20       0.7998   0.3463   0.9335   0.8356   0.2651   0.4992  
0.25       0.7998   0.3463   0.9335   0.8356   0.2651   0.4992  
0.30       0.7998   0.3463   0.9335   0.8356   0.2651   0.4992  
0.35       0.7998   0.3463   0.9335   0.8356   0.2651   0.4992  
0.40       0.7998   0.3463   0.9335   0.8356   0.2651   0.4992  
0.45       0.7998   0.3463   0.9335   0.8356   0.2651   0.4992  
0.50       0.7998   0.3463   0.9335   0.8356   0.2651   0.4992  
0.55       0.7998   0.3463   0.9335   0.8356   0.2651   0.4992  
0.60       0.7998   0.3463   0.9335   0.8356   0.2651   0.4992  
0.65       0.7998   0.3463   0.9335   0.8356   0.2651   0.4992  
0.70       0.7998   0.3463   0.9335   0.8356   0.2651   0.4992  
0.75       0.7998   0.3463   0.9335   0.8356   0.2651   0.4992  
0.80       0.7998   0.3463   0.9335   0.8356   0.2651   0.4992  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7998, F1=0.3463, Normal Recall=0.9335, Normal Precision=0.8356, Attack Recall=0.2651, Attack Precision=0.4992

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
0.15       0.7337   0.3739   0.9345   0.7479   0.2651   0.6342   <--
0.20       0.7337   0.3739   0.9345   0.7479   0.2651   0.6342  
0.25       0.7337   0.3739   0.9345   0.7479   0.2651   0.6342  
0.30       0.7337   0.3739   0.9345   0.7479   0.2651   0.6342  
0.35       0.7337   0.3739   0.9345   0.7479   0.2651   0.6342  
0.40       0.7337   0.3739   0.9345   0.7479   0.2651   0.6342  
0.45       0.7337   0.3739   0.9345   0.7479   0.2651   0.6342  
0.50       0.7337   0.3739   0.9345   0.7479   0.2651   0.6342  
0.55       0.7337   0.3739   0.9345   0.7479   0.2651   0.6342  
0.60       0.7337   0.3739   0.9345   0.7479   0.2651   0.6342  
0.65       0.7337   0.3739   0.9345   0.7479   0.2651   0.6342  
0.70       0.7337   0.3739   0.9345   0.7479   0.2651   0.6342  
0.75       0.7337   0.3739   0.9345   0.7479   0.2651   0.6342  
0.80       0.7337   0.3739   0.9345   0.7479   0.2651   0.6342  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7337, F1=0.3739, Normal Recall=0.9345, Normal Precision=0.7479, Attack Recall=0.2651, Attack Precision=0.6342

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
0.15       0.6665   0.3887   0.9342   0.6560   0.2651   0.7286   <--
0.20       0.6665   0.3887   0.9342   0.6560   0.2651   0.7286  
0.25       0.6665   0.3887   0.9342   0.6560   0.2651   0.7286  
0.30       0.6665   0.3887   0.9342   0.6560   0.2651   0.7286  
0.35       0.6665   0.3887   0.9342   0.6560   0.2651   0.7286  
0.40       0.6665   0.3887   0.9342   0.6560   0.2651   0.7286  
0.45       0.6665   0.3887   0.9342   0.6560   0.2651   0.7286  
0.50       0.6665   0.3887   0.9342   0.6560   0.2651   0.7286  
0.55       0.6665   0.3887   0.9342   0.6560   0.2651   0.7286  
0.60       0.6665   0.3887   0.9342   0.6560   0.2651   0.7286  
0.65       0.6665   0.3887   0.9342   0.6560   0.2651   0.7286  
0.70       0.6665   0.3887   0.9342   0.6560   0.2651   0.7286  
0.75       0.6665   0.3887   0.9342   0.6560   0.2651   0.7286  
0.80       0.6665   0.3887   0.9342   0.6560   0.2651   0.7286  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6665, F1=0.3887, Normal Recall=0.9342, Normal Precision=0.6560, Attack Recall=0.2651, Attack Precision=0.7286

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
0.15       0.5998   0.3985   0.9345   0.5598   0.2651   0.8019   <--
0.20       0.5998   0.3985   0.9345   0.5598   0.2651   0.8019  
0.25       0.5998   0.3985   0.9345   0.5598   0.2651   0.8019  
0.30       0.5998   0.3985   0.9345   0.5598   0.2651   0.8019  
0.35       0.5998   0.3985   0.9345   0.5598   0.2651   0.8019  
0.40       0.5998   0.3985   0.9345   0.5598   0.2651   0.8019  
0.45       0.5998   0.3985   0.9345   0.5598   0.2651   0.8019  
0.50       0.5998   0.3985   0.9345   0.5598   0.2651   0.8019  
0.55       0.5998   0.3985   0.9345   0.5598   0.2651   0.8019  
0.60       0.5998   0.3985   0.9345   0.5598   0.2651   0.8019  
0.65       0.5998   0.3985   0.9345   0.5598   0.2651   0.8019  
0.70       0.5998   0.3985   0.9345   0.5598   0.2651   0.8019  
0.75       0.5998   0.3985   0.9345   0.5598   0.2651   0.8019  
0.80       0.5998   0.3985   0.9345   0.5598   0.2651   0.8019  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5998, F1=0.3985, Normal Recall=0.9345, Normal Precision=0.5598, Attack Recall=0.2651, Attack Precision=0.8019

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
0.15       0.9191   0.7039   0.9144   0.9953   0.9615   0.5552   <--
0.20       0.9191   0.7039   0.9144   0.9953   0.9615   0.5552  
0.25       0.9191   0.7039   0.9144   0.9953   0.9615   0.5552  
0.30       0.9191   0.7039   0.9144   0.9953   0.9615   0.5552  
0.35       0.9191   0.7039   0.9144   0.9953   0.9615   0.5552  
0.40       0.9191   0.7039   0.9144   0.9953   0.9615   0.5552  
0.45       0.9191   0.7039   0.9144   0.9953   0.9615   0.5552  
0.50       0.9191   0.7039   0.9144   0.9953   0.9615   0.5552  
0.55       0.9191   0.7039   0.9144   0.9953   0.9615   0.5552  
0.60       0.9191   0.7039   0.9144   0.9953   0.9615   0.5552  
0.65       0.9191   0.7039   0.9144   0.9953   0.9615   0.5552  
0.70       0.9191   0.7039   0.9144   0.9953   0.9615   0.5552  
0.75       0.9191   0.7039   0.9144   0.9953   0.9615   0.5552  
0.80       0.9191   0.7039   0.9144   0.9953   0.9615   0.5552  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9191, F1=0.7039, Normal Recall=0.9144, Normal Precision=0.9953, Attack Recall=0.9615, Attack Precision=0.5552

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
0.15       0.9237   0.8342   0.9146   0.9892   0.9600   0.7376   <--
0.20       0.9237   0.8342   0.9146   0.9892   0.9600   0.7376  
0.25       0.9237   0.8342   0.9146   0.9892   0.9600   0.7376  
0.30       0.9237   0.8342   0.9146   0.9892   0.9600   0.7376  
0.35       0.9237   0.8342   0.9146   0.9892   0.9600   0.7376  
0.40       0.9237   0.8342   0.9146   0.9892   0.9600   0.7376  
0.45       0.9237   0.8342   0.9146   0.9892   0.9600   0.7376  
0.50       0.9237   0.8342   0.9146   0.9892   0.9600   0.7376  
0.55       0.9237   0.8342   0.9146   0.9892   0.9600   0.7376  
0.60       0.9237   0.8342   0.9146   0.9892   0.9600   0.7376  
0.65       0.9237   0.8342   0.9146   0.9892   0.9600   0.7376  
0.70       0.9237   0.8342   0.9146   0.9892   0.9600   0.7376  
0.75       0.9237   0.8342   0.9146   0.9892   0.9600   0.7376  
0.80       0.9237   0.8342   0.9146   0.9892   0.9600   0.7376  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9237, F1=0.8342, Normal Recall=0.9146, Normal Precision=0.9892, Attack Recall=0.9600, Attack Precision=0.7376

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
0.15       0.9283   0.8893   0.9147   0.9816   0.9600   0.8283   <--
0.20       0.9283   0.8893   0.9147   0.9816   0.9600   0.8283  
0.25       0.9283   0.8893   0.9147   0.9816   0.9600   0.8283  
0.30       0.9283   0.8893   0.9147   0.9816   0.9600   0.8283  
0.35       0.9283   0.8893   0.9147   0.9816   0.9600   0.8283  
0.40       0.9283   0.8893   0.9147   0.9816   0.9600   0.8283  
0.45       0.9283   0.8893   0.9147   0.9816   0.9600   0.8283  
0.50       0.9283   0.8893   0.9147   0.9816   0.9600   0.8283  
0.55       0.9283   0.8893   0.9147   0.9816   0.9600   0.8283  
0.60       0.9283   0.8893   0.9147   0.9816   0.9600   0.8283  
0.65       0.9283   0.8893   0.9147   0.9816   0.9600   0.8283  
0.70       0.9283   0.8893   0.9147   0.9816   0.9600   0.8283  
0.75       0.9283   0.8893   0.9147   0.9816   0.9600   0.8283  
0.80       0.9283   0.8893   0.9147   0.9816   0.9600   0.8283  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9283, F1=0.8893, Normal Recall=0.9147, Normal Precision=0.9816, Attack Recall=0.9600, Attack Precision=0.8283

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
0.15       0.9331   0.9198   0.9151   0.9717   0.9600   0.8829   <--
0.20       0.9331   0.9198   0.9151   0.9717   0.9600   0.8829  
0.25       0.9331   0.9198   0.9151   0.9717   0.9600   0.8829  
0.30       0.9331   0.9198   0.9151   0.9717   0.9600   0.8829  
0.35       0.9331   0.9198   0.9151   0.9717   0.9600   0.8829  
0.40       0.9331   0.9198   0.9151   0.9717   0.9600   0.8829  
0.45       0.9331   0.9198   0.9151   0.9717   0.9600   0.8829  
0.50       0.9331   0.9198   0.9151   0.9717   0.9600   0.8829  
0.55       0.9331   0.9198   0.9151   0.9717   0.9600   0.8829  
0.60       0.9331   0.9198   0.9151   0.9717   0.9600   0.8829  
0.65       0.9331   0.9198   0.9151   0.9717   0.9600   0.8829  
0.70       0.9331   0.9198   0.9151   0.9717   0.9600   0.8829  
0.75       0.9331   0.9198   0.9151   0.9717   0.9600   0.8829  
0.80       0.9331   0.9198   0.9151   0.9717   0.9600   0.8829  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9331, F1=0.9198, Normal Recall=0.9151, Normal Precision=0.9717, Attack Recall=0.9600, Attack Precision=0.8829

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
0.15       0.9375   0.9389   0.9151   0.9581   0.9600   0.9187   <--
0.20       0.9375   0.9389   0.9151   0.9581   0.9600   0.9187  
0.25       0.9375   0.9389   0.9151   0.9581   0.9600   0.9187  
0.30       0.9375   0.9389   0.9151   0.9581   0.9600   0.9187  
0.35       0.9375   0.9389   0.9151   0.9581   0.9600   0.9187  
0.40       0.9375   0.9389   0.9151   0.9581   0.9600   0.9187  
0.45       0.9375   0.9389   0.9151   0.9581   0.9600   0.9187  
0.50       0.9375   0.9389   0.9151   0.9581   0.9600   0.9187  
0.55       0.9375   0.9389   0.9151   0.9581   0.9600   0.9187  
0.60       0.9375   0.9389   0.9151   0.9581   0.9600   0.9187  
0.65       0.9375   0.9389   0.9151   0.9581   0.9600   0.9187  
0.70       0.9375   0.9389   0.9151   0.9581   0.9600   0.9187  
0.75       0.9375   0.9389   0.9151   0.9581   0.9600   0.9187  
0.80       0.9375   0.9389   0.9151   0.9581   0.9600   0.9187  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9375, F1=0.9389, Normal Recall=0.9151, Normal Precision=0.9581, Attack Recall=0.9600, Attack Precision=0.9187

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
0.15       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554   <--
0.20       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.25       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.30       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.35       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.40       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.45       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.50       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.55       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.60       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.65       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.70       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.75       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.80       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9191, F1=0.7037, Normal Recall=0.9146, Normal Precision=0.9952, Attack Recall=0.9603, Attack Precision=0.5554

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
0.15       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378   <--
0.20       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.25       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.30       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.35       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.40       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.45       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.50       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.55       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.60       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.65       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.70       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.75       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.80       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9236, F1=0.8340, Normal Recall=0.9148, Normal Precision=0.9889, Attack Recall=0.9588, Attack Precision=0.7378

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
0.15       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282   <--
0.20       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.25       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.30       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.35       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.40       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.45       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.50       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.55       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.60       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.65       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.70       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.75       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.80       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9280, F1=0.8888, Normal Recall=0.9148, Normal Precision=0.9811, Attack Recall=0.9588, Attack Precision=0.8282

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
0.15       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828   <--
0.20       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.25       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.30       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.35       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.40       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.45       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.50       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.55       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.60       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.65       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.70       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.75       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.80       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9326, F1=0.9193, Normal Recall=0.9152, Normal Precision=0.9709, Attack Recall=0.9588, Attack Precision=0.8828

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
0.15       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189   <--
0.20       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.25       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.30       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.35       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.40       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.45       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.50       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.55       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.60       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.65       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.70       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.75       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.80       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9371, F1=0.9384, Normal Recall=0.9153, Normal Precision=0.9570, Attack Recall=0.9588, Attack Precision=0.9189

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
0.15       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554   <--
0.20       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.25       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.30       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.35       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.40       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.45       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.50       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.55       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.60       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.65       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.70       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.75       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
0.80       0.9191   0.7037   0.9146   0.9952   0.9603   0.5554  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9191, F1=0.7037, Normal Recall=0.9146, Normal Precision=0.9952, Attack Recall=0.9603, Attack Precision=0.5554

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
0.15       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378   <--
0.20       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.25       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.30       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.35       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.40       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.45       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.50       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.55       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.60       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.65       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.70       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.75       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
0.80       0.9236   0.8340   0.9148   0.9889   0.9588   0.7378  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9236, F1=0.8340, Normal Recall=0.9148, Normal Precision=0.9889, Attack Recall=0.9588, Attack Precision=0.7378

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
0.15       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282   <--
0.20       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.25       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.30       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.35       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.40       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.45       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.50       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.55       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.60       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.65       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.70       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.75       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
0.80       0.9280   0.8888   0.9148   0.9811   0.9588   0.8282  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9280, F1=0.8888, Normal Recall=0.9148, Normal Precision=0.9811, Attack Recall=0.9588, Attack Precision=0.8282

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
0.15       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828   <--
0.20       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.25       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.30       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.35       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.40       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.45       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.50       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.55       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.60       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.65       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.70       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.75       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
0.80       0.9326   0.9193   0.9152   0.9709   0.9588   0.8828  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9326, F1=0.9193, Normal Recall=0.9152, Normal Precision=0.9709, Attack Recall=0.9588, Attack Precision=0.8828

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
0.15       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189   <--
0.20       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.25       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.30       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.35       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.40       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.45       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.50       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.55       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.60       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.65       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.70       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.75       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
0.80       0.9371   0.9384   0.9153   0.9570   0.9588   0.9189  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9371, F1=0.9384, Normal Recall=0.9153, Normal Precision=0.9570, Attack Recall=0.9588, Attack Precision=0.9189

```

