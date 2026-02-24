# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e2_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-17 20:11:53 |

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
| Original (TFLite) | 0.7565 | 0.7650 | 0.7746 | 0.7855 | 0.7945 | 0.8044 | 0.8142 | 0.8250 | 0.8337 | 0.8453 | 0.8550 |
| QAT+Prune only | 0.9419 | 0.9204 | 0.8974 | 0.8752 | 0.8530 | 0.8293 | 0.8080 | 0.7856 | 0.7619 | 0.7402 | 0.7178 |
| QAT+PTQ | 0.9423 | 0.9209 | 0.8981 | 0.8759 | 0.8538 | 0.8301 | 0.8089 | 0.7867 | 0.7630 | 0.7414 | 0.7191 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9423 | 0.9209 | 0.8981 | 0.8759 | 0.8538 | 0.8301 | 0.8089 | 0.7867 | 0.7630 | 0.7414 | 0.7191 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.4215 | 0.6028 | 0.7051 | 0.7690 | 0.8138 | 0.8466 | 0.8725 | 0.8916 | 0.9086 | 0.9218 |
| QAT+Prune only | 0.0000 | 0.6440 | 0.7368 | 0.7753 | 0.7962 | 0.8079 | 0.8177 | 0.8242 | 0.8283 | 0.8326 | 0.8357 |
| QAT+PTQ | 0.0000 | 0.6460 | 0.7383 | 0.7766 | 0.7974 | 0.8089 | 0.8187 | 0.8252 | 0.8292 | 0.8334 | 0.8366 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.6460 | 0.7383 | 0.7766 | 0.7974 | 0.8089 | 0.8187 | 0.8252 | 0.8292 | 0.8334 | 0.8366 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7565 | 0.7548 | 0.7545 | 0.7557 | 0.7542 | 0.7538 | 0.7529 | 0.7552 | 0.7485 | 0.7578 | 0.0000 |
| QAT+Prune only | 0.9419 | 0.9426 | 0.9424 | 0.9426 | 0.9431 | 0.9408 | 0.9433 | 0.9439 | 0.9382 | 0.9418 | 0.0000 |
| QAT+PTQ | 0.9423 | 0.9430 | 0.9428 | 0.9431 | 0.9436 | 0.9412 | 0.9437 | 0.9445 | 0.9387 | 0.9419 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9423 | 0.9430 | 0.9428 | 0.9431 | 0.9436 | 0.9412 | 0.9437 | 0.9445 | 0.9387 | 0.9419 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7565 | 0.0000 | 0.0000 | 0.0000 | 0.7565 | 1.0000 |
| 90 | 10 | 299,940 | 0.7650 | 0.2795 | 0.8561 | 0.4215 | 0.7548 | 0.9793 |
| 80 | 20 | 291,350 | 0.7746 | 0.4655 | 0.8550 | 0.6028 | 0.7545 | 0.9542 |
| 70 | 30 | 194,230 | 0.7855 | 0.6000 | 0.8550 | 0.7051 | 0.7557 | 0.9240 |
| 60 | 40 | 145,675 | 0.7945 | 0.6987 | 0.8550 | 0.7690 | 0.7542 | 0.8864 |
| 50 | 50 | 116,540 | 0.8044 | 0.7764 | 0.8550 | 0.8138 | 0.7538 | 0.8387 |
| 40 | 60 | 97,115 | 0.8142 | 0.8385 | 0.8550 | 0.8466 | 0.7529 | 0.7759 |
| 30 | 70 | 83,240 | 0.8250 | 0.8907 | 0.8550 | 0.8725 | 0.7552 | 0.6906 |
| 20 | 80 | 72,835 | 0.8337 | 0.9315 | 0.8550 | 0.8916 | 0.7485 | 0.5634 |
| 10 | 90 | 64,740 | 0.8453 | 0.9695 | 0.8550 | 0.9086 | 0.7578 | 0.3673 |
| 0 | 100 | 58,270 | 0.8550 | 1.0000 | 0.8550 | 0.9218 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9419 | 0.0000 | 0.0000 | 0.0000 | 0.9419 | 1.0000 |
| 90 | 10 | 299,940 | 0.9204 | 0.5823 | 0.7202 | 0.6440 | 0.9426 | 0.9681 |
| 80 | 20 | 291,350 | 0.8974 | 0.7569 | 0.7178 | 0.7368 | 0.9424 | 0.9303 |
| 70 | 30 | 194,230 | 0.8752 | 0.8428 | 0.7178 | 0.7753 | 0.9426 | 0.8863 |
| 60 | 40 | 145,675 | 0.8530 | 0.8937 | 0.7178 | 0.7962 | 0.9431 | 0.8337 |
| 50 | 50 | 116,540 | 0.8293 | 0.9238 | 0.7178 | 0.8079 | 0.9408 | 0.7692 |
| 40 | 60 | 97,115 | 0.8080 | 0.9500 | 0.7178 | 0.8177 | 0.9433 | 0.6903 |
| 30 | 70 | 83,240 | 0.7856 | 0.9676 | 0.7178 | 0.8242 | 0.9439 | 0.5891 |
| 20 | 80 | 72,835 | 0.7619 | 0.9789 | 0.7178 | 0.8283 | 0.9382 | 0.4539 |
| 10 | 90 | 64,740 | 0.7402 | 0.9911 | 0.7178 | 0.8326 | 0.9418 | 0.2705 |
| 0 | 100 | 58,270 | 0.7178 | 1.0000 | 0.7178 | 0.8357 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9423 | 0.0000 | 0.0000 | 0.0000 | 0.9423 | 1.0000 |
| 90 | 10 | 299,940 | 0.9209 | 0.5846 | 0.7217 | 0.6460 | 0.9430 | 0.9683 |
| 80 | 20 | 291,350 | 0.8981 | 0.7586 | 0.7191 | 0.7383 | 0.9428 | 0.9307 |
| 70 | 30 | 194,230 | 0.8759 | 0.8441 | 0.7191 | 0.7766 | 0.9431 | 0.8868 |
| 60 | 40 | 145,675 | 0.8538 | 0.8948 | 0.7191 | 0.7974 | 0.9436 | 0.8344 |
| 50 | 50 | 116,540 | 0.8301 | 0.9244 | 0.7191 | 0.8089 | 0.9412 | 0.7701 |
| 40 | 60 | 97,115 | 0.8089 | 0.9504 | 0.7191 | 0.8187 | 0.9437 | 0.6913 |
| 30 | 70 | 83,240 | 0.7867 | 0.9680 | 0.7191 | 0.8252 | 0.9445 | 0.5903 |
| 20 | 80 | 72,835 | 0.7630 | 0.9791 | 0.7191 | 0.8292 | 0.9387 | 0.4552 |
| 10 | 90 | 64,740 | 0.7414 | 0.9911 | 0.7191 | 0.8334 | 0.9419 | 0.2714 |
| 0 | 100 | 58,270 | 0.7191 | 1.0000 | 0.7191 | 0.8366 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9423 | 0.0000 | 0.0000 | 0.0000 | 0.9423 | 1.0000 |
| 90 | 10 | 299,940 | 0.9209 | 0.5846 | 0.7217 | 0.6460 | 0.9430 | 0.9683 |
| 80 | 20 | 291,350 | 0.8981 | 0.7586 | 0.7191 | 0.7383 | 0.9428 | 0.9307 |
| 70 | 30 | 194,230 | 0.8759 | 0.8441 | 0.7191 | 0.7766 | 0.9431 | 0.8868 |
| 60 | 40 | 145,675 | 0.8538 | 0.8948 | 0.7191 | 0.7974 | 0.9436 | 0.8344 |
| 50 | 50 | 116,540 | 0.8301 | 0.9244 | 0.7191 | 0.8089 | 0.9412 | 0.7701 |
| 40 | 60 | 97,115 | 0.8089 | 0.9504 | 0.7191 | 0.8187 | 0.9437 | 0.6913 |
| 30 | 70 | 83,240 | 0.7867 | 0.9680 | 0.7191 | 0.8252 | 0.9445 | 0.5903 |
| 20 | 80 | 72,835 | 0.7630 | 0.9791 | 0.7191 | 0.8292 | 0.9387 | 0.4552 |
| 10 | 90 | 64,740 | 0.7414 | 0.9911 | 0.7191 | 0.8334 | 0.9419 | 0.2714 |
| 0 | 100 | 58,270 | 0.7191 | 1.0000 | 0.7191 | 0.8366 | 0.0000 | 0.0000 |


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
0.15       0.7647   0.4207   0.7548   0.9790   0.8541   0.2791   <--
0.20       0.7647   0.4207   0.7548   0.9790   0.8541   0.2791  
0.25       0.7647   0.4207   0.7548   0.9790   0.8541   0.2791  
0.30       0.7647   0.4207   0.7548   0.9790   0.8541   0.2791  
0.35       0.7647   0.4207   0.7548   0.9790   0.8541   0.2791  
0.40       0.7647   0.4207   0.7548   0.9790   0.8541   0.2791  
0.45       0.7647   0.4207   0.7548   0.9790   0.8541   0.2791  
0.50       0.7647   0.4207   0.7548   0.9790   0.8541   0.2791  
0.55       0.7647   0.4207   0.7548   0.9790   0.8541   0.2791  
0.60       0.7647   0.4207   0.7548   0.9790   0.8541   0.2791  
0.65       0.7647   0.4207   0.7548   0.9790   0.8541   0.2791  
0.70       0.7647   0.4207   0.7548   0.9790   0.8541   0.2791  
0.75       0.7647   0.4207   0.7548   0.9790   0.8541   0.2791  
0.80       0.7647   0.4207   0.7548   0.9790   0.8541   0.2791  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7647, F1=0.4207, Normal Recall=0.7548, Normal Precision=0.9790, Attack Recall=0.8541, Attack Precision=0.2791

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
0.15       0.7751   0.6033   0.7551   0.9542   0.8550   0.4661   <--
0.20       0.7751   0.6033   0.7551   0.9542   0.8550   0.4661  
0.25       0.7751   0.6033   0.7551   0.9542   0.8550   0.4661  
0.30       0.7751   0.6033   0.7551   0.9542   0.8550   0.4661  
0.35       0.7751   0.6033   0.7551   0.9542   0.8550   0.4661  
0.40       0.7751   0.6033   0.7551   0.9542   0.8550   0.4661  
0.45       0.7751   0.6033   0.7551   0.9542   0.8550   0.4661  
0.50       0.7751   0.6033   0.7551   0.9542   0.8550   0.4661  
0.55       0.7751   0.6033   0.7551   0.9542   0.8550   0.4661  
0.60       0.7751   0.6033   0.7551   0.9542   0.8550   0.4661  
0.65       0.7751   0.6033   0.7551   0.9542   0.8550   0.4661  
0.70       0.7751   0.6033   0.7551   0.9542   0.8550   0.4661  
0.75       0.7751   0.6033   0.7551   0.9542   0.8550   0.4661  
0.80       0.7751   0.6033   0.7551   0.9542   0.8550   0.4661  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7751, F1=0.6033, Normal Recall=0.7551, Normal Precision=0.9542, Attack Recall=0.8550, Attack Precision=0.4661

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
0.15       0.7864   0.7060   0.7570   0.9241   0.8550   0.6013   <--
0.20       0.7864   0.7060   0.7570   0.9241   0.8550   0.6013  
0.25       0.7864   0.7060   0.7570   0.9241   0.8550   0.6013  
0.30       0.7864   0.7060   0.7570   0.9241   0.8550   0.6013  
0.35       0.7864   0.7060   0.7570   0.9241   0.8550   0.6013  
0.40       0.7864   0.7060   0.7570   0.9241   0.8550   0.6013  
0.45       0.7864   0.7060   0.7570   0.9241   0.8550   0.6013  
0.50       0.7864   0.7060   0.7570   0.9241   0.8550   0.6013  
0.55       0.7864   0.7060   0.7570   0.9241   0.8550   0.6013  
0.60       0.7864   0.7060   0.7570   0.9241   0.8550   0.6013  
0.65       0.7864   0.7060   0.7570   0.9241   0.8550   0.6013  
0.70       0.7864   0.7060   0.7570   0.9241   0.8550   0.6013  
0.75       0.7864   0.7060   0.7570   0.9241   0.8550   0.6013  
0.80       0.7864   0.7060   0.7570   0.9241   0.8550   0.6013  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7864, F1=0.7060, Normal Recall=0.7570, Normal Precision=0.9241, Attack Recall=0.8550, Attack Precision=0.6013

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
0.15       0.7956   0.7699   0.7560   0.8866   0.8550   0.7002   <--
0.20       0.7956   0.7699   0.7560   0.8866   0.8550   0.7002  
0.25       0.7956   0.7699   0.7560   0.8866   0.8550   0.7002  
0.30       0.7956   0.7699   0.7560   0.8866   0.8550   0.7002  
0.35       0.7956   0.7699   0.7560   0.8866   0.8550   0.7002  
0.40       0.7956   0.7699   0.7560   0.8866   0.8550   0.7002  
0.45       0.7956   0.7699   0.7560   0.8866   0.8550   0.7002  
0.50       0.7956   0.7699   0.7560   0.8866   0.8550   0.7002  
0.55       0.7956   0.7699   0.7560   0.8866   0.8550   0.7002  
0.60       0.7956   0.7699   0.7560   0.8866   0.8550   0.7002  
0.65       0.7956   0.7699   0.7560   0.8866   0.8550   0.7002  
0.70       0.7956   0.7699   0.7560   0.8866   0.8550   0.7002  
0.75       0.7956   0.7699   0.7560   0.8866   0.8550   0.7002  
0.80       0.7956   0.7699   0.7560   0.8866   0.8550   0.7002  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7956, F1=0.7699, Normal Recall=0.7560, Normal Precision=0.8866, Attack Recall=0.8550, Attack Precision=0.7002

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
0.15       0.8062   0.8152   0.7574   0.8393   0.8550   0.7789   <--
0.20       0.8062   0.8152   0.7574   0.8393   0.8550   0.7789  
0.25       0.8062   0.8152   0.7574   0.8393   0.8550   0.7789  
0.30       0.8062   0.8152   0.7574   0.8393   0.8550   0.7789  
0.35       0.8062   0.8152   0.7574   0.8393   0.8550   0.7789  
0.40       0.8062   0.8152   0.7574   0.8393   0.8550   0.7789  
0.45       0.8062   0.8152   0.7574   0.8393   0.8550   0.7789  
0.50       0.8062   0.8152   0.7574   0.8393   0.8550   0.7789  
0.55       0.8062   0.8152   0.7574   0.8393   0.8550   0.7789  
0.60       0.8062   0.8152   0.7574   0.8393   0.8550   0.7789  
0.65       0.8062   0.8152   0.7574   0.8393   0.8550   0.7789  
0.70       0.8062   0.8152   0.7574   0.8393   0.8550   0.7789  
0.75       0.8062   0.8152   0.7574   0.8393   0.8550   0.7789  
0.80       0.8062   0.8152   0.7574   0.8393   0.8550   0.7789  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8062, F1=0.8152, Normal Recall=0.7574, Normal Precision=0.8393, Attack Recall=0.8550, Attack Precision=0.7789

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
0.15       0.9203   0.6436   0.9426   0.9680   0.7196   0.5821   <--
0.20       0.9203   0.6436   0.9426   0.9680   0.7196   0.5821  
0.25       0.9203   0.6436   0.9426   0.9680   0.7196   0.5821  
0.30       0.9203   0.6436   0.9426   0.9680   0.7196   0.5821  
0.35       0.9203   0.6436   0.9426   0.9680   0.7196   0.5821  
0.40       0.9203   0.6436   0.9426   0.9680   0.7196   0.5821  
0.45       0.9203   0.6436   0.9426   0.9680   0.7196   0.5821  
0.50       0.9203   0.6436   0.9426   0.9680   0.7196   0.5821  
0.55       0.9203   0.6436   0.9426   0.9680   0.7196   0.5821  
0.60       0.9203   0.6436   0.9426   0.9680   0.7196   0.5821  
0.65       0.9203   0.6436   0.9426   0.9680   0.7196   0.5821  
0.70       0.9203   0.6436   0.9426   0.9680   0.7196   0.5821  
0.75       0.9203   0.6436   0.9426   0.9680   0.7196   0.5821  
0.80       0.9203   0.6436   0.9426   0.9680   0.7196   0.5821  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9203, F1=0.6436, Normal Recall=0.9426, Normal Precision=0.9680, Attack Recall=0.7196, Attack Precision=0.5821

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
0.15       0.8977   0.7373   0.9427   0.9304   0.7178   0.7580   <--
0.20       0.8977   0.7373   0.9427   0.9304   0.7178   0.7580  
0.25       0.8977   0.7373   0.9427   0.9304   0.7178   0.7580  
0.30       0.8977   0.7373   0.9427   0.9304   0.7178   0.7580  
0.35       0.8977   0.7373   0.9427   0.9304   0.7178   0.7580  
0.40       0.8977   0.7373   0.9427   0.9304   0.7178   0.7580  
0.45       0.8977   0.7373   0.9427   0.9304   0.7178   0.7580  
0.50       0.8977   0.7373   0.9427   0.9304   0.7178   0.7580  
0.55       0.8977   0.7373   0.9427   0.9304   0.7178   0.7580  
0.60       0.8977   0.7373   0.9427   0.9304   0.7178   0.7580  
0.65       0.8977   0.7373   0.9427   0.9304   0.7178   0.7580  
0.70       0.8977   0.7373   0.9427   0.9304   0.7178   0.7580  
0.75       0.8977   0.7373   0.9427   0.9304   0.7178   0.7580  
0.80       0.8977   0.7373   0.9427   0.9304   0.7178   0.7580  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8977, F1=0.7373, Normal Recall=0.9427, Normal Precision=0.9304, Attack Recall=0.7178, Attack Precision=0.7580

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
0.15       0.8748   0.7748   0.9421   0.8862   0.7178   0.8416   <--
0.20       0.8748   0.7748   0.9421   0.8862   0.7178   0.8416  
0.25       0.8748   0.7748   0.9421   0.8862   0.7178   0.8416  
0.30       0.8748   0.7748   0.9421   0.8862   0.7178   0.8416  
0.35       0.8748   0.7748   0.9421   0.8862   0.7178   0.8416  
0.40       0.8748   0.7748   0.9421   0.8862   0.7178   0.8416  
0.45       0.8748   0.7748   0.9421   0.8862   0.7178   0.8416  
0.50       0.8748   0.7748   0.9421   0.8862   0.7178   0.8416  
0.55       0.8748   0.7748   0.9421   0.8862   0.7178   0.8416  
0.60       0.8748   0.7748   0.9421   0.8862   0.7178   0.8416  
0.65       0.8748   0.7748   0.9421   0.8862   0.7178   0.8416  
0.70       0.8748   0.7748   0.9421   0.8862   0.7178   0.8416  
0.75       0.8748   0.7748   0.9421   0.8862   0.7178   0.8416  
0.80       0.8748   0.7748   0.9421   0.8862   0.7178   0.8416  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8748, F1=0.7748, Normal Recall=0.9421, Normal Precision=0.8862, Attack Recall=0.7178, Attack Precision=0.8416

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
0.15       0.8520   0.7951   0.9415   0.8335   0.7178   0.8911   <--
0.20       0.8520   0.7951   0.9415   0.8335   0.7178   0.8911  
0.25       0.8520   0.7951   0.9415   0.8335   0.7178   0.8911  
0.30       0.8520   0.7951   0.9415   0.8335   0.7178   0.8911  
0.35       0.8520   0.7951   0.9415   0.8335   0.7178   0.8911  
0.40       0.8520   0.7951   0.9415   0.8335   0.7178   0.8911  
0.45       0.8520   0.7951   0.9415   0.8335   0.7178   0.8911  
0.50       0.8520   0.7951   0.9415   0.8335   0.7178   0.8911  
0.55       0.8520   0.7951   0.9415   0.8335   0.7178   0.8911  
0.60       0.8520   0.7951   0.9415   0.8335   0.7178   0.8911  
0.65       0.8520   0.7951   0.9415   0.8335   0.7178   0.8911  
0.70       0.8520   0.7951   0.9415   0.8335   0.7178   0.8911  
0.75       0.8520   0.7951   0.9415   0.8335   0.7178   0.8911  
0.80       0.8520   0.7951   0.9415   0.8335   0.7178   0.8911  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8520, F1=0.7951, Normal Recall=0.9415, Normal Precision=0.8335, Attack Recall=0.7178, Attack Precision=0.8911

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
0.15       0.8290   0.8076   0.9402   0.7691   0.7178   0.9231   <--
0.20       0.8290   0.8076   0.9402   0.7691   0.7178   0.9231  
0.25       0.8290   0.8076   0.9402   0.7691   0.7178   0.9231  
0.30       0.8290   0.8076   0.9402   0.7691   0.7178   0.9231  
0.35       0.8290   0.8076   0.9402   0.7691   0.7178   0.9231  
0.40       0.8290   0.8076   0.9402   0.7691   0.7178   0.9231  
0.45       0.8290   0.8076   0.9402   0.7691   0.7178   0.9231  
0.50       0.8290   0.8076   0.9402   0.7691   0.7178   0.9231  
0.55       0.8290   0.8076   0.9402   0.7691   0.7178   0.9231  
0.60       0.8290   0.8076   0.9402   0.7691   0.7178   0.9231  
0.65       0.8290   0.8076   0.9402   0.7691   0.7178   0.9231  
0.70       0.8290   0.8076   0.9402   0.7691   0.7178   0.9231  
0.75       0.8290   0.8076   0.9402   0.7691   0.7178   0.9231  
0.80       0.8290   0.8076   0.9402   0.7691   0.7178   0.9231  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8290, F1=0.8076, Normal Recall=0.9402, Normal Precision=0.7691, Attack Recall=0.7178, Attack Precision=0.9231

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
0.15       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842   <--
0.20       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.25       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.30       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.35       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.40       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.45       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.50       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.55       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.60       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.65       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.70       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.75       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.80       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9208, F1=0.6453, Normal Recall=0.9430, Normal Precision=0.9681, Attack Recall=0.7206, Attack Precision=0.5842

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
0.15       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597   <--
0.20       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.25       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.30       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.35       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.40       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.45       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.50       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.55       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.60       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.65       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.70       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.75       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.80       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8983, F1=0.7388, Normal Recall=0.9431, Normal Precision=0.9307, Attack Recall=0.7191, Attack Precision=0.7597

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
0.15       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429   <--
0.20       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.25       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.30       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.35       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.40       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.45       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.50       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.55       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.60       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.65       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.70       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.75       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.80       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8755, F1=0.7761, Normal Recall=0.9425, Normal Precision=0.8867, Attack Recall=0.7191, Attack Precision=0.8429

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
0.15       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919   <--
0.20       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.25       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.30       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.35       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.40       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.45       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.50       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.55       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.60       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.65       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.70       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.75       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.80       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8528, F1=0.7962, Normal Recall=0.9419, Normal Precision=0.8341, Attack Recall=0.7191, Attack Precision=0.8919

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
0.15       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239   <--
0.20       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.25       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.30       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.35       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.40       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.45       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.50       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.55       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.60       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.65       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.70       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.75       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.80       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8299, F1=0.8087, Normal Recall=0.9407, Normal Precision=0.7701, Attack Recall=0.7191, Attack Precision=0.9239

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
0.15       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842   <--
0.20       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.25       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.30       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.35       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.40       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.45       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.50       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.55       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.60       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.65       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.70       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.75       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
0.80       0.9208   0.6453   0.9430   0.9681   0.7206   0.5842  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9208, F1=0.6453, Normal Recall=0.9430, Normal Precision=0.9681, Attack Recall=0.7206, Attack Precision=0.5842

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
0.15       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597   <--
0.20       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.25       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.30       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.35       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.40       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.45       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.50       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.55       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.60       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.65       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.70       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.75       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
0.80       0.8983   0.7388   0.9431   0.9307   0.7191   0.7597  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8983, F1=0.7388, Normal Recall=0.9431, Normal Precision=0.9307, Attack Recall=0.7191, Attack Precision=0.7597

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
0.15       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429   <--
0.20       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.25       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.30       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.35       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.40       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.45       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.50       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.55       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.60       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.65       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.70       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.75       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
0.80       0.8755   0.7761   0.9425   0.8867   0.7191   0.8429  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8755, F1=0.7761, Normal Recall=0.9425, Normal Precision=0.8867, Attack Recall=0.7191, Attack Precision=0.8429

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
0.15       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919   <--
0.20       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.25       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.30       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.35       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.40       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.45       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.50       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.55       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.60       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.65       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.70       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.75       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
0.80       0.8528   0.7962   0.9419   0.8341   0.7191   0.8919  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8528, F1=0.7962, Normal Recall=0.9419, Normal Precision=0.8341, Attack Recall=0.7191, Attack Precision=0.8919

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
0.15       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239   <--
0.20       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.25       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.30       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.35       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.40       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.45       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.50       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.55       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.60       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.65       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.70       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.75       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
0.80       0.8299   0.8087   0.9407   0.7701   0.7191   0.9239  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8299, F1=0.8087, Normal Recall=0.9407, Normal Precision=0.7701, Attack Recall=0.7191, Attack Precision=0.9239

```

