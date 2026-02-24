# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-13 06:27:38 |

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
| Original (TFLite) | 0.2307 | 0.3080 | 0.3843 | 0.4615 | 0.5362 | 0.6135 | 0.6900 | 0.7660 | 0.8407 | 0.9175 | 0.9946 |
| QAT+Prune only | 0.3428 | 0.4101 | 0.4755 | 0.5404 | 0.6074 | 0.6695 | 0.7373 | 0.8015 | 0.8669 | 0.9312 | 0.9972 |
| QAT+PTQ | 0.3467 | 0.4138 | 0.4787 | 0.5432 | 0.6099 | 0.6717 | 0.7391 | 0.8028 | 0.8678 | 0.9316 | 0.9972 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.3467 | 0.4138 | 0.4787 | 0.5432 | 0.6099 | 0.6717 | 0.7391 | 0.8028 | 0.8678 | 0.9316 | 0.9972 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2234 | 0.3925 | 0.5257 | 0.6318 | 0.7201 | 0.7938 | 0.8561 | 0.9090 | 0.9559 | 0.9973 |
| QAT+Prune only | 0.0000 | 0.2527 | 0.4320 | 0.5655 | 0.6702 | 0.7511 | 0.8200 | 0.8755 | 0.9230 | 0.9631 | 0.9986 |
| QAT+PTQ | 0.0000 | 0.2539 | 0.4335 | 0.5671 | 0.6716 | 0.7523 | 0.8210 | 0.8763 | 0.9235 | 0.9633 | 0.9986 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2539 | 0.4335 | 0.5671 | 0.6716 | 0.7523 | 0.8210 | 0.8763 | 0.9235 | 0.9633 | 0.9986 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2307 | 0.2316 | 0.2318 | 0.2331 | 0.2306 | 0.2323 | 0.2331 | 0.2325 | 0.2249 | 0.2234 | 0.0000 |
| QAT+Prune only | 0.3428 | 0.3448 | 0.3450 | 0.3446 | 0.3476 | 0.3418 | 0.3474 | 0.3448 | 0.3459 | 0.3375 | 0.0000 |
| QAT+PTQ | 0.3467 | 0.3489 | 0.3491 | 0.3487 | 0.3517 | 0.3462 | 0.3519 | 0.3494 | 0.3501 | 0.3417 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.3467 | 0.3489 | 0.3491 | 0.3487 | 0.3517 | 0.3462 | 0.3519 | 0.3494 | 0.3501 | 0.3417 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2307 | 0.0000 | 0.0000 | 0.0000 | 0.2307 | 1.0000 |
| 90 | 10 | 299,940 | 0.3080 | 0.1258 | 0.9952 | 0.2234 | 0.2316 | 0.9977 |
| 80 | 20 | 291,350 | 0.3843 | 0.2445 | 0.9946 | 0.3925 | 0.2318 | 0.9942 |
| 70 | 30 | 194,230 | 0.4615 | 0.3572 | 0.9946 | 0.5257 | 0.2331 | 0.9902 |
| 60 | 40 | 145,675 | 0.5362 | 0.4629 | 0.9946 | 0.6318 | 0.2306 | 0.9846 |
| 50 | 50 | 116,540 | 0.6135 | 0.5644 | 0.9946 | 0.7201 | 0.2323 | 0.9773 |
| 40 | 60 | 97,115 | 0.6900 | 0.6605 | 0.9946 | 0.7938 | 0.2331 | 0.9664 |
| 30 | 70 | 83,240 | 0.7660 | 0.7515 | 0.9946 | 0.8561 | 0.2325 | 0.9485 |
| 20 | 80 | 72,835 | 0.8407 | 0.8369 | 0.9946 | 0.9090 | 0.2249 | 0.9123 |
| 10 | 90 | 64,740 | 0.9175 | 0.9202 | 0.9946 | 0.9559 | 0.2234 | 0.8211 |
| 0 | 100 | 58,270 | 0.9946 | 1.0000 | 0.9946 | 0.9973 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3428 | 0.0000 | 0.0000 | 0.0000 | 0.3428 | 1.0000 |
| 90 | 10 | 299,940 | 0.4101 | 0.1447 | 0.9974 | 0.2527 | 0.3448 | 0.9992 |
| 80 | 20 | 291,350 | 0.4755 | 0.2757 | 0.9972 | 0.4320 | 0.3450 | 0.9980 |
| 70 | 30 | 194,230 | 0.5404 | 0.3947 | 0.9972 | 0.5655 | 0.3446 | 0.9965 |
| 60 | 40 | 145,675 | 0.6074 | 0.5047 | 0.9972 | 0.6702 | 0.3476 | 0.9946 |
| 50 | 50 | 116,540 | 0.6695 | 0.6024 | 0.9972 | 0.7511 | 0.3418 | 0.9918 |
| 40 | 60 | 97,115 | 0.7373 | 0.6962 | 0.9972 | 0.8200 | 0.3474 | 0.9880 |
| 30 | 70 | 83,240 | 0.8015 | 0.7803 | 0.9972 | 0.8755 | 0.3448 | 0.9813 |
| 20 | 80 | 72,835 | 0.8669 | 0.8591 | 0.9972 | 0.9230 | 0.3459 | 0.9685 |
| 10 | 90 | 64,740 | 0.9312 | 0.9313 | 0.9972 | 0.9631 | 0.3375 | 0.9302 |
| 0 | 100 | 58,270 | 0.9972 | 1.0000 | 0.9972 | 0.9986 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3467 | 0.0000 | 0.0000 | 0.0000 | 0.3467 | 1.0000 |
| 90 | 10 | 299,940 | 0.4138 | 0.1455 | 0.9974 | 0.2539 | 0.3489 | 0.9992 |
| 80 | 20 | 291,350 | 0.4787 | 0.2769 | 0.9972 | 0.4335 | 0.3491 | 0.9980 |
| 70 | 30 | 194,230 | 0.5432 | 0.3962 | 0.9972 | 0.5671 | 0.3487 | 0.9966 |
| 60 | 40 | 145,675 | 0.6099 | 0.5063 | 0.9972 | 0.6716 | 0.3517 | 0.9947 |
| 50 | 50 | 116,540 | 0.6717 | 0.6040 | 0.9972 | 0.7523 | 0.3462 | 0.9920 |
| 40 | 60 | 97,115 | 0.7391 | 0.6977 | 0.9972 | 0.8210 | 0.3519 | 0.9882 |
| 30 | 70 | 83,240 | 0.8028 | 0.7815 | 0.9972 | 0.8763 | 0.3494 | 0.9817 |
| 20 | 80 | 72,835 | 0.8678 | 0.8599 | 0.9972 | 0.9235 | 0.3501 | 0.9690 |
| 10 | 90 | 64,740 | 0.9316 | 0.9317 | 0.9972 | 0.9633 | 0.3417 | 0.9314 |
| 0 | 100 | 58,270 | 0.9972 | 1.0000 | 0.9972 | 0.9986 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.3467 | 0.0000 | 0.0000 | 0.0000 | 0.3467 | 1.0000 |
| 90 | 10 | 299,940 | 0.4138 | 0.1455 | 0.9974 | 0.2539 | 0.3489 | 0.9992 |
| 80 | 20 | 291,350 | 0.4787 | 0.2769 | 0.9972 | 0.4335 | 0.3491 | 0.9980 |
| 70 | 30 | 194,230 | 0.5432 | 0.3962 | 0.9972 | 0.5671 | 0.3487 | 0.9966 |
| 60 | 40 | 145,675 | 0.6099 | 0.5063 | 0.9972 | 0.6716 | 0.3517 | 0.9947 |
| 50 | 50 | 116,540 | 0.6717 | 0.6040 | 0.9972 | 0.7523 | 0.3462 | 0.9920 |
| 40 | 60 | 97,115 | 0.7391 | 0.6977 | 0.9972 | 0.8210 | 0.3519 | 0.9882 |
| 30 | 70 | 83,240 | 0.8028 | 0.7815 | 0.9972 | 0.8763 | 0.3494 | 0.9817 |
| 20 | 80 | 72,835 | 0.8678 | 0.8599 | 0.9972 | 0.9235 | 0.3501 | 0.9690 |
| 10 | 90 | 64,740 | 0.9316 | 0.9317 | 0.9972 | 0.9633 | 0.3417 | 0.9314 |
| 0 | 100 | 58,270 | 0.9972 | 1.0000 | 0.9972 | 0.9986 | 0.0000 | 0.0000 |


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
0.15       0.3080   0.2234   0.2316   0.9977   0.9952   0.1258   <--
0.20       0.3080   0.2234   0.2316   0.9977   0.9952   0.1258  
0.25       0.3080   0.2234   0.2316   0.9977   0.9952   0.1258  
0.30       0.3080   0.2234   0.2316   0.9977   0.9952   0.1258  
0.35       0.3080   0.2234   0.2316   0.9977   0.9952   0.1258  
0.40       0.3080   0.2234   0.2316   0.9977   0.9952   0.1258  
0.45       0.3080   0.2234   0.2316   0.9977   0.9952   0.1258  
0.50       0.3080   0.2234   0.2316   0.9977   0.9952   0.1258  
0.55       0.3080   0.2234   0.2316   0.9977   0.9952   0.1258  
0.60       0.3080   0.2234   0.2316   0.9977   0.9952   0.1258  
0.65       0.3080   0.2234   0.2316   0.9977   0.9952   0.1258  
0.70       0.3080   0.2234   0.2316   0.9977   0.9952   0.1258  
0.75       0.3080   0.2234   0.2316   0.9977   0.9952   0.1258  
0.80       0.3080   0.2234   0.2316   0.9977   0.9952   0.1258  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3080, F1=0.2234, Normal Recall=0.2316, Normal Precision=0.9977, Attack Recall=0.9952, Attack Precision=0.1258

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
0.15       0.3841   0.3924   0.2315   0.9942   0.9946   0.2444   <--
0.20       0.3841   0.3924   0.2315   0.9942   0.9946   0.2444  
0.25       0.3841   0.3924   0.2315   0.9942   0.9946   0.2444  
0.30       0.3841   0.3924   0.2315   0.9942   0.9946   0.2444  
0.35       0.3841   0.3924   0.2315   0.9942   0.9946   0.2444  
0.40       0.3841   0.3924   0.2315   0.9942   0.9946   0.2444  
0.45       0.3841   0.3924   0.2315   0.9942   0.9946   0.2444  
0.50       0.3841   0.3924   0.2315   0.9942   0.9946   0.2444  
0.55       0.3841   0.3924   0.2315   0.9942   0.9946   0.2444  
0.60       0.3841   0.3924   0.2315   0.9942   0.9946   0.2444  
0.65       0.3841   0.3924   0.2315   0.9942   0.9946   0.2444  
0.70       0.3841   0.3924   0.2315   0.9942   0.9946   0.2444  
0.75       0.3841   0.3924   0.2315   0.9942   0.9946   0.2444  
0.80       0.3841   0.3924   0.2315   0.9942   0.9946   0.2444  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3841, F1=0.3924, Normal Recall=0.2315, Normal Precision=0.9942, Attack Recall=0.9946, Attack Precision=0.2444

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
0.15       0.4607   0.5253   0.2319   0.9901   0.9946   0.3569   <--
0.20       0.4607   0.5253   0.2319   0.9901   0.9946   0.3569  
0.25       0.4607   0.5253   0.2319   0.9901   0.9946   0.3569  
0.30       0.4607   0.5253   0.2319   0.9901   0.9946   0.3569  
0.35       0.4607   0.5253   0.2319   0.9901   0.9946   0.3569  
0.40       0.4607   0.5253   0.2319   0.9901   0.9946   0.3569  
0.45       0.4607   0.5253   0.2319   0.9901   0.9946   0.3569  
0.50       0.4607   0.5253   0.2319   0.9901   0.9946   0.3569  
0.55       0.4607   0.5253   0.2319   0.9901   0.9946   0.3569  
0.60       0.4607   0.5253   0.2319   0.9901   0.9946   0.3569  
0.65       0.4607   0.5253   0.2319   0.9901   0.9946   0.3569  
0.70       0.4607   0.5253   0.2319   0.9901   0.9946   0.3569  
0.75       0.4607   0.5253   0.2319   0.9901   0.9946   0.3569  
0.80       0.4607   0.5253   0.2319   0.9901   0.9946   0.3569  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4607, F1=0.5253, Normal Recall=0.2319, Normal Precision=0.9901, Attack Recall=0.9946, Attack Precision=0.3569

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
0.15       0.5363   0.6318   0.2308   0.9846   0.9946   0.4629   <--
0.20       0.5363   0.6318   0.2308   0.9846   0.9946   0.4629  
0.25       0.5363   0.6318   0.2308   0.9846   0.9946   0.4629  
0.30       0.5363   0.6318   0.2308   0.9846   0.9946   0.4629  
0.35       0.5363   0.6318   0.2308   0.9846   0.9946   0.4629  
0.40       0.5363   0.6318   0.2308   0.9846   0.9946   0.4629  
0.45       0.5363   0.6318   0.2308   0.9846   0.9946   0.4629  
0.50       0.5363   0.6318   0.2308   0.9846   0.9946   0.4629  
0.55       0.5363   0.6318   0.2308   0.9846   0.9946   0.4629  
0.60       0.5363   0.6318   0.2308   0.9846   0.9946   0.4629  
0.65       0.5363   0.6318   0.2308   0.9846   0.9946   0.4629  
0.70       0.5363   0.6318   0.2308   0.9846   0.9946   0.4629  
0.75       0.5363   0.6318   0.2308   0.9846   0.9946   0.4629  
0.80       0.5363   0.6318   0.2308   0.9846   0.9946   0.4629  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5363, F1=0.6318, Normal Recall=0.2308, Normal Precision=0.9846, Attack Recall=0.9946, Attack Precision=0.4629

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
0.15       0.6128   0.7198   0.2310   0.9771   0.9946   0.5640   <--
0.20       0.6128   0.7198   0.2310   0.9771   0.9946   0.5640  
0.25       0.6128   0.7198   0.2310   0.9771   0.9946   0.5640  
0.30       0.6128   0.7198   0.2310   0.9771   0.9946   0.5640  
0.35       0.6128   0.7198   0.2310   0.9771   0.9946   0.5640  
0.40       0.6128   0.7198   0.2310   0.9771   0.9946   0.5640  
0.45       0.6128   0.7198   0.2310   0.9771   0.9946   0.5640  
0.50       0.6128   0.7198   0.2310   0.9771   0.9946   0.5640  
0.55       0.6128   0.7198   0.2310   0.9771   0.9946   0.5640  
0.60       0.6128   0.7198   0.2310   0.9771   0.9946   0.5640  
0.65       0.6128   0.7198   0.2310   0.9771   0.9946   0.5640  
0.70       0.6128   0.7198   0.2310   0.9771   0.9946   0.5640  
0.75       0.6128   0.7198   0.2310   0.9771   0.9946   0.5640  
0.80       0.6128   0.7198   0.2310   0.9771   0.9946   0.5640  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6128, F1=0.7198, Normal Recall=0.2310, Normal Precision=0.9771, Attack Recall=0.9946, Attack Precision=0.5640

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
0.15       0.4101   0.2527   0.3448   0.9992   0.9974   0.1447   <--
0.20       0.4101   0.2527   0.3448   0.9992   0.9974   0.1447  
0.25       0.4101   0.2527   0.3448   0.9992   0.9974   0.1447  
0.30       0.4101   0.2527   0.3448   0.9992   0.9974   0.1447  
0.35       0.4101   0.2527   0.3448   0.9992   0.9974   0.1447  
0.40       0.4101   0.2527   0.3448   0.9992   0.9974   0.1447  
0.45       0.4101   0.2527   0.3448   0.9992   0.9974   0.1447  
0.50       0.4101   0.2527   0.3448   0.9992   0.9974   0.1447  
0.55       0.4101   0.2527   0.3448   0.9992   0.9974   0.1447  
0.60       0.4101   0.2527   0.3448   0.9992   0.9974   0.1447  
0.65       0.4101   0.2527   0.3448   0.9992   0.9974   0.1447  
0.70       0.4101   0.2527   0.3448   0.9992   0.9974   0.1447  
0.75       0.4101   0.2527   0.3448   0.9992   0.9974   0.1447  
0.80       0.4101   0.2527   0.3448   0.9992   0.9974   0.1447  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4101, F1=0.2527, Normal Recall=0.3448, Normal Precision=0.9992, Attack Recall=0.9974, Attack Precision=0.1447

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
0.15       0.4752   0.4319   0.3447   0.9980   0.9972   0.2756   <--
0.20       0.4752   0.4319   0.3447   0.9980   0.9972   0.2756  
0.25       0.4752   0.4319   0.3447   0.9980   0.9972   0.2756  
0.30       0.4752   0.4319   0.3447   0.9980   0.9972   0.2756  
0.35       0.4752   0.4319   0.3447   0.9980   0.9972   0.2756  
0.40       0.4752   0.4319   0.3447   0.9980   0.9972   0.2756  
0.45       0.4752   0.4319   0.3447   0.9980   0.9972   0.2756  
0.50       0.4752   0.4319   0.3447   0.9980   0.9972   0.2756  
0.55       0.4752   0.4319   0.3447   0.9980   0.9972   0.2756  
0.60       0.4752   0.4319   0.3447   0.9980   0.9972   0.2756  
0.65       0.4752   0.4319   0.3447   0.9980   0.9972   0.2756  
0.70       0.4752   0.4319   0.3447   0.9980   0.9972   0.2756  
0.75       0.4752   0.4319   0.3447   0.9980   0.9972   0.2756  
0.80       0.4752   0.4319   0.3447   0.9980   0.9972   0.2756  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4752, F1=0.4319, Normal Recall=0.3447, Normal Precision=0.9980, Attack Recall=0.9972, Attack Precision=0.2756

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
0.15       0.5396   0.5651   0.3435   0.9965   0.9972   0.3943   <--
0.20       0.5396   0.5651   0.3435   0.9965   0.9972   0.3943  
0.25       0.5396   0.5651   0.3435   0.9965   0.9972   0.3943  
0.30       0.5396   0.5651   0.3435   0.9965   0.9972   0.3943  
0.35       0.5396   0.5651   0.3435   0.9965   0.9972   0.3943  
0.40       0.5396   0.5651   0.3435   0.9965   0.9972   0.3943  
0.45       0.5396   0.5651   0.3435   0.9965   0.9972   0.3943  
0.50       0.5396   0.5651   0.3435   0.9965   0.9972   0.3943  
0.55       0.5396   0.5651   0.3435   0.9965   0.9972   0.3943  
0.60       0.5396   0.5651   0.3435   0.9965   0.9972   0.3943  
0.65       0.5396   0.5651   0.3435   0.9965   0.9972   0.3943  
0.70       0.5396   0.5651   0.3435   0.9965   0.9972   0.3943  
0.75       0.5396   0.5651   0.3435   0.9965   0.9972   0.3943  
0.80       0.5396   0.5651   0.3435   0.9965   0.9972   0.3943  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5396, F1=0.5651, Normal Recall=0.3435, Normal Precision=0.9965, Attack Recall=0.9972, Attack Precision=0.3943

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
0.15       0.6045   0.6686   0.3427   0.9946   0.9972   0.5029   <--
0.20       0.6045   0.6686   0.3427   0.9946   0.9972   0.5029  
0.25       0.6045   0.6686   0.3427   0.9946   0.9972   0.5029  
0.30       0.6045   0.6686   0.3427   0.9946   0.9972   0.5029  
0.35       0.6045   0.6686   0.3427   0.9946   0.9972   0.5029  
0.40       0.6045   0.6686   0.3427   0.9946   0.9972   0.5029  
0.45       0.6045   0.6686   0.3427   0.9946   0.9972   0.5029  
0.50       0.6045   0.6686   0.3427   0.9946   0.9972   0.5029  
0.55       0.6045   0.6686   0.3427   0.9946   0.9972   0.5029  
0.60       0.6045   0.6686   0.3427   0.9946   0.9972   0.5029  
0.65       0.6045   0.6686   0.3427   0.9946   0.9972   0.5029  
0.70       0.6045   0.6686   0.3427   0.9946   0.9972   0.5029  
0.75       0.6045   0.6686   0.3427   0.9946   0.9972   0.5029  
0.80       0.6045   0.6686   0.3427   0.9946   0.9972   0.5029  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6045, F1=0.6686, Normal Recall=0.3427, Normal Precision=0.9946, Attack Recall=0.9972, Attack Precision=0.5029

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
0.15       0.6688   0.7507   0.3404   0.9918   0.9972   0.6019   <--
0.20       0.6688   0.7507   0.3404   0.9918   0.9972   0.6019  
0.25       0.6688   0.7507   0.3404   0.9918   0.9972   0.6019  
0.30       0.6688   0.7507   0.3404   0.9918   0.9972   0.6019  
0.35       0.6688   0.7507   0.3404   0.9918   0.9972   0.6019  
0.40       0.6688   0.7507   0.3404   0.9918   0.9972   0.6019  
0.45       0.6688   0.7507   0.3404   0.9918   0.9972   0.6019  
0.50       0.6688   0.7507   0.3404   0.9918   0.9972   0.6019  
0.55       0.6688   0.7507   0.3404   0.9918   0.9972   0.6019  
0.60       0.6688   0.7507   0.3404   0.9918   0.9972   0.6019  
0.65       0.6688   0.7507   0.3404   0.9918   0.9972   0.6019  
0.70       0.6688   0.7507   0.3404   0.9918   0.9972   0.6019  
0.75       0.6688   0.7507   0.3404   0.9918   0.9972   0.6019  
0.80       0.6688   0.7507   0.3404   0.9918   0.9972   0.6019  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6688, F1=0.7507, Normal Recall=0.3404, Normal Precision=0.9918, Attack Recall=0.9972, Attack Precision=0.6019

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
0.15       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455   <--
0.20       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.25       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.30       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.35       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.40       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.45       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.50       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.55       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.60       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.65       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.70       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.75       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.80       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4138, F1=0.2539, Normal Recall=0.3489, Normal Precision=0.9992, Attack Recall=0.9974, Attack Precision=0.1455

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
0.15       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768   <--
0.20       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.25       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.30       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.35       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.40       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.45       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.50       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.55       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.60       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.65       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.70       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.75       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.80       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4784, F1=0.4334, Normal Recall=0.3487, Normal Precision=0.9980, Attack Recall=0.9972, Attack Precision=0.2768

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
0.15       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957   <--
0.20       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.25       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.30       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.35       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.40       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.45       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.50       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.55       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.60       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.65       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.70       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.75       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.80       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5423, F1=0.5666, Normal Recall=0.3473, Normal Precision=0.9966, Attack Recall=0.9972, Attack Precision=0.3957

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
0.15       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043   <--
0.20       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.25       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.30       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.35       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.40       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.45       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.50       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.55       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.60       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.65       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.70       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.75       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.80       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6068, F1=0.6699, Normal Recall=0.3466, Normal Precision=0.9946, Attack Recall=0.9972, Attack Precision=0.5043

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
0.15       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033   <--
0.20       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.25       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.30       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.35       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.40       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.45       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.50       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.55       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.60       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.65       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.70       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.75       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.80       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6707, F1=0.7517, Normal Recall=0.3442, Normal Precision=0.9919, Attack Recall=0.9972, Attack Precision=0.6033

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
0.15       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455   <--
0.20       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.25       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.30       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.35       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.40       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.45       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.50       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.55       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.60       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.65       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.70       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.75       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
0.80       0.4138   0.2539   0.3489   0.9992   0.9974   0.1455  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4138, F1=0.2539, Normal Recall=0.3489, Normal Precision=0.9992, Attack Recall=0.9974, Attack Precision=0.1455

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
0.15       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768   <--
0.20       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.25       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.30       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.35       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.40       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.45       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.50       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.55       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.60       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.65       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.70       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.75       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
0.80       0.4784   0.4334   0.3487   0.9980   0.9972   0.2768  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4784, F1=0.4334, Normal Recall=0.3487, Normal Precision=0.9980, Attack Recall=0.9972, Attack Precision=0.2768

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
0.15       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957   <--
0.20       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.25       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.30       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.35       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.40       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.45       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.50       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.55       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.60       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.65       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.70       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.75       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
0.80       0.5423   0.5666   0.3473   0.9966   0.9972   0.3957  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5423, F1=0.5666, Normal Recall=0.3473, Normal Precision=0.9966, Attack Recall=0.9972, Attack Precision=0.3957

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
0.15       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043   <--
0.20       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.25       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.30       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.35       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.40       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.45       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.50       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.55       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.60       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.65       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.70       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.75       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
0.80       0.6068   0.6699   0.3466   0.9946   0.9972   0.5043  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6068, F1=0.6699, Normal Recall=0.3466, Normal Precision=0.9946, Attack Recall=0.9972, Attack Precision=0.5043

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
0.15       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033   <--
0.20       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.25       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.30       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.35       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.40       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.45       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.50       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.55       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.60       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.65       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.70       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.75       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
0.80       0.6707   0.7517   0.3442   0.9919   0.9972   0.6033  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6707, F1=0.7517, Normal Recall=0.3442, Normal Precision=0.9919, Attack Recall=0.9972, Attack Precision=0.6033

```

