# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e2_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-22 16:45:51 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3240 | 0.3674 | 0.4105 | 0.4536 | 0.4970 | 0.5414 | 0.5843 | 0.6278 | 0.6708 | 0.7139 | 0.7564 |
| QAT+Prune only | 0.8123 | 0.8200 | 0.8270 | 0.8355 | 0.8429 | 0.8495 | 0.8575 | 0.8655 | 0.8724 | 0.8804 | 0.8880 |
| QAT+PTQ | 0.8126 | 0.8195 | 0.8258 | 0.8338 | 0.8405 | 0.8466 | 0.8541 | 0.8613 | 0.8677 | 0.8751 | 0.8820 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8126 | 0.8195 | 0.8258 | 0.8338 | 0.8405 | 0.8466 | 0.8541 | 0.8613 | 0.8677 | 0.8751 | 0.8820 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1939 | 0.3392 | 0.4538 | 0.5461 | 0.6226 | 0.6859 | 0.7400 | 0.7862 | 0.8264 | 0.8613 |
| QAT+Prune only | 0.0000 | 0.4966 | 0.6724 | 0.7641 | 0.8189 | 0.8551 | 0.8820 | 0.9024 | 0.9176 | 0.9304 | 0.9407 |
| QAT+PTQ | 0.0000 | 0.4941 | 0.6695 | 0.7610 | 0.8156 | 0.8518 | 0.8788 | 0.8990 | 0.9143 | 0.9271 | 0.9373 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4941 | 0.6695 | 0.7610 | 0.8156 | 0.8518 | 0.8788 | 0.8990 | 0.9143 | 0.9271 | 0.9373 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3240 | 0.3237 | 0.3240 | 0.3238 | 0.3240 | 0.3264 | 0.3261 | 0.3278 | 0.3284 | 0.3315 | 0.0000 |
| QAT+Prune only | 0.8123 | 0.8125 | 0.8117 | 0.8131 | 0.8128 | 0.8111 | 0.8118 | 0.8133 | 0.8102 | 0.8125 | 0.0000 |
| QAT+PTQ | 0.8126 | 0.8126 | 0.8118 | 0.8132 | 0.8128 | 0.8112 | 0.8122 | 0.8131 | 0.8107 | 0.8128 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8126 | 0.8126 | 0.8118 | 0.8132 | 0.8128 | 0.8112 | 0.8122 | 0.8131 | 0.8107 | 0.8128 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3240 | 0.0000 | 0.0000 | 0.0000 | 0.3240 | 1.0000 |
| 90 | 10 | 299,940 | 0.3674 | 0.1111 | 0.7606 | 0.1939 | 0.3237 | 0.9241 |
| 80 | 20 | 291,350 | 0.4105 | 0.2186 | 0.7564 | 0.3392 | 0.3240 | 0.8418 |
| 70 | 30 | 194,230 | 0.4536 | 0.3241 | 0.7565 | 0.4538 | 0.3238 | 0.7562 |
| 60 | 40 | 145,675 | 0.4970 | 0.4273 | 0.7564 | 0.5461 | 0.3240 | 0.6662 |
| 50 | 50 | 116,540 | 0.5414 | 0.5290 | 0.7564 | 0.6226 | 0.3264 | 0.5727 |
| 40 | 60 | 97,115 | 0.5843 | 0.6274 | 0.7564 | 0.6859 | 0.3261 | 0.4716 |
| 30 | 70 | 83,240 | 0.6278 | 0.7242 | 0.7564 | 0.7400 | 0.3278 | 0.3658 |
| 20 | 80 | 72,835 | 0.6708 | 0.8184 | 0.7565 | 0.7862 | 0.3284 | 0.2521 |
| 10 | 90 | 64,740 | 0.7139 | 0.9106 | 0.7564 | 0.8264 | 0.3315 | 0.1314 |
| 0 | 100 | 58,270 | 0.7564 | 1.0000 | 0.7564 | 0.8613 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8123 | 0.0000 | 0.0000 | 0.0000 | 0.8123 | 1.0000 |
| 90 | 10 | 299,940 | 0.8200 | 0.3447 | 0.8876 | 0.4966 | 0.8125 | 0.9849 |
| 80 | 20 | 291,350 | 0.8270 | 0.5411 | 0.8880 | 0.6724 | 0.8117 | 0.9666 |
| 70 | 30 | 194,230 | 0.8355 | 0.6706 | 0.8880 | 0.7641 | 0.8131 | 0.9442 |
| 60 | 40 | 145,675 | 0.8429 | 0.7598 | 0.8880 | 0.8189 | 0.8128 | 0.9158 |
| 50 | 50 | 116,540 | 0.8495 | 0.8246 | 0.8880 | 0.8551 | 0.8111 | 0.8786 |
| 40 | 60 | 97,115 | 0.8575 | 0.8762 | 0.8880 | 0.8820 | 0.8118 | 0.8285 |
| 30 | 70 | 83,240 | 0.8655 | 0.9173 | 0.8879 | 0.9024 | 0.8133 | 0.7567 |
| 20 | 80 | 72,835 | 0.8724 | 0.9493 | 0.8879 | 0.9176 | 0.8102 | 0.6438 |
| 10 | 90 | 64,740 | 0.8804 | 0.9771 | 0.8880 | 0.9304 | 0.8125 | 0.4462 |
| 0 | 100 | 58,270 | 0.8880 | 1.0000 | 0.8880 | 0.9407 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8126 | 0.0000 | 0.0000 | 0.0000 | 0.8126 | 1.0000 |
| 90 | 10 | 299,940 | 0.8195 | 0.3432 | 0.8817 | 0.4941 | 0.8126 | 0.9841 |
| 80 | 20 | 291,350 | 0.8258 | 0.5395 | 0.8820 | 0.6695 | 0.8118 | 0.9649 |
| 70 | 30 | 194,230 | 0.8338 | 0.6693 | 0.8820 | 0.7610 | 0.8132 | 0.9415 |
| 60 | 40 | 145,675 | 0.8405 | 0.7585 | 0.8820 | 0.8156 | 0.8128 | 0.9118 |
| 50 | 50 | 116,540 | 0.8466 | 0.8237 | 0.8820 | 0.8518 | 0.8112 | 0.8730 |
| 40 | 60 | 97,115 | 0.8541 | 0.8757 | 0.8820 | 0.8788 | 0.8122 | 0.8211 |
| 30 | 70 | 83,240 | 0.8613 | 0.9167 | 0.8820 | 0.8990 | 0.8131 | 0.7470 |
| 20 | 80 | 72,835 | 0.8677 | 0.9491 | 0.8820 | 0.9143 | 0.8107 | 0.6320 |
| 10 | 90 | 64,740 | 0.8751 | 0.9770 | 0.8820 | 0.9271 | 0.8128 | 0.4336 |
| 0 | 100 | 58,270 | 0.8820 | 1.0000 | 0.8820 | 0.9373 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8126 | 0.0000 | 0.0000 | 0.0000 | 0.8126 | 1.0000 |
| 90 | 10 | 299,940 | 0.8195 | 0.3432 | 0.8817 | 0.4941 | 0.8126 | 0.9841 |
| 80 | 20 | 291,350 | 0.8258 | 0.5395 | 0.8820 | 0.6695 | 0.8118 | 0.9649 |
| 70 | 30 | 194,230 | 0.8338 | 0.6693 | 0.8820 | 0.7610 | 0.8132 | 0.9415 |
| 60 | 40 | 145,675 | 0.8405 | 0.7585 | 0.8820 | 0.8156 | 0.8128 | 0.9118 |
| 50 | 50 | 116,540 | 0.8466 | 0.8237 | 0.8820 | 0.8518 | 0.8112 | 0.8730 |
| 40 | 60 | 97,115 | 0.8541 | 0.8757 | 0.8820 | 0.8788 | 0.8122 | 0.8211 |
| 30 | 70 | 83,240 | 0.8613 | 0.9167 | 0.8820 | 0.8990 | 0.8131 | 0.7470 |
| 20 | 80 | 72,835 | 0.8677 | 0.9491 | 0.8820 | 0.9143 | 0.8107 | 0.6320 |
| 10 | 90 | 64,740 | 0.8751 | 0.9770 | 0.8820 | 0.9271 | 0.8128 | 0.4336 |
| 0 | 100 | 58,270 | 0.8820 | 1.0000 | 0.8820 | 0.9373 | 0.0000 | 0.0000 |


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
0.15       0.3673   0.1937   0.3237   0.9238   0.7598   0.1110   <--
0.20       0.3673   0.1937   0.3237   0.9238   0.7598   0.1110  
0.25       0.3673   0.1937   0.3237   0.9238   0.7598   0.1110  
0.30       0.3673   0.1937   0.3237   0.9238   0.7598   0.1110  
0.35       0.3673   0.1937   0.3237   0.9238   0.7598   0.1110  
0.40       0.3673   0.1937   0.3237   0.9238   0.7598   0.1110  
0.45       0.3673   0.1937   0.3237   0.9238   0.7598   0.1110  
0.50       0.3673   0.1937   0.3237   0.9238   0.7598   0.1110  
0.55       0.3673   0.1937   0.3237   0.9238   0.7598   0.1110  
0.60       0.3673   0.1937   0.3237   0.9238   0.7598   0.1110  
0.65       0.3673   0.1937   0.3237   0.9238   0.7598   0.1110  
0.70       0.3673   0.1937   0.3237   0.9238   0.7598   0.1110  
0.75       0.3673   0.1937   0.3237   0.9238   0.7598   0.1110  
0.80       0.3673   0.1937   0.3237   0.9238   0.7598   0.1110  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3673, F1=0.1937, Normal Recall=0.3237, Normal Precision=0.9238, Attack Recall=0.7598, Attack Precision=0.1110

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
0.15       0.4096   0.3388   0.3229   0.8413   0.7564   0.2183   <--
0.20       0.4096   0.3388   0.3229   0.8413   0.7564   0.2183  
0.25       0.4096   0.3388   0.3229   0.8413   0.7564   0.2183  
0.30       0.4096   0.3388   0.3229   0.8413   0.7564   0.2183  
0.35       0.4096   0.3388   0.3229   0.8413   0.7564   0.2183  
0.40       0.4096   0.3388   0.3229   0.8413   0.7564   0.2183  
0.45       0.4096   0.3388   0.3229   0.8413   0.7564   0.2183  
0.50       0.4096   0.3388   0.3229   0.8413   0.7564   0.2183  
0.55       0.4096   0.3388   0.3229   0.8413   0.7564   0.2183  
0.60       0.4096   0.3388   0.3229   0.8413   0.7564   0.2183  
0.65       0.4096   0.3388   0.3229   0.8413   0.7564   0.2183  
0.70       0.4096   0.3388   0.3229   0.8413   0.7564   0.2183  
0.75       0.4096   0.3388   0.3229   0.8413   0.7564   0.2183  
0.80       0.4096   0.3388   0.3229   0.8413   0.7564   0.2183  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4096, F1=0.3388, Normal Recall=0.3229, Normal Precision=0.8413, Attack Recall=0.7564, Attack Precision=0.2183

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
0.15       0.4538   0.4539   0.3242   0.7564   0.7564   0.3242   <--
0.20       0.4538   0.4539   0.3242   0.7564   0.7564   0.3242  
0.25       0.4538   0.4539   0.3242   0.7564   0.7564   0.3242  
0.30       0.4538   0.4539   0.3242   0.7564   0.7564   0.3242  
0.35       0.4538   0.4539   0.3242   0.7564   0.7564   0.3242  
0.40       0.4538   0.4539   0.3242   0.7564   0.7564   0.3242  
0.45       0.4538   0.4539   0.3242   0.7564   0.7564   0.3242  
0.50       0.4538   0.4539   0.3242   0.7564   0.7564   0.3242  
0.55       0.4538   0.4539   0.3242   0.7564   0.7564   0.3242  
0.60       0.4538   0.4539   0.3242   0.7564   0.7564   0.3242  
0.65       0.4538   0.4539   0.3242   0.7564   0.7564   0.3242  
0.70       0.4538   0.4539   0.3242   0.7564   0.7564   0.3242  
0.75       0.4538   0.4539   0.3242   0.7564   0.7564   0.3242  
0.80       0.4538   0.4539   0.3242   0.7564   0.7564   0.3242  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4538, F1=0.4539, Normal Recall=0.3242, Normal Precision=0.7564, Attack Recall=0.7564, Attack Precision=0.3242

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
0.15       0.4974   0.5463   0.3247   0.6666   0.7564   0.4275   <--
0.20       0.4974   0.5463   0.3247   0.6666   0.7564   0.4275  
0.25       0.4974   0.5463   0.3247   0.6666   0.7564   0.4275  
0.30       0.4974   0.5463   0.3247   0.6666   0.7564   0.4275  
0.35       0.4974   0.5463   0.3247   0.6666   0.7564   0.4275  
0.40       0.4974   0.5463   0.3247   0.6666   0.7564   0.4275  
0.45       0.4974   0.5463   0.3247   0.6666   0.7564   0.4275  
0.50       0.4974   0.5463   0.3247   0.6666   0.7564   0.4275  
0.55       0.4974   0.5463   0.3247   0.6666   0.7564   0.4275  
0.60       0.4974   0.5463   0.3247   0.6666   0.7564   0.4275  
0.65       0.4974   0.5463   0.3247   0.6666   0.7564   0.4275  
0.70       0.4974   0.5463   0.3247   0.6666   0.7564   0.4275  
0.75       0.4974   0.5463   0.3247   0.6666   0.7564   0.4275  
0.80       0.4974   0.5463   0.3247   0.6666   0.7564   0.4275  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4974, F1=0.5463, Normal Recall=0.3247, Normal Precision=0.6666, Attack Recall=0.7564, Attack Precision=0.4275

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
0.15       0.5401   0.6219   0.3239   0.5708   0.7564   0.5280   <--
0.20       0.5401   0.6219   0.3239   0.5708   0.7564   0.5280  
0.25       0.5401   0.6219   0.3239   0.5708   0.7564   0.5280  
0.30       0.5401   0.6219   0.3239   0.5708   0.7564   0.5280  
0.35       0.5401   0.6219   0.3239   0.5708   0.7564   0.5280  
0.40       0.5401   0.6219   0.3239   0.5708   0.7564   0.5280  
0.45       0.5401   0.6219   0.3239   0.5708   0.7564   0.5280  
0.50       0.5401   0.6219   0.3239   0.5708   0.7564   0.5280  
0.55       0.5401   0.6219   0.3239   0.5708   0.7564   0.5280  
0.60       0.5401   0.6219   0.3239   0.5708   0.7564   0.5280  
0.65       0.5401   0.6219   0.3239   0.5708   0.7564   0.5280  
0.70       0.5401   0.6219   0.3239   0.5708   0.7564   0.5280  
0.75       0.5401   0.6219   0.3239   0.5708   0.7564   0.5280  
0.80       0.5401   0.6219   0.3239   0.5708   0.7564   0.5280  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5401, F1=0.6219, Normal Recall=0.3239, Normal Precision=0.5708, Attack Recall=0.7564, Attack Precision=0.5280

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
0.15       0.8202   0.4974   0.8125   0.9851   0.8897   0.3452   <--
0.20       0.8202   0.4974   0.8125   0.9851   0.8897   0.3452  
0.25       0.8202   0.4974   0.8125   0.9851   0.8897   0.3452  
0.30       0.8202   0.4974   0.8125   0.9851   0.8897   0.3452  
0.35       0.8202   0.4974   0.8125   0.9851   0.8897   0.3452  
0.40       0.8202   0.4974   0.8125   0.9851   0.8897   0.3452  
0.45       0.8202   0.4974   0.8125   0.9851   0.8897   0.3452  
0.50       0.8202   0.4974   0.8125   0.9851   0.8897   0.3452  
0.55       0.8202   0.4974   0.8125   0.9851   0.8897   0.3452  
0.60       0.8202   0.4974   0.8125   0.9851   0.8897   0.3452  
0.65       0.8202   0.4974   0.8125   0.9851   0.8897   0.3452  
0.70       0.8202   0.4974   0.8125   0.9851   0.8897   0.3452  
0.75       0.8202   0.4974   0.8125   0.9851   0.8897   0.3452  
0.80       0.8202   0.4974   0.8125   0.9851   0.8897   0.3452  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8202, F1=0.4974, Normal Recall=0.8125, Normal Precision=0.9851, Attack Recall=0.8897, Attack Precision=0.3452

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
0.15       0.8277   0.6734   0.8127   0.9667   0.8880   0.5424   <--
0.20       0.8277   0.6734   0.8127   0.9667   0.8880   0.5424  
0.25       0.8277   0.6734   0.8127   0.9667   0.8880   0.5424  
0.30       0.8277   0.6734   0.8127   0.9667   0.8880   0.5424  
0.35       0.8277   0.6734   0.8127   0.9667   0.8880   0.5424  
0.40       0.8277   0.6734   0.8127   0.9667   0.8880   0.5424  
0.45       0.8277   0.6734   0.8127   0.9667   0.8880   0.5424  
0.50       0.8277   0.6734   0.8127   0.9667   0.8880   0.5424  
0.55       0.8277   0.6734   0.8127   0.9667   0.8880   0.5424  
0.60       0.8277   0.6734   0.8127   0.9667   0.8880   0.5424  
0.65       0.8277   0.6734   0.8127   0.9667   0.8880   0.5424  
0.70       0.8277   0.6734   0.8127   0.9667   0.8880   0.5424  
0.75       0.8277   0.6734   0.8127   0.9667   0.8880   0.5424  
0.80       0.8277   0.6734   0.8127   0.9667   0.8880   0.5424  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8277, F1=0.6734, Normal Recall=0.8127, Normal Precision=0.9667, Attack Recall=0.8880, Attack Precision=0.5424

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
0.15       0.8352   0.7638   0.8126   0.9442   0.8880   0.6700   <--
0.20       0.8352   0.7638   0.8126   0.9442   0.8880   0.6700  
0.25       0.8352   0.7638   0.8126   0.9442   0.8880   0.6700  
0.30       0.8352   0.7638   0.8126   0.9442   0.8880   0.6700  
0.35       0.8352   0.7638   0.8126   0.9442   0.8880   0.6700  
0.40       0.8352   0.7638   0.8126   0.9442   0.8880   0.6700  
0.45       0.8352   0.7638   0.8126   0.9442   0.8880   0.6700  
0.50       0.8352   0.7638   0.8126   0.9442   0.8880   0.6700  
0.55       0.8352   0.7638   0.8126   0.9442   0.8880   0.6700  
0.60       0.8352   0.7638   0.8126   0.9442   0.8880   0.6700  
0.65       0.8352   0.7638   0.8126   0.9442   0.8880   0.6700  
0.70       0.8352   0.7638   0.8126   0.9442   0.8880   0.6700  
0.75       0.8352   0.7638   0.8126   0.9442   0.8880   0.6700  
0.80       0.8352   0.7638   0.8126   0.9442   0.8880   0.6700  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8352, F1=0.7638, Normal Recall=0.8126, Normal Precision=0.9442, Attack Recall=0.8880, Attack Precision=0.6700

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
0.15       0.8424   0.8184   0.8120   0.9158   0.8880   0.7590   <--
0.20       0.8424   0.8184   0.8120   0.9158   0.8880   0.7590  
0.25       0.8424   0.8184   0.8120   0.9158   0.8880   0.7590  
0.30       0.8424   0.8184   0.8120   0.9158   0.8880   0.7590  
0.35       0.8424   0.8184   0.8120   0.9158   0.8880   0.7590  
0.40       0.8424   0.8184   0.8120   0.9158   0.8880   0.7590  
0.45       0.8424   0.8184   0.8120   0.9158   0.8880   0.7590  
0.50       0.8424   0.8184   0.8120   0.9158   0.8880   0.7590  
0.55       0.8424   0.8184   0.8120   0.9158   0.8880   0.7590  
0.60       0.8424   0.8184   0.8120   0.9158   0.8880   0.7590  
0.65       0.8424   0.8184   0.8120   0.9158   0.8880   0.7590  
0.70       0.8424   0.8184   0.8120   0.9158   0.8880   0.7590  
0.75       0.8424   0.8184   0.8120   0.9158   0.8880   0.7590  
0.80       0.8424   0.8184   0.8120   0.9158   0.8880   0.7590  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8424, F1=0.8184, Normal Recall=0.8120, Normal Precision=0.9158, Attack Recall=0.8880, Attack Precision=0.7590

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
0.15       0.8495   0.8550   0.8110   0.8786   0.8880   0.8245   <--
0.20       0.8495   0.8550   0.8110   0.8786   0.8880   0.8245  
0.25       0.8495   0.8550   0.8110   0.8786   0.8880   0.8245  
0.30       0.8495   0.8550   0.8110   0.8786   0.8880   0.8245  
0.35       0.8495   0.8550   0.8110   0.8786   0.8880   0.8245  
0.40       0.8495   0.8550   0.8110   0.8786   0.8880   0.8245  
0.45       0.8495   0.8550   0.8110   0.8786   0.8880   0.8245  
0.50       0.8495   0.8550   0.8110   0.8786   0.8880   0.8245  
0.55       0.8495   0.8550   0.8110   0.8786   0.8880   0.8245  
0.60       0.8495   0.8550   0.8110   0.8786   0.8880   0.8245  
0.65       0.8495   0.8550   0.8110   0.8786   0.8880   0.8245  
0.70       0.8495   0.8550   0.8110   0.8786   0.8880   0.8245  
0.75       0.8495   0.8550   0.8110   0.8786   0.8880   0.8245  
0.80       0.8495   0.8550   0.8110   0.8786   0.8880   0.8245  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8495, F1=0.8550, Normal Recall=0.8110, Normal Precision=0.8786, Attack Recall=0.8880, Attack Precision=0.8245

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
0.15       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438   <--
0.20       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.25       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.30       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.35       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.40       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.45       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.50       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.55       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.60       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.65       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.70       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.75       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.80       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8197, F1=0.4950, Normal Recall=0.8126, Normal Precision=0.9844, Attack Recall=0.8838, Attack Precision=0.3438

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
0.15       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408   <--
0.20       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.25       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.30       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.35       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.40       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.45       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.50       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.55       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.60       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.65       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.70       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.75       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.80       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8266, F1=0.6705, Normal Recall=0.8128, Normal Precision=0.9650, Attack Recall=0.8820, Attack Precision=0.5408

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
0.15       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686   <--
0.20       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.25       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.30       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.35       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.40       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.45       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.50       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.55       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.60       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.65       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.70       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.75       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.80       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8335, F1=0.7606, Normal Recall=0.8127, Normal Precision=0.9414, Attack Recall=0.8820, Attack Precision=0.6686

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
0.15       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579   <--
0.20       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.25       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.30       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.35       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.40       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.45       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.50       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.55       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.60       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.65       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.70       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.75       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.80       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8401, F1=0.8152, Normal Recall=0.8122, Normal Precision=0.9117, Attack Recall=0.8820, Attack Precision=0.7579

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
0.15       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237   <--
0.20       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.25       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.30       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.35       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.40       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.45       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.50       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.55       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.60       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.65       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.70       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.75       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.80       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8466, F1=0.8518, Normal Recall=0.8112, Normal Precision=0.8730, Attack Recall=0.8820, Attack Precision=0.8237

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
0.15       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438   <--
0.20       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.25       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.30       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.35       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.40       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.45       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.50       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.55       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.60       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.65       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.70       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.75       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
0.80       0.8197   0.4950   0.8126   0.9844   0.8838   0.3438  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8197, F1=0.4950, Normal Recall=0.8126, Normal Precision=0.9844, Attack Recall=0.8838, Attack Precision=0.3438

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
0.15       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408   <--
0.20       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.25       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.30       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.35       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.40       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.45       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.50       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.55       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.60       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.65       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.70       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.75       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
0.80       0.8266   0.6705   0.8128   0.9650   0.8820   0.5408  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8266, F1=0.6705, Normal Recall=0.8128, Normal Precision=0.9650, Attack Recall=0.8820, Attack Precision=0.5408

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
0.15       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686   <--
0.20       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.25       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.30       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.35       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.40       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.45       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.50       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.55       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.60       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.65       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.70       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.75       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
0.80       0.8335   0.7606   0.8127   0.9414   0.8820   0.6686  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8335, F1=0.7606, Normal Recall=0.8127, Normal Precision=0.9414, Attack Recall=0.8820, Attack Precision=0.6686

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
0.15       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579   <--
0.20       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.25       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.30       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.35       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.40       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.45       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.50       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.55       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.60       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.65       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.70       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.75       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
0.80       0.8401   0.8152   0.8122   0.9117   0.8820   0.7579  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8401, F1=0.8152, Normal Recall=0.8122, Normal Precision=0.9117, Attack Recall=0.8820, Attack Precision=0.7579

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
0.15       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237   <--
0.20       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.25       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.30       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.35       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.40       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.45       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.50       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.55       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.60       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.65       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.70       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.75       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
0.80       0.8466   0.8518   0.8112   0.8730   0.8820   0.8237  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8466, F1=0.8518, Normal Recall=0.8112, Normal Precision=0.8730, Attack Recall=0.8820, Attack Precision=0.8237

```

