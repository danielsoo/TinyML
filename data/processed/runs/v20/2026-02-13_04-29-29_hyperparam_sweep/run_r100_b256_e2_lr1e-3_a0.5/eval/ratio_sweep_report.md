# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e2_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-22 21:16:48 |

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
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2443 | 0.3193 | 0.3942 | 0.4696 | 0.5450 | 0.6191 | 0.6953 | 0.7705 | 0.8455 | 0.9212 | 0.9962 |
| QAT+Prune only | 0.7894 | 0.8032 | 0.8163 | 0.8302 | 0.8429 | 0.8547 | 0.8695 | 0.8816 | 0.8955 | 0.9082 | 0.9220 |
| QAT+PTQ | 0.7899 | 0.8037 | 0.8167 | 0.8305 | 0.8433 | 0.8550 | 0.8700 | 0.8817 | 0.8958 | 0.9083 | 0.9220 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7899 | 0.8037 | 0.8167 | 0.8305 | 0.8433 | 0.8550 | 0.8700 | 0.8817 | 0.8958 | 0.9083 | 0.9220 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2265 | 0.3968 | 0.5298 | 0.6366 | 0.7234 | 0.7969 | 0.8587 | 0.9116 | 0.9579 | 0.9981 |
| QAT+Prune only | 0.0000 | 0.4837 | 0.6675 | 0.7651 | 0.8244 | 0.8639 | 0.8945 | 0.9160 | 0.9339 | 0.9476 | 0.9594 |
| QAT+PTQ | 0.0000 | 0.4843 | 0.6680 | 0.7655 | 0.8248 | 0.8641 | 0.8948 | 0.9161 | 0.9340 | 0.9476 | 0.9594 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4843 | 0.6680 | 0.7655 | 0.8248 | 0.8641 | 0.8948 | 0.9161 | 0.9340 | 0.9476 | 0.9594 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2443 | 0.2441 | 0.2437 | 0.2439 | 0.2442 | 0.2419 | 0.2439 | 0.2438 | 0.2428 | 0.2461 | 0.0000 |
| QAT+Prune only | 0.7894 | 0.7901 | 0.7898 | 0.7908 | 0.7902 | 0.7874 | 0.7907 | 0.7874 | 0.7894 | 0.7841 | 0.0000 |
| QAT+PTQ | 0.7899 | 0.7905 | 0.7904 | 0.7912 | 0.7908 | 0.7880 | 0.7919 | 0.7876 | 0.7906 | 0.7847 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7899 | 0.7905 | 0.7904 | 0.7912 | 0.7908 | 0.7880 | 0.7919 | 0.7876 | 0.7906 | 0.7847 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2443 | 0.0000 | 0.0000 | 0.0000 | 0.2443 | 1.0000 |
| 90 | 10 | 299,940 | 0.3193 | 0.1277 | 0.9963 | 0.2265 | 0.2441 | 0.9983 |
| 80 | 20 | 291,350 | 0.3942 | 0.2477 | 0.9962 | 0.3968 | 0.2437 | 0.9961 |
| 70 | 30 | 194,230 | 0.4696 | 0.3609 | 0.9962 | 0.5298 | 0.2439 | 0.9933 |
| 60 | 40 | 145,675 | 0.5450 | 0.4677 | 0.9962 | 0.6366 | 0.2442 | 0.9897 |
| 50 | 50 | 116,540 | 0.6191 | 0.5679 | 0.9962 | 0.7234 | 0.2419 | 0.9845 |
| 40 | 60 | 97,115 | 0.6953 | 0.6640 | 0.9962 | 0.7969 | 0.2439 | 0.9771 |
| 30 | 70 | 83,240 | 0.7705 | 0.7545 | 0.9962 | 0.8587 | 0.2438 | 0.9648 |
| 20 | 80 | 72,835 | 0.8455 | 0.8403 | 0.9962 | 0.9116 | 0.2428 | 0.9409 |
| 10 | 90 | 64,740 | 0.9212 | 0.9224 | 0.9962 | 0.9579 | 0.2461 | 0.8777 |
| 0 | 100 | 58,270 | 0.9962 | 1.0000 | 0.9962 | 0.9981 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7894 | 0.0000 | 0.0000 | 0.0000 | 0.7894 | 1.0000 |
| 90 | 10 | 299,940 | 0.8032 | 0.3279 | 0.9218 | 0.4837 | 0.7901 | 0.9891 |
| 80 | 20 | 291,350 | 0.8163 | 0.5231 | 0.9220 | 0.6675 | 0.7898 | 0.9759 |
| 70 | 30 | 194,230 | 0.8302 | 0.6539 | 0.9220 | 0.7651 | 0.7908 | 0.9595 |
| 60 | 40 | 145,675 | 0.8429 | 0.7455 | 0.9220 | 0.8244 | 0.7902 | 0.9383 |
| 50 | 50 | 116,540 | 0.8547 | 0.8126 | 0.9220 | 0.8639 | 0.7874 | 0.9099 |
| 40 | 60 | 97,115 | 0.8695 | 0.8686 | 0.9220 | 0.8945 | 0.7907 | 0.8711 |
| 30 | 70 | 83,240 | 0.8816 | 0.9101 | 0.9220 | 0.9160 | 0.7874 | 0.8123 |
| 20 | 80 | 72,835 | 0.8955 | 0.9460 | 0.9220 | 0.9339 | 0.7894 | 0.7168 |
| 10 | 90 | 64,740 | 0.9082 | 0.9746 | 0.9220 | 0.9476 | 0.7841 | 0.5277 |
| 0 | 100 | 58,270 | 0.9220 | 1.0000 | 0.9220 | 0.9594 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7899 | 0.0000 | 0.0000 | 0.0000 | 0.7899 | 1.0000 |
| 90 | 10 | 299,940 | 0.8037 | 0.3284 | 0.9218 | 0.4843 | 0.7905 | 0.9891 |
| 80 | 20 | 291,350 | 0.8167 | 0.5237 | 0.9220 | 0.6680 | 0.7904 | 0.9759 |
| 70 | 30 | 194,230 | 0.8305 | 0.6543 | 0.9220 | 0.7655 | 0.7912 | 0.9595 |
| 60 | 40 | 145,675 | 0.8433 | 0.7461 | 0.9220 | 0.8248 | 0.7908 | 0.9383 |
| 50 | 50 | 116,540 | 0.8550 | 0.8131 | 0.9220 | 0.8641 | 0.7880 | 0.9100 |
| 40 | 60 | 97,115 | 0.8700 | 0.8692 | 0.9220 | 0.8948 | 0.7919 | 0.8713 |
| 30 | 70 | 83,240 | 0.8817 | 0.9102 | 0.9220 | 0.9161 | 0.7876 | 0.8124 |
| 20 | 80 | 72,835 | 0.8958 | 0.9463 | 0.9220 | 0.9340 | 0.7906 | 0.7172 |
| 10 | 90 | 64,740 | 0.9083 | 0.9747 | 0.9220 | 0.9476 | 0.7847 | 0.5279 |
| 0 | 100 | 58,270 | 0.9220 | 1.0000 | 0.9220 | 0.9594 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7899 | 0.0000 | 0.0000 | 0.0000 | 0.7899 | 1.0000 |
| 90 | 10 | 299,940 | 0.8037 | 0.3284 | 0.9218 | 0.4843 | 0.7905 | 0.9891 |
| 80 | 20 | 291,350 | 0.8167 | 0.5237 | 0.9220 | 0.6680 | 0.7904 | 0.9759 |
| 70 | 30 | 194,230 | 0.8305 | 0.6543 | 0.9220 | 0.7655 | 0.7912 | 0.9595 |
| 60 | 40 | 145,675 | 0.8433 | 0.7461 | 0.9220 | 0.8248 | 0.7908 | 0.9383 |
| 50 | 50 | 116,540 | 0.8550 | 0.8131 | 0.9220 | 0.8641 | 0.7880 | 0.9100 |
| 40 | 60 | 97,115 | 0.8700 | 0.8692 | 0.9220 | 0.8948 | 0.7919 | 0.8713 |
| 30 | 70 | 83,240 | 0.8817 | 0.9102 | 0.9220 | 0.9161 | 0.7876 | 0.8124 |
| 20 | 80 | 72,835 | 0.8958 | 0.9463 | 0.9220 | 0.9340 | 0.7906 | 0.7172 |
| 10 | 90 | 64,740 | 0.9083 | 0.9747 | 0.9220 | 0.9476 | 0.7847 | 0.5279 |
| 0 | 100 | 58,270 | 0.9220 | 1.0000 | 0.9220 | 0.9594 | 0.0000 | 0.0000 |


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
0.15       0.3193   0.2264   0.2441   0.9982   0.9960   0.1277   <--
0.20       0.3193   0.2264   0.2441   0.9982   0.9960   0.1277  
0.25       0.3193   0.2264   0.2441   0.9982   0.9960   0.1277  
0.30       0.3193   0.2264   0.2441   0.9982   0.9960   0.1277  
0.35       0.3193   0.2264   0.2441   0.9982   0.9960   0.1277  
0.40       0.3193   0.2264   0.2441   0.9982   0.9960   0.1277  
0.45       0.3193   0.2264   0.2441   0.9982   0.9960   0.1277  
0.50       0.3193   0.2264   0.2441   0.9982   0.9960   0.1277  
0.55       0.3193   0.2264   0.2441   0.9982   0.9960   0.1277  
0.60       0.3193   0.2264   0.2441   0.9982   0.9960   0.1277  
0.65       0.3193   0.2264   0.2441   0.9982   0.9960   0.1277  
0.70       0.3193   0.2264   0.2441   0.9982   0.9960   0.1277  
0.75       0.3193   0.2264   0.2441   0.9982   0.9960   0.1277  
0.80       0.3193   0.2264   0.2441   0.9982   0.9960   0.1277  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3193, F1=0.2264, Normal Recall=0.2441, Normal Precision=0.9982, Attack Recall=0.9960, Attack Precision=0.1277

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
0.15       0.3945   0.3969   0.2440   0.9961   0.9962   0.2478   <--
0.20       0.3945   0.3969   0.2440   0.9961   0.9962   0.2478  
0.25       0.3945   0.3969   0.2440   0.9961   0.9962   0.2478  
0.30       0.3945   0.3969   0.2440   0.9961   0.9962   0.2478  
0.35       0.3945   0.3969   0.2440   0.9961   0.9962   0.2478  
0.40       0.3945   0.3969   0.2440   0.9961   0.9962   0.2478  
0.45       0.3945   0.3969   0.2440   0.9961   0.9962   0.2478  
0.50       0.3945   0.3969   0.2440   0.9961   0.9962   0.2478  
0.55       0.3945   0.3969   0.2440   0.9961   0.9962   0.2478  
0.60       0.3945   0.3969   0.2440   0.9961   0.9962   0.2478  
0.65       0.3945   0.3969   0.2440   0.9961   0.9962   0.2478  
0.70       0.3945   0.3969   0.2440   0.9961   0.9962   0.2478  
0.75       0.3945   0.3969   0.2440   0.9961   0.9962   0.2478  
0.80       0.3945   0.3969   0.2440   0.9961   0.9962   0.2478  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3945, F1=0.3969, Normal Recall=0.2440, Normal Precision=0.9961, Attack Recall=0.9962, Attack Precision=0.2478

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
0.15       0.4704   0.5302   0.2450   0.9934   0.9962   0.3612   <--
0.20       0.4704   0.5302   0.2450   0.9934   0.9962   0.3612  
0.25       0.4704   0.5302   0.2450   0.9934   0.9962   0.3612  
0.30       0.4704   0.5302   0.2450   0.9934   0.9962   0.3612  
0.35       0.4704   0.5302   0.2450   0.9934   0.9962   0.3612  
0.40       0.4704   0.5302   0.2450   0.9934   0.9962   0.3612  
0.45       0.4704   0.5302   0.2450   0.9934   0.9962   0.3612  
0.50       0.4704   0.5302   0.2450   0.9934   0.9962   0.3612  
0.55       0.4704   0.5302   0.2450   0.9934   0.9962   0.3612  
0.60       0.4704   0.5302   0.2450   0.9934   0.9962   0.3612  
0.65       0.4704   0.5302   0.2450   0.9934   0.9962   0.3612  
0.70       0.4704   0.5302   0.2450   0.9934   0.9962   0.3612  
0.75       0.4704   0.5302   0.2450   0.9934   0.9962   0.3612  
0.80       0.4704   0.5302   0.2450   0.9934   0.9962   0.3612  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4704, F1=0.5302, Normal Recall=0.2450, Normal Precision=0.9934, Attack Recall=0.9962, Attack Precision=0.3612

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
0.15       0.5449   0.6365   0.2440   0.9897   0.9962   0.4677   <--
0.20       0.5449   0.6365   0.2440   0.9897   0.9962   0.4677  
0.25       0.5449   0.6365   0.2440   0.9897   0.9962   0.4677  
0.30       0.5449   0.6365   0.2440   0.9897   0.9962   0.4677  
0.35       0.5449   0.6365   0.2440   0.9897   0.9962   0.4677  
0.40       0.5449   0.6365   0.2440   0.9897   0.9962   0.4677  
0.45       0.5449   0.6365   0.2440   0.9897   0.9962   0.4677  
0.50       0.5449   0.6365   0.2440   0.9897   0.9962   0.4677  
0.55       0.5449   0.6365   0.2440   0.9897   0.9962   0.4677  
0.60       0.5449   0.6365   0.2440   0.9897   0.9962   0.4677  
0.65       0.5449   0.6365   0.2440   0.9897   0.9962   0.4677  
0.70       0.5449   0.6365   0.2440   0.9897   0.9962   0.4677  
0.75       0.5449   0.6365   0.2440   0.9897   0.9962   0.4677  
0.80       0.5449   0.6365   0.2440   0.9897   0.9962   0.4677  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5449, F1=0.6365, Normal Recall=0.2440, Normal Precision=0.9897, Attack Recall=0.9962, Attack Precision=0.4677

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
0.15       0.6196   0.7237   0.2430   0.9846   0.9962   0.5682   <--
0.20       0.6196   0.7237   0.2430   0.9846   0.9962   0.5682  
0.25       0.6196   0.7237   0.2430   0.9846   0.9962   0.5682  
0.30       0.6196   0.7237   0.2430   0.9846   0.9962   0.5682  
0.35       0.6196   0.7237   0.2430   0.9846   0.9962   0.5682  
0.40       0.6196   0.7237   0.2430   0.9846   0.9962   0.5682  
0.45       0.6196   0.7237   0.2430   0.9846   0.9962   0.5682  
0.50       0.6196   0.7237   0.2430   0.9846   0.9962   0.5682  
0.55       0.6196   0.7237   0.2430   0.9846   0.9962   0.5682  
0.60       0.6196   0.7237   0.2430   0.9846   0.9962   0.5682  
0.65       0.6196   0.7237   0.2430   0.9846   0.9962   0.5682  
0.70       0.6196   0.7237   0.2430   0.9846   0.9962   0.5682  
0.75       0.6196   0.7237   0.2430   0.9846   0.9962   0.5682  
0.80       0.6196   0.7237   0.2430   0.9846   0.9962   0.5682  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6196, F1=0.7237, Normal Recall=0.2430, Normal Precision=0.9846, Attack Recall=0.9962, Attack Precision=0.5682

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
0.15       0.8033   0.4841   0.7901   0.9892   0.9227   0.3281   <--
0.20       0.8033   0.4841   0.7901   0.9892   0.9227   0.3281  
0.25       0.8033   0.4841   0.7901   0.9892   0.9227   0.3281  
0.30       0.8033   0.4841   0.7901   0.9892   0.9227   0.3281  
0.35       0.8033   0.4841   0.7901   0.9892   0.9227   0.3281  
0.40       0.8033   0.4841   0.7901   0.9892   0.9227   0.3281  
0.45       0.8033   0.4841   0.7901   0.9892   0.9227   0.3281  
0.50       0.8033   0.4841   0.7901   0.9892   0.9227   0.3281  
0.55       0.8033   0.4841   0.7901   0.9892   0.9227   0.3281  
0.60       0.8033   0.4841   0.7901   0.9892   0.9227   0.3281  
0.65       0.8033   0.4841   0.7901   0.9892   0.9227   0.3281  
0.70       0.8033   0.4841   0.7901   0.9892   0.9227   0.3281  
0.75       0.8033   0.4841   0.7901   0.9892   0.9227   0.3281  
0.80       0.8033   0.4841   0.7901   0.9892   0.9227   0.3281  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8033, F1=0.4841, Normal Recall=0.7901, Normal Precision=0.9892, Attack Recall=0.9227, Attack Precision=0.3281

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
0.15       0.8168   0.6681   0.7905   0.9759   0.9220   0.5239   <--
0.20       0.8168   0.6681   0.7905   0.9759   0.9220   0.5239  
0.25       0.8168   0.6681   0.7905   0.9759   0.9220   0.5239  
0.30       0.8168   0.6681   0.7905   0.9759   0.9220   0.5239  
0.35       0.8168   0.6681   0.7905   0.9759   0.9220   0.5239  
0.40       0.8168   0.6681   0.7905   0.9759   0.9220   0.5239  
0.45       0.8168   0.6681   0.7905   0.9759   0.9220   0.5239  
0.50       0.8168   0.6681   0.7905   0.9759   0.9220   0.5239  
0.55       0.8168   0.6681   0.7905   0.9759   0.9220   0.5239  
0.60       0.8168   0.6681   0.7905   0.9759   0.9220   0.5239  
0.65       0.8168   0.6681   0.7905   0.9759   0.9220   0.5239  
0.70       0.8168   0.6681   0.7905   0.9759   0.9220   0.5239  
0.75       0.8168   0.6681   0.7905   0.9759   0.9220   0.5239  
0.80       0.8168   0.6681   0.7905   0.9759   0.9220   0.5239  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8168, F1=0.6681, Normal Recall=0.7905, Normal Precision=0.9759, Attack Recall=0.9220, Attack Precision=0.5239

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
0.15       0.8292   0.7641   0.7894   0.9594   0.9220   0.6523   <--
0.20       0.8292   0.7641   0.7894   0.9594   0.9220   0.6523  
0.25       0.8292   0.7641   0.7894   0.9594   0.9220   0.6523  
0.30       0.8292   0.7641   0.7894   0.9594   0.9220   0.6523  
0.35       0.8292   0.7641   0.7894   0.9594   0.9220   0.6523  
0.40       0.8292   0.7641   0.7894   0.9594   0.9220   0.6523  
0.45       0.8292   0.7641   0.7894   0.9594   0.9220   0.6523  
0.50       0.8292   0.7641   0.7894   0.9594   0.9220   0.6523  
0.55       0.8292   0.7641   0.7894   0.9594   0.9220   0.6523  
0.60       0.8292   0.7641   0.7894   0.9594   0.9220   0.6523  
0.65       0.8292   0.7641   0.7894   0.9594   0.9220   0.6523  
0.70       0.8292   0.7641   0.7894   0.9594   0.9220   0.6523  
0.75       0.8292   0.7641   0.7894   0.9594   0.9220   0.6523  
0.80       0.8292   0.7641   0.7894   0.9594   0.9220   0.6523  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8292, F1=0.7641, Normal Recall=0.7894, Normal Precision=0.9594, Attack Recall=0.9220, Attack Precision=0.6523

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
0.15       0.8424   0.8240   0.7894   0.9382   0.9220   0.7448   <--
0.20       0.8424   0.8240   0.7894   0.9382   0.9220   0.7448  
0.25       0.8424   0.8240   0.7894   0.9382   0.9220   0.7448  
0.30       0.8424   0.8240   0.7894   0.9382   0.9220   0.7448  
0.35       0.8424   0.8240   0.7894   0.9382   0.9220   0.7448  
0.40       0.8424   0.8240   0.7894   0.9382   0.9220   0.7448  
0.45       0.8424   0.8240   0.7894   0.9382   0.9220   0.7448  
0.50       0.8424   0.8240   0.7894   0.9382   0.9220   0.7448  
0.55       0.8424   0.8240   0.7894   0.9382   0.9220   0.7448  
0.60       0.8424   0.8240   0.7894   0.9382   0.9220   0.7448  
0.65       0.8424   0.8240   0.7894   0.9382   0.9220   0.7448  
0.70       0.8424   0.8240   0.7894   0.9382   0.9220   0.7448  
0.75       0.8424   0.8240   0.7894   0.9382   0.9220   0.7448  
0.80       0.8424   0.8240   0.7894   0.9382   0.9220   0.7448  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8424, F1=0.8240, Normal Recall=0.7894, Normal Precision=0.9382, Attack Recall=0.9220, Attack Precision=0.7448

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
0.15       0.8550   0.8641   0.7880   0.9099   0.9220   0.8130   <--
0.20       0.8550   0.8641   0.7880   0.9099   0.9220   0.8130  
0.25       0.8550   0.8641   0.7880   0.9099   0.9220   0.8130  
0.30       0.8550   0.8641   0.7880   0.9099   0.9220   0.8130  
0.35       0.8550   0.8641   0.7880   0.9099   0.9220   0.8130  
0.40       0.8550   0.8641   0.7880   0.9099   0.9220   0.8130  
0.45       0.8550   0.8641   0.7880   0.9099   0.9220   0.8130  
0.50       0.8550   0.8641   0.7880   0.9099   0.9220   0.8130  
0.55       0.8550   0.8641   0.7880   0.9099   0.9220   0.8130  
0.60       0.8550   0.8641   0.7880   0.9099   0.9220   0.8130  
0.65       0.8550   0.8641   0.7880   0.9099   0.9220   0.8130  
0.70       0.8550   0.8641   0.7880   0.9099   0.9220   0.8130  
0.75       0.8550   0.8641   0.7880   0.9099   0.9220   0.8130  
0.80       0.8550   0.8641   0.7880   0.9099   0.9220   0.8130  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8550, F1=0.8641, Normal Recall=0.7880, Normal Precision=0.9099, Attack Recall=0.9220, Attack Precision=0.8130

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
0.15       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286   <--
0.20       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.25       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.30       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.35       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.40       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.45       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.50       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.55       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.60       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.65       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.70       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.75       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.80       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8038, F1=0.4847, Normal Recall=0.7905, Normal Precision=0.9893, Attack Recall=0.9228, Attack Precision=0.3286

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
0.15       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245   <--
0.20       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.25       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.30       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.35       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.40       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.45       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.50       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.55       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.60       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.65       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.70       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.75       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.80       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8173, F1=0.6687, Normal Recall=0.7911, Normal Precision=0.9760, Attack Recall=0.9220, Attack Precision=0.5245

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
0.15       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529   <--
0.20       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.25       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.30       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.35       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.40       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.45       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.50       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.55       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.60       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.65       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.70       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.75       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.80       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8295, F1=0.7644, Normal Recall=0.7899, Normal Precision=0.9594, Attack Recall=0.9220, Attack Precision=0.6529

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
0.15       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453   <--
0.20       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.25       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.30       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.35       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.40       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.45       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.50       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.55       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.60       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.65       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.70       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.75       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.80       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8427, F1=0.8243, Normal Recall=0.7899, Normal Precision=0.9383, Attack Recall=0.9220, Attack Precision=0.7453

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
0.15       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134   <--
0.20       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.25       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.30       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.35       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.40       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.45       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.50       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.55       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.60       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.65       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.70       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.75       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.80       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8553, F1=0.8643, Normal Recall=0.7885, Normal Precision=0.9100, Attack Recall=0.9220, Attack Precision=0.8134

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
0.15       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286   <--
0.20       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.25       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.30       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.35       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.40       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.45       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.50       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.55       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.60       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.65       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.70       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.75       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
0.80       0.8038   0.4847   0.7905   0.9893   0.9228   0.3286  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8038, F1=0.4847, Normal Recall=0.7905, Normal Precision=0.9893, Attack Recall=0.9228, Attack Precision=0.3286

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
0.15       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245   <--
0.20       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.25       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.30       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.35       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.40       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.45       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.50       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.55       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.60       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.65       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.70       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.75       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
0.80       0.8173   0.6687   0.7911   0.9760   0.9220   0.5245  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8173, F1=0.6687, Normal Recall=0.7911, Normal Precision=0.9760, Attack Recall=0.9220, Attack Precision=0.5245

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
0.15       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529   <--
0.20       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.25       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.30       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.35       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.40       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.45       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.50       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.55       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.60       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.65       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.70       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.75       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
0.80       0.8295   0.7644   0.7899   0.9594   0.9220   0.6529  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8295, F1=0.7644, Normal Recall=0.7899, Normal Precision=0.9594, Attack Recall=0.9220, Attack Precision=0.6529

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
0.15       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453   <--
0.20       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.25       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.30       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.35       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.40       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.45       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.50       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.55       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.60       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.65       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.70       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.75       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
0.80       0.8427   0.8243   0.7899   0.9383   0.9220   0.7453  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8427, F1=0.8243, Normal Recall=0.7899, Normal Precision=0.9383, Attack Recall=0.9220, Attack Precision=0.7453

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
0.15       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134   <--
0.20       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.25       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.30       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.35       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.40       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.45       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.50       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.55       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.60       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.65       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.70       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.75       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
0.80       0.8553   0.8643   0.7885   0.9100   0.9220   0.8134  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8553, F1=0.8643, Normal Recall=0.7885, Normal Precision=0.9100, Attack Recall=0.9220, Attack Precision=0.8134

```

