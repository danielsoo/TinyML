# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e2_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-12 17:50:10 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 60 |
| **Local epochs** | 2 |
| **Batch size** | 64 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9216 | 0.9039 | 0.8858 | 0.8688 | 0.8500 | 0.8323 | 0.8154 | 0.7977 | 0.7795 | 0.7617 | 0.7441 |
| QAT+Prune only | 0.8316 | 0.8395 | 0.8476 | 0.8566 | 0.8645 | 0.8719 | 0.8805 | 0.8888 | 0.8964 | 0.9054 | 0.9138 |
| QAT+PTQ | 0.8311 | 0.8384 | 0.8458 | 0.8542 | 0.8616 | 0.8683 | 0.8763 | 0.8838 | 0.8910 | 0.8993 | 0.9072 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8311 | 0.8384 | 0.8458 | 0.8542 | 0.8616 | 0.8683 | 0.8763 | 0.8838 | 0.8910 | 0.8993 | 0.9072 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6072 | 0.7226 | 0.7728 | 0.7987 | 0.8161 | 0.8287 | 0.8374 | 0.8438 | 0.8490 | 0.8533 |
| QAT+Prune only | 0.0000 | 0.5324 | 0.7058 | 0.7927 | 0.8437 | 0.8771 | 0.9018 | 0.9200 | 0.9338 | 0.9456 | 0.9550 |
| QAT+PTQ | 0.0000 | 0.5290 | 0.7018 | 0.7888 | 0.8399 | 0.8732 | 0.8979 | 0.9162 | 0.9302 | 0.9419 | 0.9513 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5290 | 0.7018 | 0.7888 | 0.8399 | 0.8732 | 0.8979 | 0.9162 | 0.9302 | 0.9419 | 0.9513 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9216 | 0.9217 | 0.9212 | 0.9222 | 0.9206 | 0.9205 | 0.9224 | 0.9229 | 0.9215 | 0.9205 | 0.0000 |
| QAT+Prune only | 0.8316 | 0.8312 | 0.8311 | 0.8321 | 0.8316 | 0.8300 | 0.8306 | 0.8302 | 0.8266 | 0.8295 | 0.0000 |
| QAT+PTQ | 0.8311 | 0.8307 | 0.8305 | 0.8315 | 0.8313 | 0.8294 | 0.8299 | 0.8292 | 0.8265 | 0.8287 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8311 | 0.8307 | 0.8305 | 0.8315 | 0.8313 | 0.8294 | 0.8299 | 0.8292 | 0.8265 | 0.8287 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9216 | 0.0000 | 0.0000 | 0.0000 | 0.9216 | 1.0000 |
| 90 | 10 | 299,940 | 0.9039 | 0.5133 | 0.7431 | 0.6072 | 0.9217 | 0.9700 |
| 80 | 20 | 291,350 | 0.8858 | 0.7024 | 0.7441 | 0.7226 | 0.9212 | 0.9351 |
| 70 | 30 | 194,230 | 0.8688 | 0.8039 | 0.7441 | 0.7728 | 0.9222 | 0.8937 |
| 60 | 40 | 145,675 | 0.8500 | 0.8620 | 0.7441 | 0.7987 | 0.9206 | 0.8436 |
| 50 | 50 | 116,540 | 0.8323 | 0.9035 | 0.7441 | 0.8161 | 0.9205 | 0.7825 |
| 40 | 60 | 97,115 | 0.8154 | 0.9350 | 0.7441 | 0.8287 | 0.9224 | 0.7061 |
| 30 | 70 | 83,240 | 0.7977 | 0.9575 | 0.7441 | 0.8374 | 0.9229 | 0.6072 |
| 20 | 80 | 72,835 | 0.7795 | 0.9743 | 0.7441 | 0.8438 | 0.9215 | 0.4737 |
| 10 | 90 | 64,740 | 0.7617 | 0.9883 | 0.7441 | 0.8490 | 0.9205 | 0.2855 |
| 0 | 100 | 58,270 | 0.7441 | 1.0000 | 0.7441 | 0.8533 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8316 | 0.0000 | 0.0000 | 0.0000 | 0.8316 | 1.0000 |
| 90 | 10 | 299,940 | 0.8395 | 0.3756 | 0.9140 | 0.5324 | 0.8312 | 0.9886 |
| 80 | 20 | 291,350 | 0.8476 | 0.5749 | 0.9138 | 0.7058 | 0.8311 | 0.9747 |
| 70 | 30 | 194,230 | 0.8566 | 0.6999 | 0.9138 | 0.7927 | 0.8321 | 0.9575 |
| 60 | 40 | 145,675 | 0.8645 | 0.7835 | 0.9138 | 0.8437 | 0.8316 | 0.9354 |
| 50 | 50 | 116,540 | 0.8719 | 0.8432 | 0.9138 | 0.8771 | 0.8300 | 0.9060 |
| 40 | 60 | 97,115 | 0.8805 | 0.8900 | 0.9138 | 0.9018 | 0.8306 | 0.8654 |
| 30 | 70 | 83,240 | 0.8888 | 0.9263 | 0.9138 | 0.9200 | 0.8302 | 0.8051 |
| 20 | 80 | 72,835 | 0.8964 | 0.9547 | 0.9139 | 0.9338 | 0.8266 | 0.7058 |
| 10 | 90 | 64,740 | 0.9054 | 0.9797 | 0.9138 | 0.9456 | 0.8295 | 0.5168 |
| 0 | 100 | 58,270 | 0.9138 | 1.0000 | 0.9138 | 0.9550 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8311 | 0.0000 | 0.0000 | 0.0000 | 0.8311 | 1.0000 |
| 90 | 10 | 299,940 | 0.8384 | 0.3733 | 0.9076 | 0.5290 | 0.8307 | 0.9878 |
| 80 | 20 | 291,350 | 0.8458 | 0.5723 | 0.9072 | 0.7018 | 0.8305 | 0.9728 |
| 70 | 30 | 194,230 | 0.8542 | 0.6977 | 0.9072 | 0.7888 | 0.8315 | 0.9543 |
| 60 | 40 | 145,675 | 0.8616 | 0.7819 | 0.9072 | 0.8399 | 0.8313 | 0.9307 |
| 50 | 50 | 116,540 | 0.8683 | 0.8417 | 0.9072 | 0.8732 | 0.8294 | 0.8994 |
| 40 | 60 | 97,115 | 0.8763 | 0.8889 | 0.9072 | 0.8979 | 0.8299 | 0.8563 |
| 30 | 70 | 83,240 | 0.8838 | 0.9254 | 0.9072 | 0.9162 | 0.8292 | 0.7929 |
| 20 | 80 | 72,835 | 0.8910 | 0.9544 | 0.9072 | 0.9302 | 0.8265 | 0.6900 |
| 10 | 90 | 64,740 | 0.8993 | 0.9795 | 0.9072 | 0.9419 | 0.8287 | 0.4980 |
| 0 | 100 | 58,270 | 0.9072 | 1.0000 | 0.9072 | 0.9513 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8311 | 0.0000 | 0.0000 | 0.0000 | 0.8311 | 1.0000 |
| 90 | 10 | 299,940 | 0.8384 | 0.3733 | 0.9076 | 0.5290 | 0.8307 | 0.9878 |
| 80 | 20 | 291,350 | 0.8458 | 0.5723 | 0.9072 | 0.7018 | 0.8305 | 0.9728 |
| 70 | 30 | 194,230 | 0.8542 | 0.6977 | 0.9072 | 0.7888 | 0.8315 | 0.9543 |
| 60 | 40 | 145,675 | 0.8616 | 0.7819 | 0.9072 | 0.8399 | 0.8313 | 0.9307 |
| 50 | 50 | 116,540 | 0.8683 | 0.8417 | 0.9072 | 0.8732 | 0.8294 | 0.8994 |
| 40 | 60 | 97,115 | 0.8763 | 0.8889 | 0.9072 | 0.8979 | 0.8299 | 0.8563 |
| 30 | 70 | 83,240 | 0.8838 | 0.9254 | 0.9072 | 0.9162 | 0.8292 | 0.7929 |
| 20 | 80 | 72,835 | 0.8910 | 0.9544 | 0.9072 | 0.9302 | 0.8265 | 0.6900 |
| 10 | 90 | 64,740 | 0.8993 | 0.9795 | 0.9072 | 0.9419 | 0.8287 | 0.4980 |
| 0 | 100 | 58,270 | 0.9072 | 1.0000 | 0.9072 | 0.9513 | 0.0000 | 0.0000 |


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
0.15       0.9038   0.6069   0.9217   0.9699   0.7426   0.5132   <--
0.20       0.9038   0.6069   0.9217   0.9699   0.7426   0.5132  
0.25       0.9038   0.6069   0.9217   0.9699   0.7426   0.5132  
0.30       0.9038   0.6069   0.9217   0.9699   0.7426   0.5132  
0.35       0.9038   0.6069   0.9217   0.9699   0.7426   0.5132  
0.40       0.9038   0.6069   0.9217   0.9699   0.7426   0.5132  
0.45       0.9038   0.6069   0.9217   0.9699   0.7426   0.5132  
0.50       0.9038   0.6069   0.9217   0.9699   0.7426   0.5132  
0.55       0.9038   0.6069   0.9217   0.9699   0.7426   0.5132  
0.60       0.9038   0.6069   0.9217   0.9699   0.7426   0.5132  
0.65       0.9038   0.6069   0.9217   0.9699   0.7426   0.5132  
0.70       0.9038   0.6069   0.9217   0.9699   0.7426   0.5132  
0.75       0.9038   0.6069   0.9217   0.9699   0.7426   0.5132  
0.80       0.9038   0.6069   0.9217   0.9699   0.7426   0.5132  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9038, F1=0.6069, Normal Recall=0.9217, Normal Precision=0.9699, Attack Recall=0.7426, Attack Precision=0.5132

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
0.15       0.8862   0.7233   0.9217   0.9351   0.7441   0.7037   <--
0.20       0.8862   0.7233   0.9217   0.9351   0.7441   0.7037  
0.25       0.8862   0.7233   0.9217   0.9351   0.7441   0.7037  
0.30       0.8862   0.7233   0.9217   0.9351   0.7441   0.7037  
0.35       0.8862   0.7233   0.9217   0.9351   0.7441   0.7037  
0.40       0.8862   0.7233   0.9217   0.9351   0.7441   0.7037  
0.45       0.8862   0.7233   0.9217   0.9351   0.7441   0.7037  
0.50       0.8862   0.7233   0.9217   0.9351   0.7441   0.7037  
0.55       0.8862   0.7233   0.9217   0.9351   0.7441   0.7037  
0.60       0.8862   0.7233   0.9217   0.9351   0.7441   0.7037  
0.65       0.8862   0.7233   0.9217   0.9351   0.7441   0.7037  
0.70       0.8862   0.7233   0.9217   0.9351   0.7441   0.7037  
0.75       0.8862   0.7233   0.9217   0.9351   0.7441   0.7037  
0.80       0.8862   0.7233   0.9217   0.9351   0.7441   0.7037  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8862, F1=0.7233, Normal Recall=0.9217, Normal Precision=0.9351, Attack Recall=0.7441, Attack Precision=0.7037

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
0.15       0.8687   0.7727   0.9221   0.8937   0.7441   0.8036   <--
0.20       0.8687   0.7727   0.9221   0.8937   0.7441   0.8036  
0.25       0.8687   0.7727   0.9221   0.8937   0.7441   0.8036  
0.30       0.8687   0.7727   0.9221   0.8937   0.7441   0.8036  
0.35       0.8687   0.7727   0.9221   0.8937   0.7441   0.8036  
0.40       0.8687   0.7727   0.9221   0.8937   0.7441   0.8036  
0.45       0.8687   0.7727   0.9221   0.8937   0.7441   0.8036  
0.50       0.8687   0.7727   0.9221   0.8937   0.7441   0.8036  
0.55       0.8687   0.7727   0.9221   0.8937   0.7441   0.8036  
0.60       0.8687   0.7727   0.9221   0.8937   0.7441   0.8036  
0.65       0.8687   0.7727   0.9221   0.8937   0.7441   0.8036  
0.70       0.8687   0.7727   0.9221   0.8937   0.7441   0.8036  
0.75       0.8687   0.7727   0.9221   0.8937   0.7441   0.8036  
0.80       0.8687   0.7727   0.9221   0.8937   0.7441   0.8036  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8687, F1=0.7727, Normal Recall=0.9221, Normal Precision=0.8937, Attack Recall=0.7441, Attack Precision=0.8036

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
0.15       0.8509   0.7997   0.9221   0.8439   0.7441   0.8642   <--
0.20       0.8509   0.7997   0.9221   0.8439   0.7441   0.8642  
0.25       0.8509   0.7997   0.9221   0.8439   0.7441   0.8642  
0.30       0.8509   0.7997   0.9221   0.8439   0.7441   0.8642  
0.35       0.8509   0.7997   0.9221   0.8439   0.7441   0.8642  
0.40       0.8509   0.7997   0.9221   0.8439   0.7441   0.8642  
0.45       0.8509   0.7997   0.9221   0.8439   0.7441   0.8642  
0.50       0.8509   0.7997   0.9221   0.8439   0.7441   0.8642  
0.55       0.8509   0.7997   0.9221   0.8439   0.7441   0.8642  
0.60       0.8509   0.7997   0.9221   0.8439   0.7441   0.8642  
0.65       0.8509   0.7997   0.9221   0.8439   0.7441   0.8642  
0.70       0.8509   0.7997   0.9221   0.8439   0.7441   0.8642  
0.75       0.8509   0.7997   0.9221   0.8439   0.7441   0.8642  
0.80       0.8509   0.7997   0.9221   0.8439   0.7441   0.8642  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8509, F1=0.7997, Normal Recall=0.9221, Normal Precision=0.8439, Attack Recall=0.7441, Attack Precision=0.8642

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
0.15       0.8329   0.8166   0.9217   0.7827   0.7441   0.9048   <--
0.20       0.8329   0.8166   0.9217   0.7827   0.7441   0.9048  
0.25       0.8329   0.8166   0.9217   0.7827   0.7441   0.9048  
0.30       0.8329   0.8166   0.9217   0.7827   0.7441   0.9048  
0.35       0.8329   0.8166   0.9217   0.7827   0.7441   0.9048  
0.40       0.8329   0.8166   0.9217   0.7827   0.7441   0.9048  
0.45       0.8329   0.8166   0.9217   0.7827   0.7441   0.9048  
0.50       0.8329   0.8166   0.9217   0.7827   0.7441   0.9048  
0.55       0.8329   0.8166   0.9217   0.7827   0.7441   0.9048  
0.60       0.8329   0.8166   0.9217   0.7827   0.7441   0.9048  
0.65       0.8329   0.8166   0.9217   0.7827   0.7441   0.9048  
0.70       0.8329   0.8166   0.9217   0.7827   0.7441   0.9048  
0.75       0.8329   0.8166   0.9217   0.7827   0.7441   0.9048  
0.80       0.8329   0.8166   0.9217   0.7827   0.7441   0.9048  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8329, F1=0.8166, Normal Recall=0.9217, Normal Precision=0.7827, Attack Recall=0.7441, Attack Precision=0.9048

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
0.15       0.8394   0.5324   0.8312   0.9886   0.9138   0.3756   <--
0.20       0.8394   0.5324   0.8312   0.9886   0.9138   0.3756  
0.25       0.8394   0.5324   0.8312   0.9886   0.9138   0.3756  
0.30       0.8394   0.5324   0.8312   0.9886   0.9138   0.3756  
0.35       0.8394   0.5324   0.8312   0.9886   0.9138   0.3756  
0.40       0.8394   0.5324   0.8312   0.9886   0.9138   0.3756  
0.45       0.8394   0.5324   0.8312   0.9886   0.9138   0.3756  
0.50       0.8394   0.5324   0.8312   0.9886   0.9138   0.3756  
0.55       0.8394   0.5324   0.8312   0.9886   0.9138   0.3756  
0.60       0.8394   0.5324   0.8312   0.9886   0.9138   0.3756  
0.65       0.8394   0.5324   0.8312   0.9886   0.9138   0.3756  
0.70       0.8394   0.5324   0.8312   0.9886   0.9138   0.3756  
0.75       0.8394   0.5324   0.8312   0.9886   0.9138   0.3756  
0.80       0.8394   0.5324   0.8312   0.9886   0.9138   0.3756  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8394, F1=0.5324, Normal Recall=0.8312, Normal Precision=0.9886, Attack Recall=0.9138, Attack Precision=0.3756

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
0.15       0.8482   0.7065   0.8317   0.9748   0.9138   0.5759   <--
0.20       0.8482   0.7065   0.8317   0.9748   0.9138   0.5759  
0.25       0.8482   0.7065   0.8317   0.9748   0.9138   0.5759  
0.30       0.8482   0.7065   0.8317   0.9748   0.9138   0.5759  
0.35       0.8482   0.7065   0.8317   0.9748   0.9138   0.5759  
0.40       0.8482   0.7065   0.8317   0.9748   0.9138   0.5759  
0.45       0.8482   0.7065   0.8317   0.9748   0.9138   0.5759  
0.50       0.8482   0.7065   0.8317   0.9748   0.9138   0.5759  
0.55       0.8482   0.7065   0.8317   0.9748   0.9138   0.5759  
0.60       0.8482   0.7065   0.8317   0.9748   0.9138   0.5759  
0.65       0.8482   0.7065   0.8317   0.9748   0.9138   0.5759  
0.70       0.8482   0.7065   0.8317   0.9748   0.9138   0.5759  
0.75       0.8482   0.7065   0.8317   0.9748   0.9138   0.5759  
0.80       0.8482   0.7065   0.8317   0.9748   0.9138   0.5759  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8482, F1=0.7065, Normal Recall=0.8317, Normal Precision=0.9748, Attack Recall=0.9138, Attack Precision=0.5759

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
0.15       0.8561   0.7921   0.8313   0.9575   0.9138   0.6990   <--
0.20       0.8561   0.7921   0.8313   0.9575   0.9138   0.6990  
0.25       0.8561   0.7921   0.8313   0.9575   0.9138   0.6990  
0.30       0.8561   0.7921   0.8313   0.9575   0.9138   0.6990  
0.35       0.8561   0.7921   0.8313   0.9575   0.9138   0.6990  
0.40       0.8561   0.7921   0.8313   0.9575   0.9138   0.6990  
0.45       0.8561   0.7921   0.8313   0.9575   0.9138   0.6990  
0.50       0.8561   0.7921   0.8313   0.9575   0.9138   0.6990  
0.55       0.8561   0.7921   0.8313   0.9575   0.9138   0.6990  
0.60       0.8561   0.7921   0.8313   0.9575   0.9138   0.6990  
0.65       0.8561   0.7921   0.8313   0.9575   0.9138   0.6990  
0.70       0.8561   0.7921   0.8313   0.9575   0.9138   0.6990  
0.75       0.8561   0.7921   0.8313   0.9575   0.9138   0.6990  
0.80       0.8561   0.7921   0.8313   0.9575   0.9138   0.6990  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8561, F1=0.7921, Normal Recall=0.8313, Normal Precision=0.9575, Attack Recall=0.9138, Attack Precision=0.6990

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
0.15       0.8644   0.8436   0.8315   0.9354   0.9138   0.7834   <--
0.20       0.8644   0.8436   0.8315   0.9354   0.9138   0.7834  
0.25       0.8644   0.8436   0.8315   0.9354   0.9138   0.7834  
0.30       0.8644   0.8436   0.8315   0.9354   0.9138   0.7834  
0.35       0.8644   0.8436   0.8315   0.9354   0.9138   0.7834  
0.40       0.8644   0.8436   0.8315   0.9354   0.9138   0.7834  
0.45       0.8644   0.8436   0.8315   0.9354   0.9138   0.7834  
0.50       0.8644   0.8436   0.8315   0.9354   0.9138   0.7834  
0.55       0.8644   0.8436   0.8315   0.9354   0.9138   0.7834  
0.60       0.8644   0.8436   0.8315   0.9354   0.9138   0.7834  
0.65       0.8644   0.8436   0.8315   0.9354   0.9138   0.7834  
0.70       0.8644   0.8436   0.8315   0.9354   0.9138   0.7834  
0.75       0.8644   0.8436   0.8315   0.9354   0.9138   0.7834  
0.80       0.8644   0.8436   0.8315   0.9354   0.9138   0.7834  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8644, F1=0.8436, Normal Recall=0.8315, Normal Precision=0.9354, Attack Recall=0.9138, Attack Precision=0.7834

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
0.15       0.8715   0.8767   0.8292   0.9059   0.9138   0.8425   <--
0.20       0.8715   0.8767   0.8292   0.9059   0.9138   0.8425  
0.25       0.8715   0.8767   0.8292   0.9059   0.9138   0.8425  
0.30       0.8715   0.8767   0.8292   0.9059   0.9138   0.8425  
0.35       0.8715   0.8767   0.8292   0.9059   0.9138   0.8425  
0.40       0.8715   0.8767   0.8292   0.9059   0.9138   0.8425  
0.45       0.8715   0.8767   0.8292   0.9059   0.9138   0.8425  
0.50       0.8715   0.8767   0.8292   0.9059   0.9138   0.8425  
0.55       0.8715   0.8767   0.8292   0.9059   0.9138   0.8425  
0.60       0.8715   0.8767   0.8292   0.9059   0.9138   0.8425  
0.65       0.8715   0.8767   0.8292   0.9059   0.9138   0.8425  
0.70       0.8715   0.8767   0.8292   0.9059   0.9138   0.8425  
0.75       0.8715   0.8767   0.8292   0.9059   0.9138   0.8425  
0.80       0.8715   0.8767   0.8292   0.9059   0.9138   0.8425  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8715, F1=0.8767, Normal Recall=0.8292, Normal Precision=0.9059, Attack Recall=0.9138, Attack Precision=0.8425

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
0.15       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732   <--
0.20       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.25       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.30       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.35       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.40       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.45       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.50       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.55       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.60       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.65       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.70       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.75       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.80       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8383, F1=0.5288, Normal Recall=0.8307, Normal Precision=0.9877, Attack Recall=0.9071, Attack Precision=0.3732

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
0.15       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733   <--
0.20       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.25       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.30       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.35       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.40       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.45       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.50       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.55       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.60       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.65       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.70       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.75       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.80       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8464, F1=0.7026, Normal Recall=0.8312, Normal Precision=0.9728, Attack Recall=0.9072, Attack Precision=0.5733

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
0.15       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968   <--
0.20       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.25       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.30       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.35       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.40       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.45       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.50       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.55       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.60       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.65       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.70       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.75       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.80       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8537, F1=0.7882, Normal Recall=0.8308, Normal Precision=0.9543, Attack Recall=0.9072, Attack Precision=0.6968

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
0.15       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816   <--
0.20       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.25       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.30       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.35       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.40       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.45       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.50       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.55       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.60       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.65       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.70       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.75       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.80       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8615, F1=0.8397, Normal Recall=0.8310, Normal Precision=0.9307, Attack Recall=0.9072, Attack Precision=0.7816

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
0.15       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412   <--
0.20       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.25       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.30       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.35       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.40       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.45       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.50       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.55       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.60       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.65       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.70       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.75       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.80       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8680, F1=0.8729, Normal Recall=0.8288, Normal Precision=0.8993, Attack Recall=0.9072, Attack Precision=0.8412

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
0.15       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732   <--
0.20       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.25       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.30       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.35       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.40       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.45       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.50       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.55       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.60       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.65       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.70       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.75       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
0.80       0.8383   0.5288   0.8307   0.9877   0.9071   0.3732  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8383, F1=0.5288, Normal Recall=0.8307, Normal Precision=0.9877, Attack Recall=0.9071, Attack Precision=0.3732

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
0.15       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733   <--
0.20       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.25       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.30       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.35       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.40       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.45       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.50       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.55       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.60       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.65       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.70       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.75       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
0.80       0.8464   0.7026   0.8312   0.9728   0.9072   0.5733  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8464, F1=0.7026, Normal Recall=0.8312, Normal Precision=0.9728, Attack Recall=0.9072, Attack Precision=0.5733

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
0.15       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968   <--
0.20       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.25       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.30       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.35       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.40       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.45       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.50       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.55       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.60       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.65       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.70       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.75       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
0.80       0.8537   0.7882   0.8308   0.9543   0.9072   0.6968  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8537, F1=0.7882, Normal Recall=0.8308, Normal Precision=0.9543, Attack Recall=0.9072, Attack Precision=0.6968

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
0.15       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816   <--
0.20       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.25       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.30       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.35       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.40       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.45       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.50       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.55       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.60       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.65       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.70       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.75       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
0.80       0.8615   0.8397   0.8310   0.9307   0.9072   0.7816  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8615, F1=0.8397, Normal Recall=0.8310, Normal Precision=0.9307, Attack Recall=0.9072, Attack Precision=0.7816

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
0.15       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412   <--
0.20       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.25       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.30       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.35       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.40       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.45       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.50       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.55       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.60       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.65       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.70       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.75       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
0.80       0.8680   0.8729   0.8288   0.8993   0.9072   0.8412  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8680, F1=0.8729, Normal Recall=0.8288, Normal Precision=0.8993, Attack Recall=0.9072, Attack Precision=0.8412

```

