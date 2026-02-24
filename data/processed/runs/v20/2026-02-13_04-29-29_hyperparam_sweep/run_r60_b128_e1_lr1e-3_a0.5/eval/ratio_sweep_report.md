# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e1_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-14 14:04:44 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2612 | 0.3336 | 0.4050 | 0.4771 | 0.5488 | 0.6184 | 0.6914 | 0.7636 | 0.8332 | 0.9069 | 0.9773 |
| QAT+Prune only | 0.7311 | 0.7574 | 0.7833 | 0.8112 | 0.8381 | 0.8632 | 0.8912 | 0.9176 | 0.9446 | 0.9711 | 0.9980 |
| QAT+PTQ | 0.7314 | 0.7577 | 0.7836 | 0.8114 | 0.8384 | 0.8634 | 0.8915 | 0.9177 | 0.9447 | 0.9711 | 0.9980 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7314 | 0.7577 | 0.7836 | 0.8114 | 0.8384 | 0.8634 | 0.8915 | 0.9177 | 0.9447 | 0.9711 | 0.9980 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2269 | 0.3965 | 0.5286 | 0.6341 | 0.7192 | 0.7917 | 0.8527 | 0.9036 | 0.9497 | 0.9885 |
| QAT+Prune only | 0.0000 | 0.4514 | 0.6482 | 0.7603 | 0.8315 | 0.8794 | 0.9167 | 0.9443 | 0.9665 | 0.9842 | 0.9990 |
| QAT+PTQ | 0.0000 | 0.4517 | 0.6485 | 0.7605 | 0.8317 | 0.8796 | 0.9169 | 0.9444 | 0.9665 | 0.9842 | 0.9990 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4517 | 0.6485 | 0.7605 | 0.8317 | 0.8796 | 0.9169 | 0.9444 | 0.9665 | 0.9842 | 0.9990 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2612 | 0.2620 | 0.2620 | 0.2627 | 0.2632 | 0.2595 | 0.2626 | 0.2647 | 0.2568 | 0.2731 | 0.0000 |
| QAT+Prune only | 0.7311 | 0.7306 | 0.7297 | 0.7311 | 0.7316 | 0.7283 | 0.7309 | 0.7300 | 0.7309 | 0.7285 | 0.0000 |
| QAT+PTQ | 0.7314 | 0.7310 | 0.7300 | 0.7315 | 0.7320 | 0.7288 | 0.7317 | 0.7304 | 0.7315 | 0.7289 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7314 | 0.7310 | 0.7300 | 0.7315 | 0.7320 | 0.7288 | 0.7317 | 0.7304 | 0.7315 | 0.7289 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2612 | 0.0000 | 0.0000 | 0.0000 | 0.2612 | 1.0000 |
| 90 | 10 | 299,940 | 0.3336 | 0.1283 | 0.9779 | 0.2269 | 0.2620 | 0.9907 |
| 80 | 20 | 291,350 | 0.4050 | 0.2487 | 0.9773 | 0.3965 | 0.2620 | 0.9788 |
| 70 | 30 | 194,230 | 0.4771 | 0.3623 | 0.9773 | 0.5286 | 0.2627 | 0.9643 |
| 60 | 40 | 145,675 | 0.5488 | 0.4693 | 0.9773 | 0.6341 | 0.2632 | 0.9457 |
| 50 | 50 | 116,540 | 0.6184 | 0.5689 | 0.9773 | 0.7192 | 0.2595 | 0.9197 |
| 40 | 60 | 97,115 | 0.6914 | 0.6653 | 0.9773 | 0.7917 | 0.2626 | 0.8853 |
| 30 | 70 | 83,240 | 0.7636 | 0.7562 | 0.9773 | 0.8527 | 0.2647 | 0.8335 |
| 20 | 80 | 72,835 | 0.8332 | 0.8403 | 0.9773 | 0.9036 | 0.2568 | 0.7390 |
| 10 | 90 | 64,740 | 0.9069 | 0.9237 | 0.9773 | 0.9497 | 0.2731 | 0.5724 |
| 0 | 100 | 58,270 | 0.9773 | 1.0000 | 0.9773 | 0.9885 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7311 | 0.0000 | 0.0000 | 0.0000 | 0.7311 | 1.0000 |
| 90 | 10 | 299,940 | 0.7574 | 0.2916 | 0.9981 | 0.4514 | 0.7306 | 0.9997 |
| 80 | 20 | 291,350 | 0.7833 | 0.4800 | 0.9980 | 0.6482 | 0.7297 | 0.9993 |
| 70 | 30 | 194,230 | 0.8112 | 0.6140 | 0.9980 | 0.7603 | 0.7311 | 0.9988 |
| 60 | 40 | 145,675 | 0.8381 | 0.7125 | 0.9980 | 0.8315 | 0.7316 | 0.9982 |
| 50 | 50 | 116,540 | 0.8632 | 0.7860 | 0.9980 | 0.8794 | 0.7283 | 0.9973 |
| 40 | 60 | 97,115 | 0.8912 | 0.8476 | 0.9980 | 0.9167 | 0.7309 | 0.9960 |
| 30 | 70 | 83,240 | 0.9176 | 0.8961 | 0.9980 | 0.9443 | 0.7300 | 0.9937 |
| 20 | 80 | 72,835 | 0.9446 | 0.9368 | 0.9980 | 0.9665 | 0.7309 | 0.9893 |
| 10 | 90 | 64,740 | 0.9711 | 0.9707 | 0.9980 | 0.9842 | 0.7285 | 0.9762 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7314 | 0.0000 | 0.0000 | 0.0000 | 0.7314 | 1.0000 |
| 90 | 10 | 299,940 | 0.7577 | 0.2919 | 0.9981 | 0.4517 | 0.7310 | 0.9997 |
| 80 | 20 | 291,350 | 0.7836 | 0.4803 | 0.9980 | 0.6485 | 0.7300 | 0.9993 |
| 70 | 30 | 194,230 | 0.8114 | 0.6143 | 0.9980 | 0.7605 | 0.7315 | 0.9988 |
| 60 | 40 | 145,675 | 0.8384 | 0.7128 | 0.9980 | 0.8317 | 0.7320 | 0.9982 |
| 50 | 50 | 116,540 | 0.8634 | 0.7864 | 0.9980 | 0.8796 | 0.7288 | 0.9973 |
| 40 | 60 | 97,115 | 0.8915 | 0.8480 | 0.9980 | 0.9169 | 0.7317 | 0.9960 |
| 30 | 70 | 83,240 | 0.9177 | 0.8962 | 0.9980 | 0.9444 | 0.7304 | 0.9937 |
| 20 | 80 | 72,835 | 0.9447 | 0.9370 | 0.9980 | 0.9665 | 0.7315 | 0.9893 |
| 10 | 90 | 64,740 | 0.9711 | 0.9707 | 0.9980 | 0.9842 | 0.7289 | 0.9762 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7314 | 0.0000 | 0.0000 | 0.0000 | 0.7314 | 1.0000 |
| 90 | 10 | 299,940 | 0.7577 | 0.2919 | 0.9981 | 0.4517 | 0.7310 | 0.9997 |
| 80 | 20 | 291,350 | 0.7836 | 0.4803 | 0.9980 | 0.6485 | 0.7300 | 0.9993 |
| 70 | 30 | 194,230 | 0.8114 | 0.6143 | 0.9980 | 0.7605 | 0.7315 | 0.9988 |
| 60 | 40 | 145,675 | 0.8384 | 0.7128 | 0.9980 | 0.8317 | 0.7320 | 0.9982 |
| 50 | 50 | 116,540 | 0.8634 | 0.7864 | 0.9980 | 0.8796 | 0.7288 | 0.9973 |
| 40 | 60 | 97,115 | 0.8915 | 0.8480 | 0.9980 | 0.9169 | 0.7317 | 0.9960 |
| 30 | 70 | 83,240 | 0.9177 | 0.8962 | 0.9980 | 0.9444 | 0.7304 | 0.9937 |
| 20 | 80 | 72,835 | 0.9447 | 0.9370 | 0.9980 | 0.9665 | 0.7315 | 0.9893 |
| 10 | 90 | 64,740 | 0.9711 | 0.9707 | 0.9980 | 0.9842 | 0.7289 | 0.9762 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |


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
0.15       0.3337   0.2271   0.2620   0.9911   0.9789   0.1285   <--
0.20       0.3337   0.2271   0.2620   0.9911   0.9789   0.1285  
0.25       0.3337   0.2271   0.2620   0.9911   0.9789   0.1285  
0.30       0.3337   0.2271   0.2620   0.9911   0.9789   0.1285  
0.35       0.3337   0.2271   0.2620   0.9911   0.9789   0.1285  
0.40       0.3337   0.2271   0.2620   0.9911   0.9789   0.1285  
0.45       0.3337   0.2271   0.2620   0.9911   0.9789   0.1285  
0.50       0.3337   0.2271   0.2620   0.9911   0.9789   0.1285  
0.55       0.3337   0.2271   0.2620   0.9911   0.9789   0.1285  
0.60       0.3337   0.2271   0.2620   0.9911   0.9789   0.1285  
0.65       0.3337   0.2271   0.2620   0.9911   0.9789   0.1285  
0.70       0.3337   0.2271   0.2620   0.9911   0.9789   0.1285  
0.75       0.3337   0.2271   0.2620   0.9911   0.9789   0.1285  
0.80       0.3337   0.2271   0.2620   0.9911   0.9789   0.1285  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3337, F1=0.2271, Normal Recall=0.2620, Normal Precision=0.9911, Attack Recall=0.9789, Attack Precision=0.1285

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
0.15       0.4050   0.3965   0.2619   0.9788   0.9773   0.2487   <--
0.20       0.4050   0.3965   0.2619   0.9788   0.9773   0.2487  
0.25       0.4050   0.3965   0.2619   0.9788   0.9773   0.2487  
0.30       0.4050   0.3965   0.2619   0.9788   0.9773   0.2487  
0.35       0.4050   0.3965   0.2619   0.9788   0.9773   0.2487  
0.40       0.4050   0.3965   0.2619   0.9788   0.9773   0.2487  
0.45       0.4050   0.3965   0.2619   0.9788   0.9773   0.2487  
0.50       0.4050   0.3965   0.2619   0.9788   0.9773   0.2487  
0.55       0.4050   0.3965   0.2619   0.9788   0.9773   0.2487  
0.60       0.4050   0.3965   0.2619   0.9788   0.9773   0.2487  
0.65       0.4050   0.3965   0.2619   0.9788   0.9773   0.2487  
0.70       0.4050   0.3965   0.2619   0.9788   0.9773   0.2487  
0.75       0.4050   0.3965   0.2619   0.9788   0.9773   0.2487  
0.80       0.4050   0.3965   0.2619   0.9788   0.9773   0.2487  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4050, F1=0.3965, Normal Recall=0.2619, Normal Precision=0.9788, Attack Recall=0.9773, Attack Precision=0.2487

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
0.15       0.4769   0.5285   0.2625   0.9643   0.9773   0.3622   <--
0.20       0.4769   0.5285   0.2625   0.9643   0.9773   0.3622  
0.25       0.4769   0.5285   0.2625   0.9643   0.9773   0.3622  
0.30       0.4769   0.5285   0.2625   0.9643   0.9773   0.3622  
0.35       0.4769   0.5285   0.2625   0.9643   0.9773   0.3622  
0.40       0.4769   0.5285   0.2625   0.9643   0.9773   0.3622  
0.45       0.4769   0.5285   0.2625   0.9643   0.9773   0.3622  
0.50       0.4769   0.5285   0.2625   0.9643   0.9773   0.3622  
0.55       0.4769   0.5285   0.2625   0.9643   0.9773   0.3622  
0.60       0.4769   0.5285   0.2625   0.9643   0.9773   0.3622  
0.65       0.4769   0.5285   0.2625   0.9643   0.9773   0.3622  
0.70       0.4769   0.5285   0.2625   0.9643   0.9773   0.3622  
0.75       0.4769   0.5285   0.2625   0.9643   0.9773   0.3622  
0.80       0.4769   0.5285   0.2625   0.9643   0.9773   0.3622  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4769, F1=0.5285, Normal Recall=0.2625, Normal Precision=0.9643, Attack Recall=0.9773, Attack Precision=0.3622

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
0.15       0.5473   0.6333   0.2607   0.9452   0.9773   0.4685   <--
0.20       0.5473   0.6333   0.2607   0.9452   0.9773   0.4685  
0.25       0.5473   0.6333   0.2607   0.9452   0.9773   0.4685  
0.30       0.5473   0.6333   0.2607   0.9452   0.9773   0.4685  
0.35       0.5473   0.6333   0.2607   0.9452   0.9773   0.4685  
0.40       0.5473   0.6333   0.2607   0.9452   0.9773   0.4685  
0.45       0.5473   0.6333   0.2607   0.9452   0.9773   0.4685  
0.50       0.5473   0.6333   0.2607   0.9452   0.9773   0.4685  
0.55       0.5473   0.6333   0.2607   0.9452   0.9773   0.4685  
0.60       0.5473   0.6333   0.2607   0.9452   0.9773   0.4685  
0.65       0.5473   0.6333   0.2607   0.9452   0.9773   0.4685  
0.70       0.5473   0.6333   0.2607   0.9452   0.9773   0.4685  
0.75       0.5473   0.6333   0.2607   0.9452   0.9773   0.4685  
0.80       0.5473   0.6333   0.2607   0.9452   0.9773   0.4685  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5473, F1=0.6333, Normal Recall=0.2607, Normal Precision=0.9452, Attack Recall=0.9773, Attack Precision=0.4685

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
0.15       0.6190   0.7195   0.2606   0.9200   0.9773   0.5693   <--
0.20       0.6190   0.7195   0.2606   0.9200   0.9773   0.5693  
0.25       0.6190   0.7195   0.2606   0.9200   0.9773   0.5693  
0.30       0.6190   0.7195   0.2606   0.9200   0.9773   0.5693  
0.35       0.6190   0.7195   0.2606   0.9200   0.9773   0.5693  
0.40       0.6190   0.7195   0.2606   0.9200   0.9773   0.5693  
0.45       0.6190   0.7195   0.2606   0.9200   0.9773   0.5693  
0.50       0.6190   0.7195   0.2606   0.9200   0.9773   0.5693  
0.55       0.6190   0.7195   0.2606   0.9200   0.9773   0.5693  
0.60       0.6190   0.7195   0.2606   0.9200   0.9773   0.5693  
0.65       0.6190   0.7195   0.2606   0.9200   0.9773   0.5693  
0.70       0.6190   0.7195   0.2606   0.9200   0.9773   0.5693  
0.75       0.6190   0.7195   0.2606   0.9200   0.9773   0.5693  
0.80       0.6190   0.7195   0.2606   0.9200   0.9773   0.5693  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6190, F1=0.7195, Normal Recall=0.2606, Normal Precision=0.9200, Attack Recall=0.9773, Attack Precision=0.5693

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
0.15       0.7574   0.4514   0.7306   0.9997   0.9981   0.2916   <--
0.20       0.7574   0.4514   0.7306   0.9997   0.9981   0.2916  
0.25       0.7574   0.4514   0.7306   0.9997   0.9981   0.2916  
0.30       0.7574   0.4514   0.7306   0.9997   0.9981   0.2916  
0.35       0.7574   0.4514   0.7306   0.9997   0.9981   0.2916  
0.40       0.7574   0.4514   0.7306   0.9997   0.9981   0.2916  
0.45       0.7574   0.4514   0.7306   0.9997   0.9981   0.2916  
0.50       0.7574   0.4514   0.7306   0.9997   0.9981   0.2916  
0.55       0.7574   0.4514   0.7306   0.9997   0.9981   0.2916  
0.60       0.7574   0.4514   0.7306   0.9997   0.9981   0.2916  
0.65       0.7574   0.4514   0.7306   0.9997   0.9981   0.2916  
0.70       0.7574   0.4514   0.7306   0.9997   0.9981   0.2916  
0.75       0.7574   0.4514   0.7306   0.9997   0.9981   0.2916  
0.80       0.7574   0.4514   0.7306   0.9997   0.9981   0.2916  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7574, F1=0.4514, Normal Recall=0.7306, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.2916

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
0.15       0.7848   0.6498   0.7315   0.9993   0.9980   0.4817   <--
0.20       0.7848   0.6498   0.7315   0.9993   0.9980   0.4817  
0.25       0.7848   0.6498   0.7315   0.9993   0.9980   0.4817  
0.30       0.7848   0.6498   0.7315   0.9993   0.9980   0.4817  
0.35       0.7848   0.6498   0.7315   0.9993   0.9980   0.4817  
0.40       0.7848   0.6498   0.7315   0.9993   0.9980   0.4817  
0.45       0.7848   0.6498   0.7315   0.9993   0.9980   0.4817  
0.50       0.7848   0.6498   0.7315   0.9993   0.9980   0.4817  
0.55       0.7848   0.6498   0.7315   0.9993   0.9980   0.4817  
0.60       0.7848   0.6498   0.7315   0.9993   0.9980   0.4817  
0.65       0.7848   0.6498   0.7315   0.9993   0.9980   0.4817  
0.70       0.7848   0.6498   0.7315   0.9993   0.9980   0.4817  
0.75       0.7848   0.6498   0.7315   0.9993   0.9980   0.4817  
0.80       0.7848   0.6498   0.7315   0.9993   0.9980   0.4817  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7848, F1=0.6498, Normal Recall=0.7315, Normal Precision=0.9993, Attack Recall=0.9980, Attack Precision=0.4817

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
0.15       0.8116   0.7606   0.7317   0.9988   0.9980   0.6145   <--
0.20       0.8116   0.7606   0.7317   0.9988   0.9980   0.6145  
0.25       0.8116   0.7606   0.7317   0.9988   0.9980   0.6145  
0.30       0.8116   0.7606   0.7317   0.9988   0.9980   0.6145  
0.35       0.8116   0.7606   0.7317   0.9988   0.9980   0.6145  
0.40       0.8116   0.7606   0.7317   0.9988   0.9980   0.6145  
0.45       0.8116   0.7606   0.7317   0.9988   0.9980   0.6145  
0.50       0.8116   0.7606   0.7317   0.9988   0.9980   0.6145  
0.55       0.8116   0.7606   0.7317   0.9988   0.9980   0.6145  
0.60       0.8116   0.7606   0.7317   0.9988   0.9980   0.6145  
0.65       0.8116   0.7606   0.7317   0.9988   0.9980   0.6145  
0.70       0.8116   0.7606   0.7317   0.9988   0.9980   0.6145  
0.75       0.8116   0.7606   0.7317   0.9988   0.9980   0.6145  
0.80       0.8116   0.7606   0.7317   0.9988   0.9980   0.6145  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8116, F1=0.7606, Normal Recall=0.7317, Normal Precision=0.9988, Attack Recall=0.9980, Attack Precision=0.6145

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
0.15       0.8378   0.8311   0.7310   0.9982   0.9980   0.7121   <--
0.20       0.8378   0.8311   0.7310   0.9982   0.9980   0.7121  
0.25       0.8378   0.8311   0.7310   0.9982   0.9980   0.7121  
0.30       0.8378   0.8311   0.7310   0.9982   0.9980   0.7121  
0.35       0.8378   0.8311   0.7310   0.9982   0.9980   0.7121  
0.40       0.8378   0.8311   0.7310   0.9982   0.9980   0.7121  
0.45       0.8378   0.8311   0.7310   0.9982   0.9980   0.7121  
0.50       0.8378   0.8311   0.7310   0.9982   0.9980   0.7121  
0.55       0.8378   0.8311   0.7310   0.9982   0.9980   0.7121  
0.60       0.8378   0.8311   0.7310   0.9982   0.9980   0.7121  
0.65       0.8378   0.8311   0.7310   0.9982   0.9980   0.7121  
0.70       0.8378   0.8311   0.7310   0.9982   0.9980   0.7121  
0.75       0.8378   0.8311   0.7310   0.9982   0.9980   0.7121  
0.80       0.8378   0.8311   0.7310   0.9982   0.9980   0.7121  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8378, F1=0.8311, Normal Recall=0.7310, Normal Precision=0.9982, Attack Recall=0.9980, Attack Precision=0.7121

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
0.15       0.8636   0.8798   0.7292   0.9973   0.9980   0.7866   <--
0.20       0.8636   0.8798   0.7292   0.9973   0.9980   0.7866  
0.25       0.8636   0.8798   0.7292   0.9973   0.9980   0.7866  
0.30       0.8636   0.8798   0.7292   0.9973   0.9980   0.7866  
0.35       0.8636   0.8798   0.7292   0.9973   0.9980   0.7866  
0.40       0.8636   0.8798   0.7292   0.9973   0.9980   0.7866  
0.45       0.8636   0.8798   0.7292   0.9973   0.9980   0.7866  
0.50       0.8636   0.8798   0.7292   0.9973   0.9980   0.7866  
0.55       0.8636   0.8798   0.7292   0.9973   0.9980   0.7866  
0.60       0.8636   0.8798   0.7292   0.9973   0.9980   0.7866  
0.65       0.8636   0.8798   0.7292   0.9973   0.9980   0.7866  
0.70       0.8636   0.8798   0.7292   0.9973   0.9980   0.7866  
0.75       0.8636   0.8798   0.7292   0.9973   0.9980   0.7866  
0.80       0.8636   0.8798   0.7292   0.9973   0.9980   0.7866  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8636, F1=0.8798, Normal Recall=0.7292, Normal Precision=0.9973, Attack Recall=0.9980, Attack Precision=0.7866

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
0.15       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919   <--
0.20       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.25       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.30       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.35       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.40       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.45       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.50       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.55       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.60       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.65       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.70       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.75       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.80       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7577, F1=0.4517, Normal Recall=0.7310, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.2919

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
0.15       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821   <--
0.20       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.25       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.30       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.35       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.40       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.45       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.50       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.55       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.60       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.65       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.70       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.75       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.80       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7852, F1=0.6501, Normal Recall=0.7319, Normal Precision=0.9993, Attack Recall=0.9980, Attack Precision=0.4821

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
0.15       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147   <--
0.20       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.25       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.30       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.35       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.40       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.45       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.50       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.55       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.60       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.65       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.70       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.75       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.80       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8118, F1=0.7608, Normal Recall=0.7319, Normal Precision=0.9988, Attack Recall=0.9980, Attack Precision=0.6147

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
0.15       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123   <--
0.20       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.25       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.30       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.35       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.40       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.45       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.50       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.55       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.60       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.65       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.70       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.75       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.80       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8379, F1=0.8313, Normal Recall=0.7312, Normal Precision=0.9982, Attack Recall=0.9980, Attack Precision=0.7123

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
0.15       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867   <--
0.20       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.25       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.30       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.35       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.40       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.45       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.50       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.55       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.60       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.65       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.70       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.75       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.80       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8637, F1=0.8799, Normal Recall=0.7295, Normal Precision=0.9973, Attack Recall=0.9980, Attack Precision=0.7867

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
0.15       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919   <--
0.20       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.25       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.30       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.35       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.40       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.45       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.50       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.55       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.60       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.65       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.70       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.75       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
0.80       0.7577   0.4517   0.7310   0.9997   0.9981   0.2919  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7577, F1=0.4517, Normal Recall=0.7310, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.2919

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
0.15       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821   <--
0.20       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.25       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.30       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.35       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.40       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.45       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.50       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.55       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.60       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.65       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.70       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.75       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
0.80       0.7852   0.6501   0.7319   0.9993   0.9980   0.4821  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7852, F1=0.6501, Normal Recall=0.7319, Normal Precision=0.9993, Attack Recall=0.9980, Attack Precision=0.4821

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
0.15       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147   <--
0.20       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.25       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.30       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.35       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.40       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.45       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.50       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.55       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.60       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.65       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.70       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.75       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
0.80       0.8118   0.7608   0.7319   0.9988   0.9980   0.6147  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8118, F1=0.7608, Normal Recall=0.7319, Normal Precision=0.9988, Attack Recall=0.9980, Attack Precision=0.6147

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
0.15       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123   <--
0.20       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.25       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.30       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.35       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.40       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.45       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.50       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.55       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.60       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.65       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.70       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.75       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
0.80       0.8379   0.8313   0.7312   0.9982   0.9980   0.7123  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8379, F1=0.8313, Normal Recall=0.7312, Normal Precision=0.9982, Attack Recall=0.9980, Attack Precision=0.7123

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
0.15       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867   <--
0.20       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.25       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.30       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.35       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.40       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.45       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.50       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.55       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.60       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.65       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.70       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.75       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
0.80       0.8637   0.8799   0.7295   0.9973   0.9980   0.7867  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8637, F1=0.8799, Normal Recall=0.7295, Normal Precision=0.9973, Attack Recall=0.9980, Attack Precision=0.7867

```

