# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e2_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-21 19:23:53 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8187 | 0.7879 | 0.7580 | 0.7294 | 0.6992 | 0.6692 | 0.6393 | 0.6108 | 0.5797 | 0.5502 | 0.5211 |
| QAT+Prune only | 0.8377 | 0.8383 | 0.8374 | 0.8370 | 0.8364 | 0.8350 | 0.8342 | 0.8334 | 0.8343 | 0.8323 | 0.8320 |
| QAT+PTQ | 0.8369 | 0.8375 | 0.8367 | 0.8363 | 0.8358 | 0.8346 | 0.8340 | 0.8332 | 0.8342 | 0.8322 | 0.8320 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8369 | 0.8375 | 0.8367 | 0.8363 | 0.8358 | 0.8346 | 0.8340 | 0.8332 | 0.8342 | 0.8322 | 0.8320 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3292 | 0.4628 | 0.5360 | 0.5809 | 0.6117 | 0.6342 | 0.6521 | 0.6648 | 0.6759 | 0.6852 |
| QAT+Prune only | 0.0000 | 0.5071 | 0.6718 | 0.7538 | 0.8027 | 0.8345 | 0.8576 | 0.8749 | 0.8893 | 0.8993 | 0.9083 |
| QAT+PTQ | 0.0000 | 0.5060 | 0.6708 | 0.7531 | 0.8021 | 0.8341 | 0.8574 | 0.8748 | 0.8892 | 0.8993 | 0.9083 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5060 | 0.6708 | 0.7531 | 0.8021 | 0.8341 | 0.8574 | 0.8748 | 0.8892 | 0.8993 | 0.9083 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8187 | 0.8176 | 0.8172 | 0.8186 | 0.8179 | 0.8173 | 0.8165 | 0.8200 | 0.8140 | 0.8117 | 0.0000 |
| QAT+Prune only | 0.8377 | 0.8390 | 0.8388 | 0.8391 | 0.8393 | 0.8380 | 0.8376 | 0.8369 | 0.8434 | 0.8353 | 0.0000 |
| QAT+PTQ | 0.8369 | 0.8380 | 0.8378 | 0.8381 | 0.8383 | 0.8371 | 0.8368 | 0.8360 | 0.8427 | 0.8340 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8369 | 0.8380 | 0.8378 | 0.8381 | 0.8383 | 0.8371 | 0.8368 | 0.8360 | 0.8427 | 0.8340 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8187 | 0.0000 | 0.0000 | 0.0000 | 0.8187 | 1.0000 |
| 90 | 10 | 299,940 | 0.7879 | 0.2407 | 0.5205 | 0.3292 | 0.8176 | 0.9388 |
| 80 | 20 | 291,350 | 0.7580 | 0.4162 | 0.5211 | 0.4628 | 0.8172 | 0.8722 |
| 70 | 30 | 194,230 | 0.7294 | 0.5519 | 0.5211 | 0.5360 | 0.8186 | 0.7995 |
| 60 | 40 | 145,675 | 0.6992 | 0.6561 | 0.5211 | 0.5809 | 0.8179 | 0.7193 |
| 50 | 50 | 116,540 | 0.6692 | 0.7404 | 0.5211 | 0.6117 | 0.8173 | 0.6305 |
| 40 | 60 | 97,115 | 0.6393 | 0.8099 | 0.5211 | 0.6342 | 0.8165 | 0.5320 |
| 30 | 70 | 83,240 | 0.6108 | 0.8710 | 0.5211 | 0.6521 | 0.8200 | 0.4232 |
| 20 | 80 | 72,835 | 0.5797 | 0.9181 | 0.5211 | 0.6648 | 0.8140 | 0.2982 |
| 10 | 90 | 64,740 | 0.5502 | 0.9614 | 0.5211 | 0.6759 | 0.8117 | 0.1585 |
| 0 | 100 | 58,270 | 0.5211 | 1.0000 | 0.5211 | 0.6852 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8377 | 0.0000 | 0.0000 | 0.0000 | 0.8377 | 1.0000 |
| 90 | 10 | 299,940 | 0.8383 | 0.3647 | 0.8320 | 0.5071 | 0.8390 | 0.9782 |
| 80 | 20 | 291,350 | 0.8374 | 0.5634 | 0.8320 | 0.6718 | 0.8388 | 0.9523 |
| 70 | 30 | 194,230 | 0.8370 | 0.6891 | 0.8320 | 0.7538 | 0.8391 | 0.9210 |
| 60 | 40 | 145,675 | 0.8364 | 0.7754 | 0.8320 | 0.8027 | 0.8393 | 0.8823 |
| 50 | 50 | 116,540 | 0.8350 | 0.8370 | 0.8320 | 0.8345 | 0.8380 | 0.8330 |
| 40 | 60 | 97,115 | 0.8342 | 0.8849 | 0.8320 | 0.8576 | 0.8376 | 0.7687 |
| 30 | 70 | 83,240 | 0.8334 | 0.9225 | 0.8320 | 0.8749 | 0.8369 | 0.6810 |
| 20 | 80 | 72,835 | 0.8343 | 0.9551 | 0.8320 | 0.8893 | 0.8434 | 0.5566 |
| 10 | 90 | 64,740 | 0.8323 | 0.9785 | 0.8320 | 0.8993 | 0.8353 | 0.3558 |
| 0 | 100 | 58,270 | 0.8320 | 1.0000 | 0.8320 | 0.9083 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8369 | 0.0000 | 0.0000 | 0.0000 | 0.8369 | 1.0000 |
| 90 | 10 | 299,940 | 0.8375 | 0.3635 | 0.8324 | 0.5060 | 0.8380 | 0.9783 |
| 80 | 20 | 291,350 | 0.8367 | 0.5619 | 0.8320 | 0.6708 | 0.8378 | 0.9523 |
| 70 | 30 | 194,230 | 0.8363 | 0.6878 | 0.8320 | 0.7531 | 0.8381 | 0.9209 |
| 60 | 40 | 145,675 | 0.8358 | 0.7743 | 0.8320 | 0.8021 | 0.8383 | 0.8822 |
| 50 | 50 | 116,540 | 0.8346 | 0.8363 | 0.8320 | 0.8341 | 0.8371 | 0.8329 |
| 40 | 60 | 97,115 | 0.8340 | 0.8844 | 0.8320 | 0.8574 | 0.8368 | 0.7686 |
| 30 | 70 | 83,240 | 0.8332 | 0.9221 | 0.8320 | 0.8748 | 0.8360 | 0.6808 |
| 20 | 80 | 72,835 | 0.8342 | 0.9549 | 0.8321 | 0.8892 | 0.8427 | 0.5564 |
| 10 | 90 | 64,740 | 0.8322 | 0.9783 | 0.8320 | 0.8993 | 0.8340 | 0.3555 |
| 0 | 100 | 58,270 | 0.8320 | 1.0000 | 0.8320 | 0.9083 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8369 | 0.0000 | 0.0000 | 0.0000 | 0.8369 | 1.0000 |
| 90 | 10 | 299,940 | 0.8375 | 0.3635 | 0.8324 | 0.5060 | 0.8380 | 0.9783 |
| 80 | 20 | 291,350 | 0.8367 | 0.5619 | 0.8320 | 0.6708 | 0.8378 | 0.9523 |
| 70 | 30 | 194,230 | 0.8363 | 0.6878 | 0.8320 | 0.7531 | 0.8381 | 0.9209 |
| 60 | 40 | 145,675 | 0.8358 | 0.7743 | 0.8320 | 0.8021 | 0.8383 | 0.8822 |
| 50 | 50 | 116,540 | 0.8346 | 0.8363 | 0.8320 | 0.8341 | 0.8371 | 0.8329 |
| 40 | 60 | 97,115 | 0.8340 | 0.8844 | 0.8320 | 0.8574 | 0.8368 | 0.7686 |
| 30 | 70 | 83,240 | 0.8332 | 0.9221 | 0.8320 | 0.8748 | 0.8360 | 0.6808 |
| 20 | 80 | 72,835 | 0.8342 | 0.9549 | 0.8321 | 0.8892 | 0.8427 | 0.5564 |
| 10 | 90 | 64,740 | 0.8322 | 0.9783 | 0.8320 | 0.8993 | 0.8340 | 0.3555 |
| 0 | 100 | 58,270 | 0.8320 | 1.0000 | 0.8320 | 0.9083 | 0.0000 | 0.0000 |


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
0.15       0.7882   0.3307   0.8176   0.9392   0.5232   0.2417   <--
0.20       0.7882   0.3307   0.8176   0.9392   0.5232   0.2417  
0.25       0.7882   0.3307   0.8176   0.9392   0.5232   0.2417  
0.30       0.7882   0.3307   0.8176   0.9392   0.5232   0.2417  
0.35       0.7882   0.3307   0.8176   0.9392   0.5232   0.2417  
0.40       0.7882   0.3307   0.8176   0.9392   0.5232   0.2417  
0.45       0.7882   0.3307   0.8176   0.9392   0.5232   0.2417  
0.50       0.7882   0.3307   0.8176   0.9392   0.5232   0.2417  
0.55       0.7882   0.3307   0.8176   0.9392   0.5232   0.2417  
0.60       0.7882   0.3307   0.8176   0.9392   0.5232   0.2417  
0.65       0.7882   0.3307   0.8176   0.9392   0.5232   0.2417  
0.70       0.7882   0.3307   0.8176   0.9392   0.5232   0.2417  
0.75       0.7882   0.3307   0.8176   0.9392   0.5232   0.2417  
0.80       0.7882   0.3307   0.8176   0.9392   0.5232   0.2417  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7882, F1=0.3307, Normal Recall=0.8176, Normal Precision=0.9392, Attack Recall=0.5232, Attack Precision=0.2417

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
0.15       0.7584   0.4632   0.8177   0.8723   0.5211   0.4168   <--
0.20       0.7584   0.4632   0.8177   0.8723   0.5211   0.4168  
0.25       0.7584   0.4632   0.8177   0.8723   0.5211   0.4168  
0.30       0.7584   0.4632   0.8177   0.8723   0.5211   0.4168  
0.35       0.7584   0.4632   0.8177   0.8723   0.5211   0.4168  
0.40       0.7584   0.4632   0.8177   0.8723   0.5211   0.4168  
0.45       0.7584   0.4632   0.8177   0.8723   0.5211   0.4168  
0.50       0.7584   0.4632   0.8177   0.8723   0.5211   0.4168  
0.55       0.7584   0.4632   0.8177   0.8723   0.5211   0.4168  
0.60       0.7584   0.4632   0.8177   0.8723   0.5211   0.4168  
0.65       0.7584   0.4632   0.8177   0.8723   0.5211   0.4168  
0.70       0.7584   0.4632   0.8177   0.8723   0.5211   0.4168  
0.75       0.7584   0.4632   0.8177   0.8723   0.5211   0.4168  
0.80       0.7584   0.4632   0.8177   0.8723   0.5211   0.4168  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7584, F1=0.4632, Normal Recall=0.8177, Normal Precision=0.8723, Attack Recall=0.5211, Attack Precision=0.4168

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
0.15       0.7298   0.5364   0.8192   0.7997   0.5211   0.5526   <--
0.20       0.7298   0.5364   0.8192   0.7997   0.5211   0.5526  
0.25       0.7298   0.5364   0.8192   0.7997   0.5211   0.5526  
0.30       0.7298   0.5364   0.8192   0.7997   0.5211   0.5526  
0.35       0.7298   0.5364   0.8192   0.7997   0.5211   0.5526  
0.40       0.7298   0.5364   0.8192   0.7997   0.5211   0.5526  
0.45       0.7298   0.5364   0.8192   0.7997   0.5211   0.5526  
0.50       0.7298   0.5364   0.8192   0.7997   0.5211   0.5526  
0.55       0.7298   0.5364   0.8192   0.7997   0.5211   0.5526  
0.60       0.7298   0.5364   0.8192   0.7997   0.5211   0.5526  
0.65       0.7298   0.5364   0.8192   0.7997   0.5211   0.5526  
0.70       0.7298   0.5364   0.8192   0.7997   0.5211   0.5526  
0.75       0.7298   0.5364   0.8192   0.7997   0.5211   0.5526  
0.80       0.7298   0.5364   0.8192   0.7997   0.5211   0.5526  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7298, F1=0.5364, Normal Recall=0.8192, Normal Precision=0.7997, Attack Recall=0.5211, Attack Precision=0.5526

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
0.15       0.6999   0.5814   0.8191   0.7195   0.5211   0.6576   <--
0.20       0.6999   0.5814   0.8191   0.7195   0.5211   0.6576  
0.25       0.6999   0.5814   0.8191   0.7195   0.5211   0.6576  
0.30       0.6999   0.5814   0.8191   0.7195   0.5211   0.6576  
0.35       0.6999   0.5814   0.8191   0.7195   0.5211   0.6576  
0.40       0.6999   0.5814   0.8191   0.7195   0.5211   0.6576  
0.45       0.6999   0.5814   0.8191   0.7195   0.5211   0.6576  
0.50       0.6999   0.5814   0.8191   0.7195   0.5211   0.6576  
0.55       0.6999   0.5814   0.8191   0.7195   0.5211   0.6576  
0.60       0.6999   0.5814   0.8191   0.7195   0.5211   0.6576  
0.65       0.6999   0.5814   0.8191   0.7195   0.5211   0.6576  
0.70       0.6999   0.5814   0.8191   0.7195   0.5211   0.6576  
0.75       0.6999   0.5814   0.8191   0.7195   0.5211   0.6576  
0.80       0.6999   0.5814   0.8191   0.7195   0.5211   0.6576  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6999, F1=0.5814, Normal Recall=0.8191, Normal Precision=0.7195, Attack Recall=0.5211, Attack Precision=0.6576

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
0.15       0.6699   0.6122   0.8186   0.6309   0.5211   0.7418   <--
0.20       0.6699   0.6122   0.8186   0.6309   0.5211   0.7418  
0.25       0.6699   0.6122   0.8186   0.6309   0.5211   0.7418  
0.30       0.6699   0.6122   0.8186   0.6309   0.5211   0.7418  
0.35       0.6699   0.6122   0.8186   0.6309   0.5211   0.7418  
0.40       0.6699   0.6122   0.8186   0.6309   0.5211   0.7418  
0.45       0.6699   0.6122   0.8186   0.6309   0.5211   0.7418  
0.50       0.6699   0.6122   0.8186   0.6309   0.5211   0.7418  
0.55       0.6699   0.6122   0.8186   0.6309   0.5211   0.7418  
0.60       0.6699   0.6122   0.8186   0.6309   0.5211   0.7418  
0.65       0.6699   0.6122   0.8186   0.6309   0.5211   0.7418  
0.70       0.6699   0.6122   0.8186   0.6309   0.5211   0.7418  
0.75       0.6699   0.6122   0.8186   0.6309   0.5211   0.7418  
0.80       0.6699   0.6122   0.8186   0.6309   0.5211   0.7418  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6699, F1=0.6122, Normal Recall=0.8186, Normal Precision=0.6309, Attack Recall=0.5211, Attack Precision=0.7418

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
0.15       0.8385   0.5079   0.8390   0.9785   0.8337   0.3652   <--
0.20       0.8385   0.5079   0.8390   0.9785   0.8337   0.3652  
0.25       0.8385   0.5079   0.8390   0.9785   0.8337   0.3652  
0.30       0.8385   0.5079   0.8390   0.9785   0.8337   0.3652  
0.35       0.8385   0.5079   0.8390   0.9785   0.8337   0.3652  
0.40       0.8385   0.5079   0.8390   0.9785   0.8337   0.3652  
0.45       0.8385   0.5079   0.8390   0.9785   0.8337   0.3652  
0.50       0.8385   0.5079   0.8390   0.9785   0.8337   0.3652  
0.55       0.8385   0.5079   0.8390   0.9785   0.8337   0.3652  
0.60       0.8385   0.5079   0.8390   0.9785   0.8337   0.3652  
0.65       0.8385   0.5079   0.8390   0.9785   0.8337   0.3652  
0.70       0.8385   0.5079   0.8390   0.9785   0.8337   0.3652  
0.75       0.8385   0.5079   0.8390   0.9785   0.8337   0.3652  
0.80       0.8385   0.5079   0.8390   0.9785   0.8337   0.3652  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8385, F1=0.5079, Normal Recall=0.8390, Normal Precision=0.9785, Attack Recall=0.8337, Attack Precision=0.3652

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
0.15       0.8377   0.6722   0.8392   0.9523   0.8320   0.5639   <--
0.20       0.8377   0.6722   0.8392   0.9523   0.8320   0.5639  
0.25       0.8377   0.6722   0.8392   0.9523   0.8320   0.5639  
0.30       0.8377   0.6722   0.8392   0.9523   0.8320   0.5639  
0.35       0.8377   0.6722   0.8392   0.9523   0.8320   0.5639  
0.40       0.8377   0.6722   0.8392   0.9523   0.8320   0.5639  
0.45       0.8377   0.6722   0.8392   0.9523   0.8320   0.5639  
0.50       0.8377   0.6722   0.8392   0.9523   0.8320   0.5639  
0.55       0.8377   0.6722   0.8392   0.9523   0.8320   0.5639  
0.60       0.8377   0.6722   0.8392   0.9523   0.8320   0.5639  
0.65       0.8377   0.6722   0.8392   0.9523   0.8320   0.5639  
0.70       0.8377   0.6722   0.8392   0.9523   0.8320   0.5639  
0.75       0.8377   0.6722   0.8392   0.9523   0.8320   0.5639  
0.80       0.8377   0.6722   0.8392   0.9523   0.8320   0.5639  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8377, F1=0.6722, Normal Recall=0.8392, Normal Precision=0.9523, Attack Recall=0.8320, Attack Precision=0.5639

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
0.15       0.8364   0.7532   0.8383   0.9209   0.8320   0.6880   <--
0.20       0.8364   0.7532   0.8383   0.9209   0.8320   0.6880  
0.25       0.8364   0.7532   0.8383   0.9209   0.8320   0.6880  
0.30       0.8364   0.7532   0.8383   0.9209   0.8320   0.6880  
0.35       0.8364   0.7532   0.8383   0.9209   0.8320   0.6880  
0.40       0.8364   0.7532   0.8383   0.9209   0.8320   0.6880  
0.45       0.8364   0.7532   0.8383   0.9209   0.8320   0.6880  
0.50       0.8364   0.7532   0.8383   0.9209   0.8320   0.6880  
0.55       0.8364   0.7532   0.8383   0.9209   0.8320   0.6880  
0.60       0.8364   0.7532   0.8383   0.9209   0.8320   0.6880  
0.65       0.8364   0.7532   0.8383   0.9209   0.8320   0.6880  
0.70       0.8364   0.7532   0.8383   0.9209   0.8320   0.6880  
0.75       0.8364   0.7532   0.8383   0.9209   0.8320   0.6880  
0.80       0.8364   0.7532   0.8383   0.9209   0.8320   0.6880  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8364, F1=0.7532, Normal Recall=0.8383, Normal Precision=0.9209, Attack Recall=0.8320, Attack Precision=0.6880

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
0.15       0.8352   0.8015   0.8373   0.8820   0.8320   0.7732   <--
0.20       0.8352   0.8015   0.8373   0.8820   0.8320   0.7732  
0.25       0.8352   0.8015   0.8373   0.8820   0.8320   0.7732  
0.30       0.8352   0.8015   0.8373   0.8820   0.8320   0.7732  
0.35       0.8352   0.8015   0.8373   0.8820   0.8320   0.7732  
0.40       0.8352   0.8015   0.8373   0.8820   0.8320   0.7732  
0.45       0.8352   0.8015   0.8373   0.8820   0.8320   0.7732  
0.50       0.8352   0.8015   0.8373   0.8820   0.8320   0.7732  
0.55       0.8352   0.8015   0.8373   0.8820   0.8320   0.7732  
0.60       0.8352   0.8015   0.8373   0.8820   0.8320   0.7732  
0.65       0.8352   0.8015   0.8373   0.8820   0.8320   0.7732  
0.70       0.8352   0.8015   0.8373   0.8820   0.8320   0.7732  
0.75       0.8352   0.8015   0.8373   0.8820   0.8320   0.7732  
0.80       0.8352   0.8015   0.8373   0.8820   0.8320   0.7732  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8352, F1=0.8015, Normal Recall=0.8373, Normal Precision=0.8820, Attack Recall=0.8320, Attack Precision=0.7732

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
0.15       0.8340   0.8336   0.8360   0.8327   0.8320   0.8353   <--
0.20       0.8340   0.8336   0.8360   0.8327   0.8320   0.8353  
0.25       0.8340   0.8336   0.8360   0.8327   0.8320   0.8353  
0.30       0.8340   0.8336   0.8360   0.8327   0.8320   0.8353  
0.35       0.8340   0.8336   0.8360   0.8327   0.8320   0.8353  
0.40       0.8340   0.8336   0.8360   0.8327   0.8320   0.8353  
0.45       0.8340   0.8336   0.8360   0.8327   0.8320   0.8353  
0.50       0.8340   0.8336   0.8360   0.8327   0.8320   0.8353  
0.55       0.8340   0.8336   0.8360   0.8327   0.8320   0.8353  
0.60       0.8340   0.8336   0.8360   0.8327   0.8320   0.8353  
0.65       0.8340   0.8336   0.8360   0.8327   0.8320   0.8353  
0.70       0.8340   0.8336   0.8360   0.8327   0.8320   0.8353  
0.75       0.8340   0.8336   0.8360   0.8327   0.8320   0.8353  
0.80       0.8340   0.8336   0.8360   0.8327   0.8320   0.8353  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8340, F1=0.8336, Normal Recall=0.8360, Normal Precision=0.8327, Attack Recall=0.8320, Attack Precision=0.8353

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
0.15       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639   <--
0.20       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.25       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.30       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.35       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.40       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.45       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.50       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.55       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.60       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.65       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.70       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.75       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.80       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8376, F1=0.5067, Normal Recall=0.8381, Normal Precision=0.9784, Attack Recall=0.8339, Attack Precision=0.3639

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
0.15       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625   <--
0.20       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.25       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.30       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.35       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.40       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.45       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.50       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.55       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.60       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.65       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.70       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.75       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.80       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8370, F1=0.6712, Normal Recall=0.8382, Normal Precision=0.9523, Attack Recall=0.8320, Attack Precision=0.5625

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
0.15       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868   <--
0.20       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.25       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.30       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.35       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.40       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.45       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.50       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.55       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.60       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.65       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.70       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.75       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.80       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8358, F1=0.7524, Normal Recall=0.8374, Normal Precision=0.9208, Attack Recall=0.8320, Attack Precision=0.6868

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
0.15       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722   <--
0.20       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.25       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.30       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.35       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.40       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.45       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.50       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.55       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.60       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.65       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.70       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.75       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.80       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8346, F1=0.8010, Normal Recall=0.8364, Normal Precision=0.8819, Attack Recall=0.8320, Attack Precision=0.7722

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
0.15       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347   <--
0.20       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.25       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.30       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.35       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.40       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.45       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.50       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.55       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.60       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.65       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.70       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.75       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.80       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8336, F1=0.8334, Normal Recall=0.8352, Normal Precision=0.8326, Attack Recall=0.8320, Attack Precision=0.8347

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
0.15       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639   <--
0.20       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.25       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.30       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.35       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.40       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.45       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.50       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.55       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.60       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.65       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.70       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.75       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
0.80       0.8376   0.5067   0.8381   0.9784   0.8339   0.3639  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8376, F1=0.5067, Normal Recall=0.8381, Normal Precision=0.9784, Attack Recall=0.8339, Attack Precision=0.3639

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
0.15       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625   <--
0.20       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.25       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.30       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.35       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.40       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.45       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.50       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.55       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.60       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.65       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.70       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.75       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
0.80       0.8370   0.6712   0.8382   0.9523   0.8320   0.5625  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8370, F1=0.6712, Normal Recall=0.8382, Normal Precision=0.9523, Attack Recall=0.8320, Attack Precision=0.5625

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
0.15       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868   <--
0.20       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.25       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.30       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.35       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.40       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.45       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.50       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.55       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.60       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.65       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.70       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.75       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
0.80       0.8358   0.7524   0.8374   0.9208   0.8320   0.6868  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8358, F1=0.7524, Normal Recall=0.8374, Normal Precision=0.9208, Attack Recall=0.8320, Attack Precision=0.6868

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
0.15       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722   <--
0.20       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.25       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.30       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.35       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.40       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.45       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.50       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.55       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.60       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.65       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.70       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.75       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
0.80       0.8346   0.8010   0.8364   0.8819   0.8320   0.7722  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8346, F1=0.8010, Normal Recall=0.8364, Normal Precision=0.8819, Attack Recall=0.8320, Attack Precision=0.7722

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
0.15       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347   <--
0.20       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.25       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.30       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.35       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.40       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.45       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.50       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.55       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.60       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.65       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.70       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.75       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
0.80       0.8336   0.8334   0.8352   0.8326   0.8320   0.8347  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8336, F1=0.8334, Normal Recall=0.8352, Normal Precision=0.8326, Attack Recall=0.8320, Attack Precision=0.8347

```

