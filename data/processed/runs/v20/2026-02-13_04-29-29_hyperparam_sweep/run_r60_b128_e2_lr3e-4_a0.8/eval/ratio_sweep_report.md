# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e2_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-14 20:02:49 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4043 | 0.4532 | 0.5026 | 0.5517 | 0.6014 | 0.6493 | 0.7002 | 0.7485 | 0.7971 | 0.8472 | 0.8960 |
| QAT+Prune only | 0.9188 | 0.9237 | 0.9284 | 0.9340 | 0.9389 | 0.9427 | 0.9489 | 0.9538 | 0.9582 | 0.9633 | 0.9683 |
| QAT+PTQ | 0.9186 | 0.9235 | 0.9280 | 0.9335 | 0.9383 | 0.9421 | 0.9481 | 0.9531 | 0.9575 | 0.9624 | 0.9673 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9186 | 0.9235 | 0.9280 | 0.9335 | 0.9383 | 0.9421 | 0.9481 | 0.9531 | 0.9575 | 0.9624 | 0.9673 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2464 | 0.4188 | 0.5453 | 0.6427 | 0.7187 | 0.7820 | 0.8330 | 0.8760 | 0.9135 | 0.9451 |
| QAT+Prune only | 0.0000 | 0.7176 | 0.8439 | 0.8980 | 0.9269 | 0.9441 | 0.9579 | 0.9670 | 0.9737 | 0.9794 | 0.9839 |
| QAT+PTQ | 0.0000 | 0.7166 | 0.8431 | 0.8972 | 0.9262 | 0.9436 | 0.9572 | 0.9665 | 0.9732 | 0.9789 | 0.9834 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.7166 | 0.8431 | 0.8972 | 0.9262 | 0.9436 | 0.9572 | 0.9665 | 0.9732 | 0.9789 | 0.9834 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4043 | 0.4042 | 0.4043 | 0.4041 | 0.4051 | 0.4026 | 0.4065 | 0.4045 | 0.4014 | 0.4086 | 0.0000 |
| QAT+Prune only | 0.9188 | 0.9188 | 0.9184 | 0.9193 | 0.9193 | 0.9170 | 0.9198 | 0.9200 | 0.9180 | 0.9183 | 0.0000 |
| QAT+PTQ | 0.9186 | 0.9185 | 0.9182 | 0.9191 | 0.9191 | 0.9170 | 0.9195 | 0.9200 | 0.9182 | 0.9184 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9186 | 0.9185 | 0.9182 | 0.9191 | 0.9191 | 0.9170 | 0.9195 | 0.9200 | 0.9182 | 0.9184 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4043 | 0.0000 | 0.0000 | 0.0000 | 0.4043 | 1.0000 |
| 90 | 10 | 299,940 | 0.4532 | 0.1429 | 0.8940 | 0.2464 | 0.4042 | 0.9717 |
| 80 | 20 | 291,350 | 0.5026 | 0.2733 | 0.8960 | 0.4188 | 0.4043 | 0.9396 |
| 70 | 30 | 194,230 | 0.5517 | 0.3919 | 0.8960 | 0.5453 | 0.4041 | 0.9006 |
| 60 | 40 | 145,675 | 0.6014 | 0.5010 | 0.8960 | 0.6427 | 0.4051 | 0.8538 |
| 50 | 50 | 116,540 | 0.6493 | 0.6000 | 0.8960 | 0.7187 | 0.4026 | 0.7947 |
| 40 | 60 | 97,115 | 0.7002 | 0.6937 | 0.8960 | 0.7820 | 0.4065 | 0.7226 |
| 30 | 70 | 83,240 | 0.7485 | 0.7783 | 0.8960 | 0.8330 | 0.4045 | 0.6250 |
| 20 | 80 | 72,835 | 0.7971 | 0.8569 | 0.8960 | 0.8760 | 0.4014 | 0.4910 |
| 10 | 90 | 64,740 | 0.8472 | 0.9317 | 0.8960 | 0.9135 | 0.4086 | 0.3038 |
| 0 | 100 | 58,270 | 0.8960 | 1.0000 | 0.8960 | 0.9451 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9188 | 0.0000 | 0.0000 | 0.0000 | 0.9188 | 1.0000 |
| 90 | 10 | 299,940 | 0.9237 | 0.5699 | 0.9687 | 0.7176 | 0.9188 | 0.9962 |
| 80 | 20 | 291,350 | 0.9284 | 0.7478 | 0.9683 | 0.8439 | 0.9184 | 0.9914 |
| 70 | 30 | 194,230 | 0.9340 | 0.8372 | 0.9683 | 0.8980 | 0.9193 | 0.9854 |
| 60 | 40 | 145,675 | 0.9389 | 0.8889 | 0.9683 | 0.9269 | 0.9193 | 0.9775 |
| 50 | 50 | 116,540 | 0.9427 | 0.9211 | 0.9683 | 0.9441 | 0.9170 | 0.9666 |
| 40 | 60 | 97,115 | 0.9489 | 0.9476 | 0.9683 | 0.9579 | 0.9198 | 0.9508 |
| 30 | 70 | 83,240 | 0.9538 | 0.9658 | 0.9683 | 0.9670 | 0.9200 | 0.9256 |
| 20 | 80 | 72,835 | 0.9582 | 0.9793 | 0.9683 | 0.9737 | 0.9180 | 0.8786 |
| 10 | 90 | 64,740 | 0.9633 | 0.9907 | 0.9683 | 0.9794 | 0.9183 | 0.7630 |
| 0 | 100 | 58,270 | 0.9683 | 1.0000 | 0.9683 | 0.9839 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9186 | 0.0000 | 0.0000 | 0.0000 | 0.9186 | 1.0000 |
| 90 | 10 | 299,940 | 0.9235 | 0.5690 | 0.9677 | 0.7166 | 0.9185 | 0.9961 |
| 80 | 20 | 291,350 | 0.9280 | 0.7472 | 0.9673 | 0.8431 | 0.9182 | 0.9912 |
| 70 | 30 | 194,230 | 0.9335 | 0.8366 | 0.9673 | 0.8972 | 0.9191 | 0.9850 |
| 60 | 40 | 145,675 | 0.9383 | 0.8885 | 0.9673 | 0.9262 | 0.9191 | 0.9768 |
| 50 | 50 | 116,540 | 0.9421 | 0.9210 | 0.9673 | 0.9436 | 0.9170 | 0.9655 |
| 40 | 60 | 97,115 | 0.9481 | 0.9474 | 0.9673 | 0.9572 | 0.9195 | 0.9493 |
| 30 | 70 | 83,240 | 0.9531 | 0.9658 | 0.9673 | 0.9665 | 0.9200 | 0.9233 |
| 20 | 80 | 72,835 | 0.9575 | 0.9793 | 0.9673 | 0.9732 | 0.9182 | 0.8752 |
| 10 | 90 | 64,740 | 0.9624 | 0.9907 | 0.9673 | 0.9789 | 0.9184 | 0.7572 |
| 0 | 100 | 58,270 | 0.9673 | 1.0000 | 0.9673 | 0.9834 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9186 | 0.0000 | 0.0000 | 0.0000 | 0.9186 | 1.0000 |
| 90 | 10 | 299,940 | 0.9235 | 0.5690 | 0.9677 | 0.7166 | 0.9185 | 0.9961 |
| 80 | 20 | 291,350 | 0.9280 | 0.7472 | 0.9673 | 0.8431 | 0.9182 | 0.9912 |
| 70 | 30 | 194,230 | 0.9335 | 0.8366 | 0.9673 | 0.8972 | 0.9191 | 0.9850 |
| 60 | 40 | 145,675 | 0.9383 | 0.8885 | 0.9673 | 0.9262 | 0.9191 | 0.9768 |
| 50 | 50 | 116,540 | 0.9421 | 0.9210 | 0.9673 | 0.9436 | 0.9170 | 0.9655 |
| 40 | 60 | 97,115 | 0.9481 | 0.9474 | 0.9673 | 0.9572 | 0.9195 | 0.9493 |
| 30 | 70 | 83,240 | 0.9531 | 0.9658 | 0.9673 | 0.9665 | 0.9200 | 0.9233 |
| 20 | 80 | 72,835 | 0.9575 | 0.9793 | 0.9673 | 0.9732 | 0.9182 | 0.8752 |
| 10 | 90 | 64,740 | 0.9624 | 0.9907 | 0.9673 | 0.9789 | 0.9184 | 0.7572 |
| 0 | 100 | 58,270 | 0.9673 | 1.0000 | 0.9673 | 0.9834 | 0.0000 | 0.0000 |


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
0.15       0.4535   0.2471   0.4043   0.9724   0.8967   0.1433   <--
0.20       0.4535   0.2471   0.4043   0.9724   0.8967   0.1433  
0.25       0.4535   0.2471   0.4043   0.9724   0.8967   0.1433  
0.30       0.4535   0.2471   0.4043   0.9724   0.8967   0.1433  
0.35       0.4535   0.2471   0.4043   0.9724   0.8967   0.1433  
0.40       0.4535   0.2471   0.4043   0.9724   0.8967   0.1433  
0.45       0.4535   0.2471   0.4043   0.9724   0.8967   0.1433  
0.50       0.4535   0.2471   0.4043   0.9724   0.8967   0.1433  
0.55       0.4535   0.2471   0.4043   0.9724   0.8967   0.1433  
0.60       0.4535   0.2471   0.4043   0.9724   0.8967   0.1433  
0.65       0.4535   0.2471   0.4043   0.9724   0.8967   0.1433  
0.70       0.4535   0.2471   0.4043   0.9724   0.8967   0.1433  
0.75       0.4535   0.2471   0.4043   0.9724   0.8967   0.1433  
0.80       0.4535   0.2471   0.4043   0.9724   0.8967   0.1433  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4535, F1=0.2471, Normal Recall=0.4043, Normal Precision=0.9724, Attack Recall=0.8967, Attack Precision=0.1433

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
0.15       0.5027   0.4188   0.4044   0.9396   0.8960   0.2733   <--
0.20       0.5027   0.4188   0.4044   0.9396   0.8960   0.2733  
0.25       0.5027   0.4188   0.4044   0.9396   0.8960   0.2733  
0.30       0.5027   0.4188   0.4044   0.9396   0.8960   0.2733  
0.35       0.5027   0.4188   0.4044   0.9396   0.8960   0.2733  
0.40       0.5027   0.4188   0.4044   0.9396   0.8960   0.2733  
0.45       0.5027   0.4188   0.4044   0.9396   0.8960   0.2733  
0.50       0.5027   0.4188   0.4044   0.9396   0.8960   0.2733  
0.55       0.5027   0.4188   0.4044   0.9396   0.8960   0.2733  
0.60       0.5027   0.4188   0.4044   0.9396   0.8960   0.2733  
0.65       0.5027   0.4188   0.4044   0.9396   0.8960   0.2733  
0.70       0.5027   0.4188   0.4044   0.9396   0.8960   0.2733  
0.75       0.5027   0.4188   0.4044   0.9396   0.8960   0.2733  
0.80       0.5027   0.4188   0.4044   0.9396   0.8960   0.2733  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5027, F1=0.4188, Normal Recall=0.4044, Normal Precision=0.9396, Attack Recall=0.8960, Attack Precision=0.2733

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
0.15       0.5514   0.5451   0.4037   0.9006   0.8960   0.3917   <--
0.20       0.5514   0.5451   0.4037   0.9006   0.8960   0.3917  
0.25       0.5514   0.5451   0.4037   0.9006   0.8960   0.3917  
0.30       0.5514   0.5451   0.4037   0.9006   0.8960   0.3917  
0.35       0.5514   0.5451   0.4037   0.9006   0.8960   0.3917  
0.40       0.5514   0.5451   0.4037   0.9006   0.8960   0.3917  
0.45       0.5514   0.5451   0.4037   0.9006   0.8960   0.3917  
0.50       0.5514   0.5451   0.4037   0.9006   0.8960   0.3917  
0.55       0.5514   0.5451   0.4037   0.9006   0.8960   0.3917  
0.60       0.5514   0.5451   0.4037   0.9006   0.8960   0.3917  
0.65       0.5514   0.5451   0.4037   0.9006   0.8960   0.3917  
0.70       0.5514   0.5451   0.4037   0.9006   0.8960   0.3917  
0.75       0.5514   0.5451   0.4037   0.9006   0.8960   0.3917  
0.80       0.5514   0.5451   0.4037   0.9006   0.8960   0.3917  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5514, F1=0.5451, Normal Recall=0.4037, Normal Precision=0.9006, Attack Recall=0.8960, Attack Precision=0.3917

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
0.15       0.6014   0.6426   0.4050   0.8538   0.8960   0.5010   <--
0.20       0.6014   0.6426   0.4050   0.8538   0.8960   0.5010  
0.25       0.6014   0.6426   0.4050   0.8538   0.8960   0.5010  
0.30       0.6014   0.6426   0.4050   0.8538   0.8960   0.5010  
0.35       0.6014   0.6426   0.4050   0.8538   0.8960   0.5010  
0.40       0.6014   0.6426   0.4050   0.8538   0.8960   0.5010  
0.45       0.6014   0.6426   0.4050   0.8538   0.8960   0.5010  
0.50       0.6014   0.6426   0.4050   0.8538   0.8960   0.5010  
0.55       0.6014   0.6426   0.4050   0.8538   0.8960   0.5010  
0.60       0.6014   0.6426   0.4050   0.8538   0.8960   0.5010  
0.65       0.6014   0.6426   0.4050   0.8538   0.8960   0.5010  
0.70       0.6014   0.6426   0.4050   0.8538   0.8960   0.5010  
0.75       0.6014   0.6426   0.4050   0.8538   0.8960   0.5010  
0.80       0.6014   0.6426   0.4050   0.8538   0.8960   0.5010  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6014, F1=0.6426, Normal Recall=0.4050, Normal Precision=0.8538, Attack Recall=0.8960, Attack Precision=0.5010

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
0.15       0.6507   0.7195   0.4054   0.7958   0.8960   0.6011   <--
0.20       0.6507   0.7195   0.4054   0.7958   0.8960   0.6011  
0.25       0.6507   0.7195   0.4054   0.7958   0.8960   0.6011  
0.30       0.6507   0.7195   0.4054   0.7958   0.8960   0.6011  
0.35       0.6507   0.7195   0.4054   0.7958   0.8960   0.6011  
0.40       0.6507   0.7195   0.4054   0.7958   0.8960   0.6011  
0.45       0.6507   0.7195   0.4054   0.7958   0.8960   0.6011  
0.50       0.6507   0.7195   0.4054   0.7958   0.8960   0.6011  
0.55       0.6507   0.7195   0.4054   0.7958   0.8960   0.6011  
0.60       0.6507   0.7195   0.4054   0.7958   0.8960   0.6011  
0.65       0.6507   0.7195   0.4054   0.7958   0.8960   0.6011  
0.70       0.6507   0.7195   0.4054   0.7958   0.8960   0.6011  
0.75       0.6507   0.7195   0.4054   0.7958   0.8960   0.6011  
0.80       0.6507   0.7195   0.4054   0.7958   0.8960   0.6011  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6507, F1=0.7195, Normal Recall=0.4054, Normal Precision=0.7958, Attack Recall=0.8960, Attack Precision=0.6011

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
0.15       0.9238   0.7178   0.9188   0.9963   0.9692   0.5700   <--
0.20       0.9238   0.7178   0.9188   0.9963   0.9692   0.5700  
0.25       0.9238   0.7178   0.9188   0.9963   0.9692   0.5700  
0.30       0.9238   0.7178   0.9188   0.9963   0.9692   0.5700  
0.35       0.9238   0.7178   0.9188   0.9963   0.9692   0.5700  
0.40       0.9238   0.7178   0.9188   0.9963   0.9692   0.5700  
0.45       0.9238   0.7178   0.9188   0.9963   0.9692   0.5700  
0.50       0.9238   0.7178   0.9188   0.9963   0.9692   0.5700  
0.55       0.9238   0.7178   0.9188   0.9963   0.9692   0.5700  
0.60       0.9238   0.7178   0.9188   0.9963   0.9692   0.5700  
0.65       0.9238   0.7178   0.9188   0.9963   0.9692   0.5700  
0.70       0.9238   0.7178   0.9188   0.9963   0.9692   0.5700  
0.75       0.9238   0.7178   0.9188   0.9963   0.9692   0.5700  
0.80       0.9238   0.7178   0.9188   0.9963   0.9692   0.5700  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9238, F1=0.7178, Normal Recall=0.9188, Normal Precision=0.9963, Attack Recall=0.9692, Attack Precision=0.5700

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
0.15       0.9289   0.8449   0.9191   0.9914   0.9683   0.7494   <--
0.20       0.9289   0.8449   0.9191   0.9914   0.9683   0.7494  
0.25       0.9289   0.8449   0.9191   0.9914   0.9683   0.7494  
0.30       0.9289   0.8449   0.9191   0.9914   0.9683   0.7494  
0.35       0.9289   0.8449   0.9191   0.9914   0.9683   0.7494  
0.40       0.9289   0.8449   0.9191   0.9914   0.9683   0.7494  
0.45       0.9289   0.8449   0.9191   0.9914   0.9683   0.7494  
0.50       0.9289   0.8449   0.9191   0.9914   0.9683   0.7494  
0.55       0.9289   0.8449   0.9191   0.9914   0.9683   0.7494  
0.60       0.9289   0.8449   0.9191   0.9914   0.9683   0.7494  
0.65       0.9289   0.8449   0.9191   0.9914   0.9683   0.7494  
0.70       0.9289   0.8449   0.9191   0.9914   0.9683   0.7494  
0.75       0.9289   0.8449   0.9191   0.9914   0.9683   0.7494  
0.80       0.9289   0.8449   0.9191   0.9914   0.9683   0.7494  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9289, F1=0.8449, Normal Recall=0.9191, Normal Precision=0.9914, Attack Recall=0.9683, Attack Precision=0.7494

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
0.15       0.9340   0.8980   0.9193   0.9854   0.9683   0.8372   <--
0.20       0.9340   0.8980   0.9193   0.9854   0.9683   0.8372  
0.25       0.9340   0.8980   0.9193   0.9854   0.9683   0.8372  
0.30       0.9340   0.8980   0.9193   0.9854   0.9683   0.8372  
0.35       0.9340   0.8980   0.9193   0.9854   0.9683   0.8372  
0.40       0.9340   0.8980   0.9193   0.9854   0.9683   0.8372  
0.45       0.9340   0.8980   0.9193   0.9854   0.9683   0.8372  
0.50       0.9340   0.8980   0.9193   0.9854   0.9683   0.8372  
0.55       0.9340   0.8980   0.9193   0.9854   0.9683   0.8372  
0.60       0.9340   0.8980   0.9193   0.9854   0.9683   0.8372  
0.65       0.9340   0.8980   0.9193   0.9854   0.9683   0.8372  
0.70       0.9340   0.8980   0.9193   0.9854   0.9683   0.8372  
0.75       0.9340   0.8980   0.9193   0.9854   0.9683   0.8372  
0.80       0.9340   0.8980   0.9193   0.9854   0.9683   0.8372  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9340, F1=0.8980, Normal Recall=0.9193, Normal Precision=0.9854, Attack Recall=0.9683, Attack Precision=0.8372

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
0.15       0.9388   0.9267   0.9191   0.9775   0.9683   0.8886   <--
0.20       0.9388   0.9267   0.9191   0.9775   0.9683   0.8886  
0.25       0.9388   0.9267   0.9191   0.9775   0.9683   0.8886  
0.30       0.9388   0.9267   0.9191   0.9775   0.9683   0.8886  
0.35       0.9388   0.9267   0.9191   0.9775   0.9683   0.8886  
0.40       0.9388   0.9267   0.9191   0.9775   0.9683   0.8886  
0.45       0.9388   0.9267   0.9191   0.9775   0.9683   0.8886  
0.50       0.9388   0.9267   0.9191   0.9775   0.9683   0.8886  
0.55       0.9388   0.9267   0.9191   0.9775   0.9683   0.8886  
0.60       0.9388   0.9267   0.9191   0.9775   0.9683   0.8886  
0.65       0.9388   0.9267   0.9191   0.9775   0.9683   0.8886  
0.70       0.9388   0.9267   0.9191   0.9775   0.9683   0.8886  
0.75       0.9388   0.9267   0.9191   0.9775   0.9683   0.8886  
0.80       0.9388   0.9267   0.9191   0.9775   0.9683   0.8886  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9388, F1=0.9267, Normal Recall=0.9191, Normal Precision=0.9775, Attack Recall=0.9683, Attack Precision=0.8886

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
0.15       0.9441   0.9454   0.9199   0.9667   0.9683   0.9236   <--
0.20       0.9441   0.9454   0.9199   0.9667   0.9683   0.9236  
0.25       0.9441   0.9454   0.9199   0.9667   0.9683   0.9236  
0.30       0.9441   0.9454   0.9199   0.9667   0.9683   0.9236  
0.35       0.9441   0.9454   0.9199   0.9667   0.9683   0.9236  
0.40       0.9441   0.9454   0.9199   0.9667   0.9683   0.9236  
0.45       0.9441   0.9454   0.9199   0.9667   0.9683   0.9236  
0.50       0.9441   0.9454   0.9199   0.9667   0.9683   0.9236  
0.55       0.9441   0.9454   0.9199   0.9667   0.9683   0.9236  
0.60       0.9441   0.9454   0.9199   0.9667   0.9683   0.9236  
0.65       0.9441   0.9454   0.9199   0.9667   0.9683   0.9236  
0.70       0.9441   0.9454   0.9199   0.9667   0.9683   0.9236  
0.75       0.9441   0.9454   0.9199   0.9667   0.9683   0.9236  
0.80       0.9441   0.9454   0.9199   0.9667   0.9683   0.9236  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9441, F1=0.9454, Normal Recall=0.9199, Normal Precision=0.9667, Attack Recall=0.9683, Attack Precision=0.9236

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
0.15       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691   <--
0.20       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.25       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.30       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.35       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.40       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.45       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.50       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.55       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.60       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.65       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.70       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.75       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.80       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9235, F1=0.7168, Normal Recall=0.9185, Normal Precision=0.9962, Attack Recall=0.9681, Attack Precision=0.5691

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
0.15       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487   <--
0.20       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.25       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.30       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.35       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.40       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.45       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.50       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.55       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.60       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.65       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.70       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.75       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.80       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9285, F1=0.8441, Normal Recall=0.9189, Normal Precision=0.9912, Attack Recall=0.9673, Attack Precision=0.7487

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
0.15       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367   <--
0.20       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.25       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.30       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.35       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.40       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.45       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.50       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.55       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.60       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.65       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.70       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.75       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.80       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9336, F1=0.8973, Normal Recall=0.9191, Normal Precision=0.9850, Attack Recall=0.9673, Attack Precision=0.8367

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
0.15       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884   <--
0.20       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.25       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.30       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.35       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.40       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.45       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.50       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.55       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.60       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.65       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.70       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.75       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.80       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9383, F1=0.9261, Normal Recall=0.9190, Normal Precision=0.9768, Attack Recall=0.9673, Attack Precision=0.8884

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
0.15       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234   <--
0.20       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.25       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.30       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.35       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.40       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.45       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.50       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.55       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.60       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.65       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.70       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.75       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.80       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9435, F1=0.9448, Normal Recall=0.9197, Normal Precision=0.9656, Attack Recall=0.9673, Attack Precision=0.9234

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
0.15       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691   <--
0.20       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.25       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.30       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.35       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.40       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.45       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.50       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.55       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.60       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.65       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.70       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.75       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
0.80       0.9235   0.7168   0.9185   0.9962   0.9681   0.5691  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9235, F1=0.7168, Normal Recall=0.9185, Normal Precision=0.9962, Attack Recall=0.9681, Attack Precision=0.5691

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
0.15       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487   <--
0.20       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.25       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.30       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.35       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.40       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.45       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.50       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.55       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.60       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.65       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.70       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.75       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
0.80       0.9285   0.8441   0.9189   0.9912   0.9673   0.7487  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9285, F1=0.8441, Normal Recall=0.9189, Normal Precision=0.9912, Attack Recall=0.9673, Attack Precision=0.7487

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
0.15       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367   <--
0.20       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.25       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.30       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.35       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.40       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.45       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.50       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.55       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.60       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.65       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.70       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.75       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
0.80       0.9336   0.8973   0.9191   0.9850   0.9673   0.8367  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9336, F1=0.8973, Normal Recall=0.9191, Normal Precision=0.9850, Attack Recall=0.9673, Attack Precision=0.8367

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
0.15       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884   <--
0.20       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.25       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.30       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.35       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.40       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.45       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.50       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.55       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.60       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.65       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.70       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.75       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
0.80       0.9383   0.9261   0.9190   0.9768   0.9673   0.8884  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9383, F1=0.9261, Normal Recall=0.9190, Normal Precision=0.9768, Attack Recall=0.9673, Attack Precision=0.8884

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
0.15       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234   <--
0.20       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.25       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.30       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.35       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.40       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.45       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.50       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.55       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.60       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.65       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.70       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.75       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
0.80       0.9435   0.9448   0.9197   0.9656   0.9673   0.9234  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9435, F1=0.9448, Normal Recall=0.9197, Normal Precision=0.9656, Attack Recall=0.9673, Attack Precision=0.9234

```

