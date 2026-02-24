# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e2_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-15 01:45:24 |

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
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5351 | 0.5511 | 0.5672 | 0.5833 | 0.5999 | 0.6179 | 0.6318 | 0.6479 | 0.6651 | 0.6819 | 0.6972 |
| QAT+Prune only | 0.4430 | 0.4989 | 0.5540 | 0.6098 | 0.6653 | 0.7191 | 0.7750 | 0.8291 | 0.8843 | 0.9399 | 0.9944 |
| QAT+PTQ | 0.4429 | 0.4989 | 0.5540 | 0.6099 | 0.6653 | 0.7191 | 0.7749 | 0.8293 | 0.8843 | 0.9400 | 0.9945 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.4429 | 0.4989 | 0.5540 | 0.6099 | 0.6653 | 0.7191 | 0.7749 | 0.8293 | 0.8843 | 0.9400 | 0.9945 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2371 | 0.3919 | 0.5009 | 0.5823 | 0.6459 | 0.6944 | 0.7349 | 0.7691 | 0.7978 | 0.8216 |
| QAT+Prune only | 0.0000 | 0.2841 | 0.4714 | 0.6046 | 0.7039 | 0.7797 | 0.8414 | 0.8907 | 0.9322 | 0.9675 | 0.9972 |
| QAT+PTQ | 0.0000 | 0.2841 | 0.4714 | 0.6047 | 0.7039 | 0.7797 | 0.8413 | 0.8908 | 0.9322 | 0.9676 | 0.9972 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2841 | 0.4714 | 0.6047 | 0.7039 | 0.7797 | 0.8413 | 0.8908 | 0.9322 | 0.9676 | 0.9972 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5351 | 0.5348 | 0.5348 | 0.5345 | 0.5351 | 0.5385 | 0.5337 | 0.5330 | 0.5369 | 0.5443 | 0.0000 |
| QAT+Prune only | 0.4430 | 0.4438 | 0.4439 | 0.4450 | 0.4459 | 0.4438 | 0.4459 | 0.4435 | 0.4441 | 0.4496 | 0.0000 |
| QAT+PTQ | 0.4429 | 0.4438 | 0.4439 | 0.4451 | 0.4458 | 0.4437 | 0.4456 | 0.4439 | 0.4436 | 0.4498 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.4429 | 0.4438 | 0.4439 | 0.4451 | 0.4458 | 0.4437 | 0.4456 | 0.4439 | 0.4436 | 0.4498 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5351 | 0.0000 | 0.0000 | 0.0000 | 0.5351 | 1.0000 |
| 90 | 10 | 299,940 | 0.5511 | 0.1428 | 0.6975 | 0.2371 | 0.5348 | 0.9409 |
| 80 | 20 | 291,350 | 0.5672 | 0.2725 | 0.6972 | 0.3919 | 0.5348 | 0.8760 |
| 70 | 30 | 194,230 | 0.5833 | 0.3909 | 0.6972 | 0.5009 | 0.5345 | 0.8046 |
| 60 | 40 | 145,675 | 0.5999 | 0.5000 | 0.6972 | 0.5823 | 0.5351 | 0.7261 |
| 50 | 50 | 116,540 | 0.6179 | 0.6017 | 0.6972 | 0.6459 | 0.5385 | 0.6401 |
| 40 | 60 | 97,115 | 0.6318 | 0.6916 | 0.6972 | 0.6944 | 0.5337 | 0.5402 |
| 30 | 70 | 83,240 | 0.6479 | 0.7769 | 0.6972 | 0.7349 | 0.5330 | 0.4300 |
| 20 | 80 | 72,835 | 0.6651 | 0.8576 | 0.6972 | 0.7691 | 0.5369 | 0.3071 |
| 10 | 90 | 64,740 | 0.6819 | 0.9323 | 0.6971 | 0.7978 | 0.5443 | 0.1665 |
| 0 | 100 | 58,270 | 0.6972 | 1.0000 | 0.6972 | 0.8216 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4430 | 0.0000 | 0.0000 | 0.0000 | 0.4430 | 1.0000 |
| 90 | 10 | 299,940 | 0.4989 | 0.1657 | 0.9944 | 0.2841 | 0.4438 | 0.9986 |
| 80 | 20 | 291,350 | 0.5540 | 0.3089 | 0.9944 | 0.4714 | 0.4439 | 0.9969 |
| 70 | 30 | 194,230 | 0.6098 | 0.4343 | 0.9944 | 0.6046 | 0.4450 | 0.9946 |
| 60 | 40 | 145,675 | 0.6653 | 0.5447 | 0.9944 | 0.7039 | 0.4459 | 0.9917 |
| 50 | 50 | 116,540 | 0.7191 | 0.6413 | 0.9944 | 0.7797 | 0.4438 | 0.9876 |
| 40 | 60 | 97,115 | 0.7750 | 0.7291 | 0.9944 | 0.8414 | 0.4459 | 0.9815 |
| 30 | 70 | 83,240 | 0.8291 | 0.8066 | 0.9944 | 0.8907 | 0.4435 | 0.9714 |
| 20 | 80 | 72,835 | 0.8843 | 0.8774 | 0.9944 | 0.9322 | 0.4441 | 0.9520 |
| 10 | 90 | 64,740 | 0.9399 | 0.9421 | 0.9944 | 0.9675 | 0.4496 | 0.8993 |
| 0 | 100 | 58,270 | 0.9944 | 1.0000 | 0.9944 | 0.9972 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4429 | 0.0000 | 0.0000 | 0.0000 | 0.4429 | 1.0000 |
| 90 | 10 | 299,940 | 0.4989 | 0.1657 | 0.9945 | 0.2841 | 0.4438 | 0.9986 |
| 80 | 20 | 291,350 | 0.5540 | 0.3090 | 0.9945 | 0.4714 | 0.4439 | 0.9969 |
| 70 | 30 | 194,230 | 0.6099 | 0.4344 | 0.9945 | 0.6047 | 0.4451 | 0.9947 |
| 60 | 40 | 145,675 | 0.6653 | 0.5447 | 0.9945 | 0.7039 | 0.4458 | 0.9919 |
| 50 | 50 | 116,540 | 0.7191 | 0.6413 | 0.9945 | 0.7797 | 0.4437 | 0.9878 |
| 40 | 60 | 97,115 | 0.7749 | 0.7290 | 0.9945 | 0.8413 | 0.4456 | 0.9818 |
| 30 | 70 | 83,240 | 0.8293 | 0.8067 | 0.9945 | 0.8908 | 0.4439 | 0.9719 |
| 20 | 80 | 72,835 | 0.8843 | 0.8773 | 0.9945 | 0.9322 | 0.4436 | 0.9528 |
| 10 | 90 | 64,740 | 0.9400 | 0.9421 | 0.9945 | 0.9676 | 0.4498 | 0.9010 |
| 0 | 100 | 58,270 | 0.9945 | 1.0000 | 0.9945 | 0.9972 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.4429 | 0.0000 | 0.0000 | 0.0000 | 0.4429 | 1.0000 |
| 90 | 10 | 299,940 | 0.4989 | 0.1657 | 0.9945 | 0.2841 | 0.4438 | 0.9986 |
| 80 | 20 | 291,350 | 0.5540 | 0.3090 | 0.9945 | 0.4714 | 0.4439 | 0.9969 |
| 70 | 30 | 194,230 | 0.6099 | 0.4344 | 0.9945 | 0.6047 | 0.4451 | 0.9947 |
| 60 | 40 | 145,675 | 0.6653 | 0.5447 | 0.9945 | 0.7039 | 0.4458 | 0.9919 |
| 50 | 50 | 116,540 | 0.7191 | 0.6413 | 0.9945 | 0.7797 | 0.4437 | 0.9878 |
| 40 | 60 | 97,115 | 0.7749 | 0.7290 | 0.9945 | 0.8413 | 0.4456 | 0.9818 |
| 30 | 70 | 83,240 | 0.8293 | 0.8067 | 0.9945 | 0.8908 | 0.4439 | 0.9719 |
| 20 | 80 | 72,835 | 0.8843 | 0.8773 | 0.9945 | 0.9322 | 0.4436 | 0.9528 |
| 10 | 90 | 64,740 | 0.9400 | 0.9421 | 0.9945 | 0.9676 | 0.4498 | 0.9010 |
| 0 | 100 | 58,270 | 0.9945 | 1.0000 | 0.9945 | 0.9972 | 0.0000 | 0.0000 |


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
0.15       0.5511   0.2372   0.5348   0.9409   0.6977   0.1429   <--
0.20       0.5511   0.2372   0.5348   0.9409   0.6977   0.1429  
0.25       0.5511   0.2372   0.5348   0.9409   0.6977   0.1429  
0.30       0.5511   0.2372   0.5348   0.9409   0.6977   0.1429  
0.35       0.5511   0.2372   0.5348   0.9409   0.6977   0.1429  
0.40       0.5511   0.2372   0.5348   0.9409   0.6977   0.1429  
0.45       0.5511   0.2372   0.5348   0.9409   0.6977   0.1429  
0.50       0.5511   0.2372   0.5348   0.9409   0.6977   0.1429  
0.55       0.5511   0.2372   0.5348   0.9409   0.6977   0.1429  
0.60       0.5511   0.2372   0.5348   0.9409   0.6977   0.1429  
0.65       0.5511   0.2372   0.5348   0.9409   0.6977   0.1429  
0.70       0.5511   0.2372   0.5348   0.9409   0.6977   0.1429  
0.75       0.5511   0.2372   0.5348   0.9409   0.6977   0.1429  
0.80       0.5511   0.2372   0.5348   0.9409   0.6977   0.1429  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5511, F1=0.2372, Normal Recall=0.5348, Normal Precision=0.9409, Attack Recall=0.6977, Attack Precision=0.1429

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
0.15       0.5670   0.3917   0.5345   0.8759   0.6972   0.2724   <--
0.20       0.5670   0.3917   0.5345   0.8759   0.6972   0.2724  
0.25       0.5670   0.3917   0.5345   0.8759   0.6972   0.2724  
0.30       0.5670   0.3917   0.5345   0.8759   0.6972   0.2724  
0.35       0.5670   0.3917   0.5345   0.8759   0.6972   0.2724  
0.40       0.5670   0.3917   0.5345   0.8759   0.6972   0.2724  
0.45       0.5670   0.3917   0.5345   0.8759   0.6972   0.2724  
0.50       0.5670   0.3917   0.5345   0.8759   0.6972   0.2724  
0.55       0.5670   0.3917   0.5345   0.8759   0.6972   0.2724  
0.60       0.5670   0.3917   0.5345   0.8759   0.6972   0.2724  
0.65       0.5670   0.3917   0.5345   0.8759   0.6972   0.2724  
0.70       0.5670   0.3917   0.5345   0.8759   0.6972   0.2724  
0.75       0.5670   0.3917   0.5345   0.8759   0.6972   0.2724  
0.80       0.5670   0.3917   0.5345   0.8759   0.6972   0.2724  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5670, F1=0.3917, Normal Recall=0.5345, Normal Precision=0.8759, Attack Recall=0.6972, Attack Precision=0.2724

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
0.15       0.5836   0.5012   0.5350   0.8048   0.6972   0.3912   <--
0.20       0.5836   0.5012   0.5350   0.8048   0.6972   0.3912  
0.25       0.5836   0.5012   0.5350   0.8048   0.6972   0.3912  
0.30       0.5836   0.5012   0.5350   0.8048   0.6972   0.3912  
0.35       0.5836   0.5012   0.5350   0.8048   0.6972   0.3912  
0.40       0.5836   0.5012   0.5350   0.8048   0.6972   0.3912  
0.45       0.5836   0.5012   0.5350   0.8048   0.6972   0.3912  
0.50       0.5836   0.5012   0.5350   0.8048   0.6972   0.3912  
0.55       0.5836   0.5012   0.5350   0.8048   0.6972   0.3912  
0.60       0.5836   0.5012   0.5350   0.8048   0.6972   0.3912  
0.65       0.5836   0.5012   0.5350   0.8048   0.6972   0.3912  
0.70       0.5836   0.5012   0.5350   0.8048   0.6972   0.3912  
0.75       0.5836   0.5012   0.5350   0.8048   0.6972   0.3912  
0.80       0.5836   0.5012   0.5350   0.8048   0.6972   0.3912  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5836, F1=0.5012, Normal Recall=0.5350, Normal Precision=0.8048, Attack Recall=0.6972, Attack Precision=0.3912

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
0.15       0.6003   0.5825   0.5357   0.7263   0.6972   0.5003   <--
0.20       0.6003   0.5825   0.5357   0.7263   0.6972   0.5003  
0.25       0.6003   0.5825   0.5357   0.7263   0.6972   0.5003  
0.30       0.6003   0.5825   0.5357   0.7263   0.6972   0.5003  
0.35       0.6003   0.5825   0.5357   0.7263   0.6972   0.5003  
0.40       0.6003   0.5825   0.5357   0.7263   0.6972   0.5003  
0.45       0.6003   0.5825   0.5357   0.7263   0.6972   0.5003  
0.50       0.6003   0.5825   0.5357   0.7263   0.6972   0.5003  
0.55       0.6003   0.5825   0.5357   0.7263   0.6972   0.5003  
0.60       0.6003   0.5825   0.5357   0.7263   0.6972   0.5003  
0.65       0.6003   0.5825   0.5357   0.7263   0.6972   0.5003  
0.70       0.6003   0.5825   0.5357   0.7263   0.6972   0.5003  
0.75       0.6003   0.5825   0.5357   0.7263   0.6972   0.5003  
0.80       0.6003   0.5825   0.5357   0.7263   0.6972   0.5003  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6003, F1=0.5825, Normal Recall=0.5357, Normal Precision=0.7263, Attack Recall=0.6972, Attack Precision=0.5003

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
0.15       0.6164   0.6451   0.5357   0.6388   0.6972   0.6002   <--
0.20       0.6164   0.6451   0.5357   0.6388   0.6972   0.6002  
0.25       0.6164   0.6451   0.5357   0.6388   0.6972   0.6002  
0.30       0.6164   0.6451   0.5357   0.6388   0.6972   0.6002  
0.35       0.6164   0.6451   0.5357   0.6388   0.6972   0.6002  
0.40       0.6164   0.6451   0.5357   0.6388   0.6972   0.6002  
0.45       0.6164   0.6451   0.5357   0.6388   0.6972   0.6002  
0.50       0.6164   0.6451   0.5357   0.6388   0.6972   0.6002  
0.55       0.6164   0.6451   0.5357   0.6388   0.6972   0.6002  
0.60       0.6164   0.6451   0.5357   0.6388   0.6972   0.6002  
0.65       0.6164   0.6451   0.5357   0.6388   0.6972   0.6002  
0.70       0.6164   0.6451   0.5357   0.6388   0.6972   0.6002  
0.75       0.6164   0.6451   0.5357   0.6388   0.6972   0.6002  
0.80       0.6164   0.6451   0.5357   0.6388   0.6972   0.6002  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6164, F1=0.6451, Normal Recall=0.5357, Normal Precision=0.6388, Attack Recall=0.6972, Attack Precision=0.6002

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
0.15       0.4989   0.2842   0.4438   0.9987   0.9947   0.1658   <--
0.20       0.4989   0.2842   0.4438   0.9987   0.9947   0.1658  
0.25       0.4989   0.2842   0.4438   0.9987   0.9947   0.1658  
0.30       0.4989   0.2842   0.4438   0.9987   0.9947   0.1658  
0.35       0.4989   0.2842   0.4438   0.9987   0.9947   0.1658  
0.40       0.4989   0.2842   0.4438   0.9987   0.9947   0.1658  
0.45       0.4989   0.2842   0.4438   0.9987   0.9947   0.1658  
0.50       0.4989   0.2842   0.4438   0.9987   0.9947   0.1658  
0.55       0.4989   0.2842   0.4438   0.9987   0.9947   0.1658  
0.60       0.4989   0.2842   0.4438   0.9987   0.9947   0.1658  
0.65       0.4989   0.2842   0.4438   0.9987   0.9947   0.1658  
0.70       0.4989   0.2842   0.4438   0.9987   0.9947   0.1658  
0.75       0.4989   0.2842   0.4438   0.9987   0.9947   0.1658  
0.80       0.4989   0.2842   0.4438   0.9987   0.9947   0.1658  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4989, F1=0.2842, Normal Recall=0.4438, Normal Precision=0.9987, Attack Recall=0.9947, Attack Precision=0.1658

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
0.15       0.5538   0.4713   0.4436   0.9969   0.9944   0.3088   <--
0.20       0.5538   0.4713   0.4436   0.9969   0.9944   0.3088  
0.25       0.5538   0.4713   0.4436   0.9969   0.9944   0.3088  
0.30       0.5538   0.4713   0.4436   0.9969   0.9944   0.3088  
0.35       0.5538   0.4713   0.4436   0.9969   0.9944   0.3088  
0.40       0.5538   0.4713   0.4436   0.9969   0.9944   0.3088  
0.45       0.5538   0.4713   0.4436   0.9969   0.9944   0.3088  
0.50       0.5538   0.4713   0.4436   0.9969   0.9944   0.3088  
0.55       0.5538   0.4713   0.4436   0.9969   0.9944   0.3088  
0.60       0.5538   0.4713   0.4436   0.9969   0.9944   0.3088  
0.65       0.5538   0.4713   0.4436   0.9969   0.9944   0.3088  
0.70       0.5538   0.4713   0.4436   0.9969   0.9944   0.3088  
0.75       0.5538   0.4713   0.4436   0.9969   0.9944   0.3088  
0.80       0.5538   0.4713   0.4436   0.9969   0.9944   0.3088  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5538, F1=0.4713, Normal Recall=0.4436, Normal Precision=0.9969, Attack Recall=0.9944, Attack Precision=0.3088

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
0.15       0.6092   0.6042   0.4441   0.9946   0.9944   0.4340   <--
0.20       0.6092   0.6042   0.4441   0.9946   0.9944   0.4340  
0.25       0.6092   0.6042   0.4441   0.9946   0.9944   0.4340  
0.30       0.6092   0.6042   0.4441   0.9946   0.9944   0.4340  
0.35       0.6092   0.6042   0.4441   0.9946   0.9944   0.4340  
0.40       0.6092   0.6042   0.4441   0.9946   0.9944   0.4340  
0.45       0.6092   0.6042   0.4441   0.9946   0.9944   0.4340  
0.50       0.6092   0.6042   0.4441   0.9946   0.9944   0.4340  
0.55       0.6092   0.6042   0.4441   0.9946   0.9944   0.4340  
0.60       0.6092   0.6042   0.4441   0.9946   0.9944   0.4340  
0.65       0.6092   0.6042   0.4441   0.9946   0.9944   0.4340  
0.70       0.6092   0.6042   0.4441   0.9946   0.9944   0.4340  
0.75       0.6092   0.6042   0.4441   0.9946   0.9944   0.4340  
0.80       0.6092   0.6042   0.4441   0.9946   0.9944   0.4340  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6092, F1=0.6042, Normal Recall=0.4441, Normal Precision=0.9946, Attack Recall=0.9944, Attack Precision=0.4340

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
0.15       0.6636   0.7028   0.4431   0.9917   0.9944   0.5435   <--
0.20       0.6636   0.7028   0.4431   0.9917   0.9944   0.5435  
0.25       0.6636   0.7028   0.4431   0.9917   0.9944   0.5435  
0.30       0.6636   0.7028   0.4431   0.9917   0.9944   0.5435  
0.35       0.6636   0.7028   0.4431   0.9917   0.9944   0.5435  
0.40       0.6636   0.7028   0.4431   0.9917   0.9944   0.5435  
0.45       0.6636   0.7028   0.4431   0.9917   0.9944   0.5435  
0.50       0.6636   0.7028   0.4431   0.9917   0.9944   0.5435  
0.55       0.6636   0.7028   0.4431   0.9917   0.9944   0.5435  
0.60       0.6636   0.7028   0.4431   0.9917   0.9944   0.5435  
0.65       0.6636   0.7028   0.4431   0.9917   0.9944   0.5435  
0.70       0.6636   0.7028   0.4431   0.9917   0.9944   0.5435  
0.75       0.6636   0.7028   0.4431   0.9917   0.9944   0.5435  
0.80       0.6636   0.7028   0.4431   0.9917   0.9944   0.5435  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6636, F1=0.7028, Normal Recall=0.4431, Normal Precision=0.9917, Attack Recall=0.9944, Attack Precision=0.5435

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
0.15       0.7181   0.7791   0.4418   0.9875   0.9944   0.6405   <--
0.20       0.7181   0.7791   0.4418   0.9875   0.9944   0.6405  
0.25       0.7181   0.7791   0.4418   0.9875   0.9944   0.6405  
0.30       0.7181   0.7791   0.4418   0.9875   0.9944   0.6405  
0.35       0.7181   0.7791   0.4418   0.9875   0.9944   0.6405  
0.40       0.7181   0.7791   0.4418   0.9875   0.9944   0.6405  
0.45       0.7181   0.7791   0.4418   0.9875   0.9944   0.6405  
0.50       0.7181   0.7791   0.4418   0.9875   0.9944   0.6405  
0.55       0.7181   0.7791   0.4418   0.9875   0.9944   0.6405  
0.60       0.7181   0.7791   0.4418   0.9875   0.9944   0.6405  
0.65       0.7181   0.7791   0.4418   0.9875   0.9944   0.6405  
0.70       0.7181   0.7791   0.4418   0.9875   0.9944   0.6405  
0.75       0.7181   0.7791   0.4418   0.9875   0.9944   0.6405  
0.80       0.7181   0.7791   0.4418   0.9875   0.9944   0.6405  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7181, F1=0.7791, Normal Recall=0.4418, Normal Precision=0.9875, Attack Recall=0.9944, Attack Precision=0.6405

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
0.15       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658   <--
0.20       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.25       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.30       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.35       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.40       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.45       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.50       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.55       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.60       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.65       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.70       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.75       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.80       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4989, F1=0.2842, Normal Recall=0.4438, Normal Precision=0.9987, Attack Recall=0.9949, Attack Precision=0.1658

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
0.15       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088   <--
0.20       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.25       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.30       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.35       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.40       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.45       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.50       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.55       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.60       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.65       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.70       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.75       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.80       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5538, F1=0.4713, Normal Recall=0.4436, Normal Precision=0.9969, Attack Recall=0.9945, Attack Precision=0.3088

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
0.15       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340   <--
0.20       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.25       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.30       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.35       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.40       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.45       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.50       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.55       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.60       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.65       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.70       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.75       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.80       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6092, F1=0.6043, Normal Recall=0.4441, Normal Precision=0.9947, Attack Recall=0.9945, Attack Precision=0.4340

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
0.15       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434   <--
0.20       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.25       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.30       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.35       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.40       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.45       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.50       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.55       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.60       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.65       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.70       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.75       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.80       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6635, F1=0.7028, Normal Recall=0.4428, Normal Precision=0.9918, Attack Recall=0.9945, Attack Precision=0.5434

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
0.15       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403   <--
0.20       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.25       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.30       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.35       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.40       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.45       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.50       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.55       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.60       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.65       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.70       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.75       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.80       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7179, F1=0.7790, Normal Recall=0.4412, Normal Precision=0.9877, Attack Recall=0.9945, Attack Precision=0.6403

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
0.15       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658   <--
0.20       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.25       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.30       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.35       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.40       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.45       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.50       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.55       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.60       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.65       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.70       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.75       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
0.80       0.4989   0.2842   0.4438   0.9987   0.9949   0.1658  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4989, F1=0.2842, Normal Recall=0.4438, Normal Precision=0.9987, Attack Recall=0.9949, Attack Precision=0.1658

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
0.15       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088   <--
0.20       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.25       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.30       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.35       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.40       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.45       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.50       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.55       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.60       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.65       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.70       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.75       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
0.80       0.5538   0.4713   0.4436   0.9969   0.9945   0.3088  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5538, F1=0.4713, Normal Recall=0.4436, Normal Precision=0.9969, Attack Recall=0.9945, Attack Precision=0.3088

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
0.15       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340   <--
0.20       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.25       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.30       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.35       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.40       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.45       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.50       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.55       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.60       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.65       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.70       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.75       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
0.80       0.6092   0.6043   0.4441   0.9947   0.9945   0.4340  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6092, F1=0.6043, Normal Recall=0.4441, Normal Precision=0.9947, Attack Recall=0.9945, Attack Precision=0.4340

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
0.15       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434   <--
0.20       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.25       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.30       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.35       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.40       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.45       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.50       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.55       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.60       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.65       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.70       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.75       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
0.80       0.6635   0.7028   0.4428   0.9918   0.9945   0.5434  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6635, F1=0.7028, Normal Recall=0.4428, Normal Precision=0.9918, Attack Recall=0.9945, Attack Precision=0.5434

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
0.15       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403   <--
0.20       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.25       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.30       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.35       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.40       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.45       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.50       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.55       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.60       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.65       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.70       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.75       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
0.80       0.7179   0.7790   0.4412   0.9877   0.9945   0.6403  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7179, F1=0.7790, Normal Recall=0.4412, Normal Precision=0.9877, Attack Recall=0.9945, Attack Precision=0.6403

```

