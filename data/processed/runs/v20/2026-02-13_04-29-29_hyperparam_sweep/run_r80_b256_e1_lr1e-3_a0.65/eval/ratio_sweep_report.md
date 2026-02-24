# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e1_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-18 15:20:41 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 80 |
| **Local epochs** | 1 |
| **Batch size** | 256 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8211 | 0.8382 | 0.8543 | 0.8717 | 0.8864 | 0.9031 | 0.9207 | 0.9366 | 0.9528 | 0.9690 | 0.9855 |
| QAT+Prune only | 0.5503 | 0.5958 | 0.6402 | 0.6854 | 0.7307 | 0.7743 | 0.8195 | 0.8646 | 0.9105 | 0.9535 | 1.0000 |
| QAT+PTQ | 0.5499 | 0.5955 | 0.6399 | 0.6852 | 0.7305 | 0.7741 | 0.8196 | 0.8646 | 0.9105 | 0.9535 | 1.0000 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.5499 | 0.5955 | 0.6399 | 0.6852 | 0.7305 | 0.7741 | 0.8196 | 0.8646 | 0.9105 | 0.9535 | 1.0000 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.5495 | 0.7301 | 0.8217 | 0.8740 | 0.9105 | 0.9372 | 0.9561 | 0.9709 | 0.9828 | 0.9927 |
| QAT+Prune only | 0.0000 | 0.3310 | 0.5265 | 0.6560 | 0.7482 | 0.8158 | 0.8693 | 0.9118 | 0.9470 | 0.9748 | 1.0000 |
| QAT+PTQ | 0.0000 | 0.3309 | 0.5262 | 0.6559 | 0.7480 | 0.8157 | 0.8693 | 0.9118 | 0.9470 | 0.9748 | 1.0000 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3309 | 0.5262 | 0.6559 | 0.7480 | 0.8157 | 0.8693 | 0.9118 | 0.9470 | 0.9748 | 1.0000 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8211 | 0.8218 | 0.8215 | 0.8229 | 0.8203 | 0.8208 | 0.8235 | 0.8226 | 0.8218 | 0.8205 | 0.0000 |
| QAT+Prune only | 0.5503 | 0.5509 | 0.5503 | 0.5506 | 0.5512 | 0.5486 | 0.5489 | 0.5488 | 0.5528 | 0.5349 | 0.0000 |
| QAT+PTQ | 0.5499 | 0.5506 | 0.5499 | 0.5503 | 0.5508 | 0.5483 | 0.5491 | 0.5487 | 0.5524 | 0.5349 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.5499 | 0.5506 | 0.5499 | 0.5503 | 0.5508 | 0.5483 | 0.5491 | 0.5487 | 0.5524 | 0.5349 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8211 | 0.0000 | 0.0000 | 0.0000 | 0.8211 | 1.0000 |
| 90 | 10 | 299,940 | 0.8382 | 0.3808 | 0.9865 | 0.5495 | 0.8218 | 0.9982 |
| 80 | 20 | 291,350 | 0.8543 | 0.5799 | 0.9855 | 0.7301 | 0.8215 | 0.9956 |
| 70 | 30 | 194,230 | 0.8717 | 0.7046 | 0.9855 | 0.8217 | 0.8229 | 0.9925 |
| 60 | 40 | 145,675 | 0.8864 | 0.7852 | 0.9855 | 0.8740 | 0.8203 | 0.9884 |
| 50 | 50 | 116,540 | 0.9031 | 0.8461 | 0.9855 | 0.9105 | 0.8208 | 0.9826 |
| 40 | 60 | 97,115 | 0.9207 | 0.8933 | 0.9855 | 0.9372 | 0.8235 | 0.9743 |
| 30 | 70 | 83,240 | 0.9366 | 0.9284 | 0.9855 | 0.9561 | 0.8226 | 0.9605 |
| 20 | 80 | 72,835 | 0.9528 | 0.9567 | 0.9855 | 0.9709 | 0.8218 | 0.9341 |
| 10 | 90 | 64,740 | 0.9690 | 0.9802 | 0.9855 | 0.9828 | 0.8205 | 0.8628 |
| 0 | 100 | 58,270 | 0.9855 | 1.0000 | 0.9855 | 0.9927 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5503 | 0.0000 | 0.0000 | 0.0000 | 0.5503 | 1.0000 |
| 90 | 10 | 299,940 | 0.5958 | 0.1983 | 1.0000 | 0.3310 | 0.5509 | 1.0000 |
| 80 | 20 | 291,350 | 0.6402 | 0.3573 | 1.0000 | 0.5265 | 0.5503 | 1.0000 |
| 70 | 30 | 194,230 | 0.6854 | 0.4881 | 1.0000 | 0.6560 | 0.5506 | 1.0000 |
| 60 | 40 | 145,675 | 0.7307 | 0.5977 | 1.0000 | 0.7482 | 0.5512 | 1.0000 |
| 50 | 50 | 116,540 | 0.7743 | 0.6890 | 1.0000 | 0.8158 | 0.5486 | 0.9999 |
| 40 | 60 | 97,115 | 0.8195 | 0.7688 | 1.0000 | 0.8693 | 0.5489 | 0.9999 |
| 30 | 70 | 83,240 | 0.8646 | 0.8379 | 1.0000 | 0.9118 | 0.5488 | 0.9999 |
| 20 | 80 | 72,835 | 0.9105 | 0.8994 | 1.0000 | 0.9470 | 0.5528 | 0.9998 |
| 10 | 90 | 64,740 | 0.9535 | 0.9509 | 1.0000 | 0.9748 | 0.5349 | 0.9994 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5499 | 0.0000 | 0.0000 | 0.0000 | 0.5499 | 1.0000 |
| 90 | 10 | 299,940 | 0.5955 | 0.1982 | 1.0000 | 0.3309 | 0.5506 | 1.0000 |
| 80 | 20 | 291,350 | 0.6399 | 0.3571 | 1.0000 | 0.5262 | 0.5499 | 1.0000 |
| 70 | 30 | 194,230 | 0.6852 | 0.4880 | 1.0000 | 0.6559 | 0.5503 | 1.0000 |
| 60 | 40 | 145,675 | 0.7305 | 0.5975 | 1.0000 | 0.7480 | 0.5508 | 1.0000 |
| 50 | 50 | 116,540 | 0.7741 | 0.6888 | 1.0000 | 0.8157 | 0.5483 | 0.9999 |
| 40 | 60 | 97,115 | 0.8196 | 0.7689 | 1.0000 | 0.8693 | 0.5491 | 0.9999 |
| 30 | 70 | 83,240 | 0.8646 | 0.8379 | 1.0000 | 0.9118 | 0.5487 | 0.9999 |
| 20 | 80 | 72,835 | 0.9105 | 0.8994 | 1.0000 | 0.9470 | 0.5524 | 0.9998 |
| 10 | 90 | 64,740 | 0.9535 | 0.9509 | 1.0000 | 0.9748 | 0.5349 | 0.9994 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.5499 | 0.0000 | 0.0000 | 0.0000 | 0.5499 | 1.0000 |
| 90 | 10 | 299,940 | 0.5955 | 0.1982 | 1.0000 | 0.3309 | 0.5506 | 1.0000 |
| 80 | 20 | 291,350 | 0.6399 | 0.3571 | 1.0000 | 0.5262 | 0.5499 | 1.0000 |
| 70 | 30 | 194,230 | 0.6852 | 0.4880 | 1.0000 | 0.6559 | 0.5503 | 1.0000 |
| 60 | 40 | 145,675 | 0.7305 | 0.5975 | 1.0000 | 0.7480 | 0.5508 | 1.0000 |
| 50 | 50 | 116,540 | 0.7741 | 0.6888 | 1.0000 | 0.8157 | 0.5483 | 0.9999 |
| 40 | 60 | 97,115 | 0.8196 | 0.7689 | 1.0000 | 0.8693 | 0.5491 | 0.9999 |
| 30 | 70 | 83,240 | 0.8646 | 0.8379 | 1.0000 | 0.9118 | 0.5487 | 0.9999 |
| 20 | 80 | 72,835 | 0.9105 | 0.8994 | 1.0000 | 0.9470 | 0.5524 | 0.9998 |
| 10 | 90 | 64,740 | 0.9535 | 0.9509 | 1.0000 | 0.9748 | 0.5349 | 0.9994 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |


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
0.15       0.8382   0.5493   0.8218   0.9981   0.9861   0.3807   <--
0.20       0.8382   0.5493   0.8218   0.9981   0.9861   0.3807  
0.25       0.8382   0.5493   0.8218   0.9981   0.9861   0.3807  
0.30       0.8382   0.5493   0.8218   0.9981   0.9861   0.3807  
0.35       0.8382   0.5493   0.8218   0.9981   0.9861   0.3807  
0.40       0.8382   0.5493   0.8218   0.9981   0.9861   0.3807  
0.45       0.8382   0.5493   0.8218   0.9981   0.9861   0.3807  
0.50       0.8382   0.5493   0.8218   0.9981   0.9861   0.3807  
0.55       0.8382   0.5493   0.8218   0.9981   0.9861   0.3807  
0.60       0.8382   0.5493   0.8218   0.9981   0.9861   0.3807  
0.65       0.8382   0.5493   0.8218   0.9981   0.9861   0.3807  
0.70       0.8382   0.5493   0.8218   0.9981   0.9861   0.3807  
0.75       0.8382   0.5493   0.8218   0.9981   0.9861   0.3807  
0.80       0.8382   0.5493   0.8218   0.9981   0.9861   0.3807  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8382, F1=0.5493, Normal Recall=0.8218, Normal Precision=0.9981, Attack Recall=0.9861, Attack Precision=0.3807

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
0.15       0.8548   0.7308   0.8221   0.9956   0.9855   0.5807   <--
0.20       0.8548   0.7308   0.8221   0.9956   0.9855   0.5807  
0.25       0.8548   0.7308   0.8221   0.9956   0.9855   0.5807  
0.30       0.8548   0.7308   0.8221   0.9956   0.9855   0.5807  
0.35       0.8548   0.7308   0.8221   0.9956   0.9855   0.5807  
0.40       0.8548   0.7308   0.8221   0.9956   0.9855   0.5807  
0.45       0.8548   0.7308   0.8221   0.9956   0.9855   0.5807  
0.50       0.8548   0.7308   0.8221   0.9956   0.9855   0.5807  
0.55       0.8548   0.7308   0.8221   0.9956   0.9855   0.5807  
0.60       0.8548   0.7308   0.8221   0.9956   0.9855   0.5807  
0.65       0.8548   0.7308   0.8221   0.9956   0.9855   0.5807  
0.70       0.8548   0.7308   0.8221   0.9956   0.9855   0.5807  
0.75       0.8548   0.7308   0.8221   0.9956   0.9855   0.5807  
0.80       0.8548   0.7308   0.8221   0.9956   0.9855   0.5807  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8548, F1=0.7308, Normal Recall=0.8221, Normal Precision=0.9956, Attack Recall=0.9855, Attack Precision=0.5807

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
0.15       0.8708   0.8207   0.8217   0.9925   0.9855   0.7031   <--
0.20       0.8708   0.8207   0.8217   0.9925   0.9855   0.7031  
0.25       0.8708   0.8207   0.8217   0.9925   0.9855   0.7031  
0.30       0.8708   0.8207   0.8217   0.9925   0.9855   0.7031  
0.35       0.8708   0.8207   0.8217   0.9925   0.9855   0.7031  
0.40       0.8708   0.8207   0.8217   0.9925   0.9855   0.7031  
0.45       0.8708   0.8207   0.8217   0.9925   0.9855   0.7031  
0.50       0.8708   0.8207   0.8217   0.9925   0.9855   0.7031  
0.55       0.8708   0.8207   0.8217   0.9925   0.9855   0.7031  
0.60       0.8708   0.8207   0.8217   0.9925   0.9855   0.7031  
0.65       0.8708   0.8207   0.8217   0.9925   0.9855   0.7031  
0.70       0.8708   0.8207   0.8217   0.9925   0.9855   0.7031  
0.75       0.8708   0.8207   0.8217   0.9925   0.9855   0.7031  
0.80       0.8708   0.8207   0.8217   0.9925   0.9855   0.7031  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8708, F1=0.8207, Normal Recall=0.8217, Normal Precision=0.9925, Attack Recall=0.9855, Attack Precision=0.7031

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
0.15       0.8873   0.8750   0.8219   0.9884   0.9855   0.7867   <--
0.20       0.8873   0.8750   0.8219   0.9884   0.9855   0.7867  
0.25       0.8873   0.8750   0.8219   0.9884   0.9855   0.7867  
0.30       0.8873   0.8750   0.8219   0.9884   0.9855   0.7867  
0.35       0.8873   0.8750   0.8219   0.9884   0.9855   0.7867  
0.40       0.8873   0.8750   0.8219   0.9884   0.9855   0.7867  
0.45       0.8873   0.8750   0.8219   0.9884   0.9855   0.7867  
0.50       0.8873   0.8750   0.8219   0.9884   0.9855   0.7867  
0.55       0.8873   0.8750   0.8219   0.9884   0.9855   0.7867  
0.60       0.8873   0.8750   0.8219   0.9884   0.9855   0.7867  
0.65       0.8873   0.8750   0.8219   0.9884   0.9855   0.7867  
0.70       0.8873   0.8750   0.8219   0.9884   0.9855   0.7867  
0.75       0.8873   0.8750   0.8219   0.9884   0.9855   0.7867  
0.80       0.8873   0.8750   0.8219   0.9884   0.9855   0.7867  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8873, F1=0.8750, Normal Recall=0.8219, Normal Precision=0.9884, Attack Recall=0.9855, Attack Precision=0.7867

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
0.15       0.9036   0.9109   0.8218   0.9827   0.9855   0.8469   <--
0.20       0.9036   0.9109   0.8218   0.9827   0.9855   0.8469  
0.25       0.9036   0.9109   0.8218   0.9827   0.9855   0.8469  
0.30       0.9036   0.9109   0.8218   0.9827   0.9855   0.8469  
0.35       0.9036   0.9109   0.8218   0.9827   0.9855   0.8469  
0.40       0.9036   0.9109   0.8218   0.9827   0.9855   0.8469  
0.45       0.9036   0.9109   0.8218   0.9827   0.9855   0.8469  
0.50       0.9036   0.9109   0.8218   0.9827   0.9855   0.8469  
0.55       0.9036   0.9109   0.8218   0.9827   0.9855   0.8469  
0.60       0.9036   0.9109   0.8218   0.9827   0.9855   0.8469  
0.65       0.9036   0.9109   0.8218   0.9827   0.9855   0.8469  
0.70       0.9036   0.9109   0.8218   0.9827   0.9855   0.8469  
0.75       0.9036   0.9109   0.8218   0.9827   0.9855   0.8469  
0.80       0.9036   0.9109   0.8218   0.9827   0.9855   0.8469  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9036, F1=0.9109, Normal Recall=0.8218, Normal Precision=0.9827, Attack Recall=0.9855, Attack Precision=0.8469

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
0.15       0.5958   0.3310   0.5509   1.0000   1.0000   0.1983   <--
0.20       0.5958   0.3310   0.5509   1.0000   1.0000   0.1983  
0.25       0.5958   0.3310   0.5509   1.0000   1.0000   0.1983  
0.30       0.5958   0.3310   0.5509   1.0000   1.0000   0.1983  
0.35       0.5958   0.3310   0.5509   1.0000   1.0000   0.1983  
0.40       0.5958   0.3310   0.5509   1.0000   1.0000   0.1983  
0.45       0.5958   0.3310   0.5509   1.0000   1.0000   0.1983  
0.50       0.5958   0.3310   0.5509   1.0000   1.0000   0.1983  
0.55       0.5958   0.3310   0.5509   1.0000   1.0000   0.1983  
0.60       0.5958   0.3310   0.5509   1.0000   1.0000   0.1983  
0.65       0.5958   0.3310   0.5509   1.0000   1.0000   0.1983  
0.70       0.5958   0.3310   0.5509   1.0000   1.0000   0.1983  
0.75       0.5958   0.3310   0.5509   1.0000   1.0000   0.1983  
0.80       0.5958   0.3310   0.5509   1.0000   1.0000   0.1983  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5958, F1=0.3310, Normal Recall=0.5509, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1983

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
0.15       0.6410   0.5270   0.5513   1.0000   1.0000   0.3578   <--
0.20       0.6410   0.5270   0.5513   1.0000   1.0000   0.3578  
0.25       0.6410   0.5270   0.5513   1.0000   1.0000   0.3578  
0.30       0.6410   0.5270   0.5513   1.0000   1.0000   0.3578  
0.35       0.6410   0.5270   0.5513   1.0000   1.0000   0.3578  
0.40       0.6410   0.5270   0.5513   1.0000   1.0000   0.3578  
0.45       0.6410   0.5270   0.5513   1.0000   1.0000   0.3578  
0.50       0.6410   0.5270   0.5513   1.0000   1.0000   0.3578  
0.55       0.6410   0.5270   0.5513   1.0000   1.0000   0.3578  
0.60       0.6410   0.5270   0.5513   1.0000   1.0000   0.3578  
0.65       0.6410   0.5270   0.5513   1.0000   1.0000   0.3578  
0.70       0.6410   0.5270   0.5513   1.0000   1.0000   0.3578  
0.75       0.6410   0.5270   0.5513   1.0000   1.0000   0.3578  
0.80       0.6410   0.5270   0.5513   1.0000   1.0000   0.3578  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6410, F1=0.5270, Normal Recall=0.5513, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3578

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
0.15       0.6857   0.6562   0.5510   1.0000   1.0000   0.4884   <--
0.20       0.6857   0.6562   0.5510   1.0000   1.0000   0.4884  
0.25       0.6857   0.6562   0.5510   1.0000   1.0000   0.4884  
0.30       0.6857   0.6562   0.5510   1.0000   1.0000   0.4884  
0.35       0.6857   0.6562   0.5510   1.0000   1.0000   0.4884  
0.40       0.6857   0.6562   0.5510   1.0000   1.0000   0.4884  
0.45       0.6857   0.6562   0.5510   1.0000   1.0000   0.4884  
0.50       0.6857   0.6562   0.5510   1.0000   1.0000   0.4884  
0.55       0.6857   0.6562   0.5510   1.0000   1.0000   0.4884  
0.60       0.6857   0.6562   0.5510   1.0000   1.0000   0.4884  
0.65       0.6857   0.6562   0.5510   1.0000   1.0000   0.4884  
0.70       0.6857   0.6562   0.5510   1.0000   1.0000   0.4884  
0.75       0.6857   0.6562   0.5510   1.0000   1.0000   0.4884  
0.80       0.6857   0.6562   0.5510   1.0000   1.0000   0.4884  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6857, F1=0.6562, Normal Recall=0.5510, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4884

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
0.15       0.7303   0.7479   0.5505   1.0000   1.0000   0.5973   <--
0.20       0.7303   0.7479   0.5505   1.0000   1.0000   0.5973  
0.25       0.7303   0.7479   0.5505   1.0000   1.0000   0.5973  
0.30       0.7303   0.7479   0.5505   1.0000   1.0000   0.5973  
0.35       0.7303   0.7479   0.5505   1.0000   1.0000   0.5973  
0.40       0.7303   0.7479   0.5505   1.0000   1.0000   0.5973  
0.45       0.7303   0.7479   0.5505   1.0000   1.0000   0.5973  
0.50       0.7303   0.7479   0.5505   1.0000   1.0000   0.5973  
0.55       0.7303   0.7479   0.5505   1.0000   1.0000   0.5973  
0.60       0.7303   0.7479   0.5505   1.0000   1.0000   0.5973  
0.65       0.7303   0.7479   0.5505   1.0000   1.0000   0.5973  
0.70       0.7303   0.7479   0.5505   1.0000   1.0000   0.5973  
0.75       0.7303   0.7479   0.5505   1.0000   1.0000   0.5973  
0.80       0.7303   0.7479   0.5505   1.0000   1.0000   0.5973  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7303, F1=0.7479, Normal Recall=0.5505, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5973

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
0.15       0.7745   0.8160   0.5490   0.9999   1.0000   0.6892   <--
0.20       0.7745   0.8160   0.5490   0.9999   1.0000   0.6892  
0.25       0.7745   0.8160   0.5490   0.9999   1.0000   0.6892  
0.30       0.7745   0.8160   0.5490   0.9999   1.0000   0.6892  
0.35       0.7745   0.8160   0.5490   0.9999   1.0000   0.6892  
0.40       0.7745   0.8160   0.5490   0.9999   1.0000   0.6892  
0.45       0.7745   0.8160   0.5490   0.9999   1.0000   0.6892  
0.50       0.7745   0.8160   0.5490   0.9999   1.0000   0.6892  
0.55       0.7745   0.8160   0.5490   0.9999   1.0000   0.6892  
0.60       0.7745   0.8160   0.5490   0.9999   1.0000   0.6892  
0.65       0.7745   0.8160   0.5490   0.9999   1.0000   0.6892  
0.70       0.7745   0.8160   0.5490   0.9999   1.0000   0.6892  
0.75       0.7745   0.8160   0.5490   0.9999   1.0000   0.6892  
0.80       0.7745   0.8160   0.5490   0.9999   1.0000   0.6892  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7745, F1=0.8160, Normal Recall=0.5490, Normal Precision=0.9999, Attack Recall=1.0000, Attack Precision=0.6892

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
0.15       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982   <--
0.20       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.25       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.30       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.35       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.40       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.45       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.50       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.55       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.60       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.65       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.70       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.75       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.80       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5955, F1=0.3309, Normal Recall=0.5506, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1982

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
0.15       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576   <--
0.20       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.25       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.30       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.35       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.40       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.45       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.50       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.55       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.60       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.65       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.70       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.75       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.80       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6407, F1=0.5268, Normal Recall=0.5509, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3576

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
0.15       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882   <--
0.20       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.25       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.30       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.35       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.40       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.45       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.50       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.55       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.60       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.65       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.70       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.75       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.80       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6854, F1=0.6560, Normal Recall=0.5506, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4882

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
0.15       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971   <--
0.20       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.25       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.30       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.35       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.40       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.45       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.50       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.55       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.60       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.65       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.70       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.75       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.80       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7301, F1=0.7477, Normal Recall=0.5502, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5971

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
0.15       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890   <--
0.20       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.25       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.30       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.35       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.40       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.45       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.50       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.55       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.60       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.65       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.70       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.75       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.80       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7743, F1=0.8159, Normal Recall=0.5487, Normal Precision=0.9999, Attack Recall=1.0000, Attack Precision=0.6890

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
0.15       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982   <--
0.20       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.25       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.30       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.35       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.40       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.45       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.50       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.55       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.60       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.65       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.70       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.75       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
0.80       0.5955   0.3309   0.5506   1.0000   1.0000   0.1982  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5955, F1=0.3309, Normal Recall=0.5506, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1982

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
0.15       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576   <--
0.20       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.25       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.30       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.35       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.40       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.45       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.50       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.55       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.60       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.65       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.70       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.75       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
0.80       0.6407   0.5268   0.5509   1.0000   1.0000   0.3576  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6407, F1=0.5268, Normal Recall=0.5509, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3576

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
0.15       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882   <--
0.20       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.25       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.30       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.35       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.40       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.45       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.50       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.55       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.60       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.65       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.70       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.75       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
0.80       0.6854   0.6560   0.5506   1.0000   1.0000   0.4882  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6854, F1=0.6560, Normal Recall=0.5506, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4882

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
0.15       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971   <--
0.20       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.25       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.30       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.35       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.40       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.45       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.50       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.55       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.60       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.65       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.70       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.75       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
0.80       0.7301   0.7477   0.5502   1.0000   1.0000   0.5971  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7301, F1=0.7477, Normal Recall=0.5502, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5971

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
0.15       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890   <--
0.20       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.25       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.30       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.35       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.40       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.45       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.50       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.55       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.60       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.65       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.70       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.75       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
0.80       0.7743   0.8159   0.5487   0.9999   1.0000   0.6890  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7743, F1=0.8159, Normal Recall=0.5487, Normal Precision=0.9999, Attack Recall=1.0000, Attack Precision=0.6890

```

