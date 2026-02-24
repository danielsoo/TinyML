# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e2_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-17 00:53:21 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7788 | 0.7715 | 0.7632 | 0.7558 | 0.7453 | 0.7369 | 0.7290 | 0.7204 | 0.7122 | 0.7038 | 0.6959 |
| QAT+Prune only | 0.6700 | 0.7042 | 0.7365 | 0.7686 | 0.8019 | 0.8342 | 0.8678 | 0.8989 | 0.9329 | 0.9646 | 0.9981 |
| QAT+PTQ | 0.6688 | 0.7030 | 0.7355 | 0.7678 | 0.8011 | 0.8336 | 0.8673 | 0.8987 | 0.9325 | 0.9645 | 0.9982 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6688 | 0.7030 | 0.7355 | 0.7678 | 0.8011 | 0.8336 | 0.8673 | 0.8987 | 0.9325 | 0.9645 | 0.9982 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3781 | 0.5404 | 0.6310 | 0.6861 | 0.7257 | 0.7550 | 0.7770 | 0.7946 | 0.8087 | 0.8207 |
| QAT+Prune only | 0.0000 | 0.4029 | 0.6024 | 0.7213 | 0.8012 | 0.8576 | 0.9006 | 0.9325 | 0.9597 | 0.9807 | 0.9991 |
| QAT+PTQ | 0.0000 | 0.4020 | 0.6016 | 0.7206 | 0.8006 | 0.8571 | 0.9003 | 0.9324 | 0.9595 | 0.9806 | 0.9991 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4020 | 0.6016 | 0.7206 | 0.8006 | 0.8571 | 0.9003 | 0.9324 | 0.9595 | 0.9806 | 0.9991 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7788 | 0.7800 | 0.7801 | 0.7814 | 0.7782 | 0.7779 | 0.7787 | 0.7776 | 0.7775 | 0.7745 | 0.0000 |
| QAT+Prune only | 0.6700 | 0.6715 | 0.6711 | 0.6702 | 0.6710 | 0.6703 | 0.6723 | 0.6673 | 0.6721 | 0.6631 | 0.0000 |
| QAT+PTQ | 0.6688 | 0.6702 | 0.6699 | 0.6691 | 0.6698 | 0.6690 | 0.6711 | 0.6665 | 0.6699 | 0.6617 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6688 | 0.6702 | 0.6699 | 0.6691 | 0.6698 | 0.6690 | 0.6711 | 0.6665 | 0.6699 | 0.6617 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7788 | 0.0000 | 0.0000 | 0.0000 | 0.7788 | 1.0000 |
| 90 | 10 | 299,940 | 0.7715 | 0.2597 | 0.6947 | 0.3781 | 0.7800 | 0.9583 |
| 80 | 20 | 291,350 | 0.7632 | 0.4417 | 0.6959 | 0.5404 | 0.7801 | 0.9112 |
| 70 | 30 | 194,230 | 0.7558 | 0.5771 | 0.6959 | 0.6310 | 0.7814 | 0.8571 |
| 60 | 40 | 145,675 | 0.7453 | 0.6766 | 0.6959 | 0.6861 | 0.7782 | 0.7933 |
| 50 | 50 | 116,540 | 0.7369 | 0.7581 | 0.6959 | 0.7257 | 0.7779 | 0.7190 |
| 40 | 60 | 97,115 | 0.7290 | 0.8251 | 0.6959 | 0.7550 | 0.7787 | 0.6306 |
| 30 | 70 | 83,240 | 0.7204 | 0.8795 | 0.6959 | 0.7770 | 0.7776 | 0.5228 |
| 20 | 80 | 72,835 | 0.7122 | 0.9260 | 0.6959 | 0.7946 | 0.7775 | 0.3900 |
| 10 | 90 | 64,740 | 0.7038 | 0.9652 | 0.6959 | 0.8087 | 0.7745 | 0.2206 |
| 0 | 100 | 58,270 | 0.6959 | 1.0000 | 0.6959 | 0.8207 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6700 | 0.0000 | 0.0000 | 0.0000 | 0.6700 | 1.0000 |
| 90 | 10 | 299,940 | 0.7042 | 0.2524 | 0.9982 | 0.4029 | 0.6715 | 0.9997 |
| 80 | 20 | 291,350 | 0.7365 | 0.4314 | 0.9981 | 0.6024 | 0.6711 | 0.9993 |
| 70 | 30 | 194,230 | 0.7686 | 0.5647 | 0.9981 | 0.7213 | 0.6702 | 0.9988 |
| 60 | 40 | 145,675 | 0.8019 | 0.6692 | 0.9981 | 0.8012 | 0.6710 | 0.9982 |
| 50 | 50 | 116,540 | 0.8342 | 0.7517 | 0.9981 | 0.8576 | 0.6703 | 0.9972 |
| 40 | 60 | 97,115 | 0.8678 | 0.8204 | 0.9981 | 0.9006 | 0.6723 | 0.9959 |
| 30 | 70 | 83,240 | 0.8989 | 0.8750 | 0.9981 | 0.9325 | 0.6673 | 0.9936 |
| 20 | 80 | 72,835 | 0.9329 | 0.9241 | 0.9981 | 0.9597 | 0.6721 | 0.9891 |
| 10 | 90 | 64,740 | 0.9646 | 0.9639 | 0.9981 | 0.9807 | 0.6631 | 0.9755 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9991 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6688 | 0.0000 | 0.0000 | 0.0000 | 0.6688 | 1.0000 |
| 90 | 10 | 299,940 | 0.7030 | 0.2517 | 0.9982 | 0.4020 | 0.6702 | 0.9997 |
| 80 | 20 | 291,350 | 0.7355 | 0.4305 | 0.9982 | 0.6016 | 0.6699 | 0.9993 |
| 70 | 30 | 194,230 | 0.7678 | 0.5638 | 0.9982 | 0.7206 | 0.6691 | 0.9988 |
| 60 | 40 | 145,675 | 0.8011 | 0.6683 | 0.9982 | 0.8006 | 0.6698 | 0.9982 |
| 50 | 50 | 116,540 | 0.8336 | 0.7510 | 0.9982 | 0.8571 | 0.6690 | 0.9973 |
| 40 | 60 | 97,115 | 0.8673 | 0.8199 | 0.9982 | 0.9003 | 0.6711 | 0.9959 |
| 30 | 70 | 83,240 | 0.8987 | 0.8747 | 0.9982 | 0.9324 | 0.6665 | 0.9936 |
| 20 | 80 | 72,835 | 0.9325 | 0.9236 | 0.9982 | 0.9595 | 0.6699 | 0.9892 |
| 10 | 90 | 64,740 | 0.9645 | 0.9637 | 0.9982 | 0.9806 | 0.6617 | 0.9756 |
| 0 | 100 | 58,270 | 0.9982 | 1.0000 | 0.9982 | 0.9991 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6688 | 0.0000 | 0.0000 | 0.0000 | 0.6688 | 1.0000 |
| 90 | 10 | 299,940 | 0.7030 | 0.2517 | 0.9982 | 0.4020 | 0.6702 | 0.9997 |
| 80 | 20 | 291,350 | 0.7355 | 0.4305 | 0.9982 | 0.6016 | 0.6699 | 0.9993 |
| 70 | 30 | 194,230 | 0.7678 | 0.5638 | 0.9982 | 0.7206 | 0.6691 | 0.9988 |
| 60 | 40 | 145,675 | 0.8011 | 0.6683 | 0.9982 | 0.8006 | 0.6698 | 0.9982 |
| 50 | 50 | 116,540 | 0.8336 | 0.7510 | 0.9982 | 0.8571 | 0.6690 | 0.9973 |
| 40 | 60 | 97,115 | 0.8673 | 0.8199 | 0.9982 | 0.9003 | 0.6711 | 0.9959 |
| 30 | 70 | 83,240 | 0.8987 | 0.8747 | 0.9982 | 0.9324 | 0.6665 | 0.9936 |
| 20 | 80 | 72,835 | 0.9325 | 0.9236 | 0.9982 | 0.9595 | 0.6699 | 0.9892 |
| 10 | 90 | 64,740 | 0.9645 | 0.9637 | 0.9982 | 0.9806 | 0.6617 | 0.9756 |
| 0 | 100 | 58,270 | 0.9982 | 1.0000 | 0.9982 | 0.9991 | 0.0000 | 0.0000 |


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
0.15       0.7717   0.3790   0.7800   0.9586   0.6967   0.2603   <--
0.20       0.7717   0.3790   0.7800   0.9586   0.6967   0.2603  
0.25       0.7717   0.3790   0.7800   0.9586   0.6967   0.2603  
0.30       0.7717   0.3790   0.7800   0.9586   0.6967   0.2603  
0.35       0.7717   0.3790   0.7800   0.9586   0.6967   0.2603  
0.40       0.7717   0.3790   0.7800   0.9586   0.6967   0.2603  
0.45       0.7717   0.3790   0.7800   0.9586   0.6967   0.2603  
0.50       0.7717   0.3790   0.7800   0.9586   0.6967   0.2603  
0.55       0.7717   0.3790   0.7800   0.9586   0.6967   0.2603  
0.60       0.7717   0.3790   0.7800   0.9586   0.6967   0.2603  
0.65       0.7717   0.3790   0.7800   0.9586   0.6967   0.2603  
0.70       0.7717   0.3790   0.7800   0.9586   0.6967   0.2603  
0.75       0.7717   0.3790   0.7800   0.9586   0.6967   0.2603  
0.80       0.7717   0.3790   0.7800   0.9586   0.6967   0.2603  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7717, F1=0.3790, Normal Recall=0.7800, Normal Precision=0.9586, Attack Recall=0.6967, Attack Precision=0.2603

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
0.15       0.7633   0.5405   0.7802   0.9112   0.6959   0.4418   <--
0.20       0.7633   0.5405   0.7802   0.9112   0.6959   0.4418  
0.25       0.7633   0.5405   0.7802   0.9112   0.6959   0.4418  
0.30       0.7633   0.5405   0.7802   0.9112   0.6959   0.4418  
0.35       0.7633   0.5405   0.7802   0.9112   0.6959   0.4418  
0.40       0.7633   0.5405   0.7802   0.9112   0.6959   0.4418  
0.45       0.7633   0.5405   0.7802   0.9112   0.6959   0.4418  
0.50       0.7633   0.5405   0.7802   0.9112   0.6959   0.4418  
0.55       0.7633   0.5405   0.7802   0.9112   0.6959   0.4418  
0.60       0.7633   0.5405   0.7802   0.9112   0.6959   0.4418  
0.65       0.7633   0.5405   0.7802   0.9112   0.6959   0.4418  
0.70       0.7633   0.5405   0.7802   0.9112   0.6959   0.4418  
0.75       0.7633   0.5405   0.7802   0.9112   0.6959   0.4418  
0.80       0.7633   0.5405   0.7802   0.9112   0.6959   0.4418  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7633, F1=0.5405, Normal Recall=0.7802, Normal Precision=0.9112, Attack Recall=0.6959, Attack Precision=0.4418

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
0.15       0.7548   0.6300   0.7801   0.8568   0.6959   0.5756   <--
0.20       0.7548   0.6300   0.7801   0.8568   0.6959   0.5756  
0.25       0.7548   0.6300   0.7801   0.8568   0.6959   0.5756  
0.30       0.7548   0.6300   0.7801   0.8568   0.6959   0.5756  
0.35       0.7548   0.6300   0.7801   0.8568   0.6959   0.5756  
0.40       0.7548   0.6300   0.7801   0.8568   0.6959   0.5756  
0.45       0.7548   0.6300   0.7801   0.8568   0.6959   0.5756  
0.50       0.7548   0.6300   0.7801   0.8568   0.6959   0.5756  
0.55       0.7548   0.6300   0.7801   0.8568   0.6959   0.5756  
0.60       0.7548   0.6300   0.7801   0.8568   0.6959   0.5756  
0.65       0.7548   0.6300   0.7801   0.8568   0.6959   0.5756  
0.70       0.7548   0.6300   0.7801   0.8568   0.6959   0.5756  
0.75       0.7548   0.6300   0.7801   0.8568   0.6959   0.5756  
0.80       0.7548   0.6300   0.7801   0.8568   0.6959   0.5756  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7548, F1=0.6300, Normal Recall=0.7801, Normal Precision=0.8568, Attack Recall=0.6959, Attack Precision=0.5756

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
0.15       0.7453   0.6861   0.7783   0.7933   0.6959   0.6766   <--
0.20       0.7453   0.6861   0.7783   0.7933   0.6959   0.6766  
0.25       0.7453   0.6861   0.7783   0.7933   0.6959   0.6766  
0.30       0.7453   0.6861   0.7783   0.7933   0.6959   0.6766  
0.35       0.7453   0.6861   0.7783   0.7933   0.6959   0.6766  
0.40       0.7453   0.6861   0.7783   0.7933   0.6959   0.6766  
0.45       0.7453   0.6861   0.7783   0.7933   0.6959   0.6766  
0.50       0.7453   0.6861   0.7783   0.7933   0.6959   0.6766  
0.55       0.7453   0.6861   0.7783   0.7933   0.6959   0.6766  
0.60       0.7453   0.6861   0.7783   0.7933   0.6959   0.6766  
0.65       0.7453   0.6861   0.7783   0.7933   0.6959   0.6766  
0.70       0.7453   0.6861   0.7783   0.7933   0.6959   0.6766  
0.75       0.7453   0.6861   0.7783   0.7933   0.6959   0.6766  
0.80       0.7453   0.6861   0.7783   0.7933   0.6959   0.6766  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7453, F1=0.6861, Normal Recall=0.7783, Normal Precision=0.7933, Attack Recall=0.6959, Attack Precision=0.6766

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
0.15       0.7375   0.7261   0.7791   0.7193   0.6959   0.7591   <--
0.20       0.7375   0.7261   0.7791   0.7193   0.6959   0.7591  
0.25       0.7375   0.7261   0.7791   0.7193   0.6959   0.7591  
0.30       0.7375   0.7261   0.7791   0.7193   0.6959   0.7591  
0.35       0.7375   0.7261   0.7791   0.7193   0.6959   0.7591  
0.40       0.7375   0.7261   0.7791   0.7193   0.6959   0.7591  
0.45       0.7375   0.7261   0.7791   0.7193   0.6959   0.7591  
0.50       0.7375   0.7261   0.7791   0.7193   0.6959   0.7591  
0.55       0.7375   0.7261   0.7791   0.7193   0.6959   0.7591  
0.60       0.7375   0.7261   0.7791   0.7193   0.6959   0.7591  
0.65       0.7375   0.7261   0.7791   0.7193   0.6959   0.7591  
0.70       0.7375   0.7261   0.7791   0.7193   0.6959   0.7591  
0.75       0.7375   0.7261   0.7791   0.7193   0.6959   0.7591  
0.80       0.7375   0.7261   0.7791   0.7193   0.6959   0.7591  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7375, F1=0.7261, Normal Recall=0.7791, Normal Precision=0.7193, Attack Recall=0.6959, Attack Precision=0.7591

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
0.15       0.7042   0.4030   0.6715   0.9997   0.9983   0.2524   <--
0.20       0.7042   0.4030   0.6715   0.9997   0.9983   0.2524  
0.25       0.7042   0.4030   0.6715   0.9997   0.9983   0.2524  
0.30       0.7042   0.4030   0.6715   0.9997   0.9983   0.2524  
0.35       0.7042   0.4030   0.6715   0.9997   0.9983   0.2524  
0.40       0.7042   0.4030   0.6715   0.9997   0.9983   0.2524  
0.45       0.7042   0.4030   0.6715   0.9997   0.9983   0.2524  
0.50       0.7042   0.4030   0.6715   0.9997   0.9983   0.2524  
0.55       0.7042   0.4030   0.6715   0.9997   0.9983   0.2524  
0.60       0.7042   0.4030   0.6715   0.9997   0.9983   0.2524  
0.65       0.7042   0.4030   0.6715   0.9997   0.9983   0.2524  
0.70       0.7042   0.4030   0.6715   0.9997   0.9983   0.2524  
0.75       0.7042   0.4030   0.6715   0.9997   0.9983   0.2524  
0.80       0.7042   0.4030   0.6715   0.9997   0.9983   0.2524  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7042, F1=0.4030, Normal Recall=0.6715, Normal Precision=0.9997, Attack Recall=0.9983, Attack Precision=0.2524

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
0.15       0.7372   0.6031   0.6720   0.9993   0.9981   0.4321   <--
0.20       0.7372   0.6031   0.6720   0.9993   0.9981   0.4321  
0.25       0.7372   0.6031   0.6720   0.9993   0.9981   0.4321  
0.30       0.7372   0.6031   0.6720   0.9993   0.9981   0.4321  
0.35       0.7372   0.6031   0.6720   0.9993   0.9981   0.4321  
0.40       0.7372   0.6031   0.6720   0.9993   0.9981   0.4321  
0.45       0.7372   0.6031   0.6720   0.9993   0.9981   0.4321  
0.50       0.7372   0.6031   0.6720   0.9993   0.9981   0.4321  
0.55       0.7372   0.6031   0.6720   0.9993   0.9981   0.4321  
0.60       0.7372   0.6031   0.6720   0.9993   0.9981   0.4321  
0.65       0.7372   0.6031   0.6720   0.9993   0.9981   0.4321  
0.70       0.7372   0.6031   0.6720   0.9993   0.9981   0.4321  
0.75       0.7372   0.6031   0.6720   0.9993   0.9981   0.4321  
0.80       0.7372   0.6031   0.6720   0.9993   0.9981   0.4321  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7372, F1=0.6031, Normal Recall=0.6720, Normal Precision=0.9993, Attack Recall=0.9981, Attack Precision=0.4321

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
0.15       0.7693   0.7219   0.6712   0.9988   0.9981   0.5654   <--
0.20       0.7693   0.7219   0.6712   0.9988   0.9981   0.5654  
0.25       0.7693   0.7219   0.6712   0.9988   0.9981   0.5654  
0.30       0.7693   0.7219   0.6712   0.9988   0.9981   0.5654  
0.35       0.7693   0.7219   0.6712   0.9988   0.9981   0.5654  
0.40       0.7693   0.7219   0.6712   0.9988   0.9981   0.5654  
0.45       0.7693   0.7219   0.6712   0.9988   0.9981   0.5654  
0.50       0.7693   0.7219   0.6712   0.9988   0.9981   0.5654  
0.55       0.7693   0.7219   0.6712   0.9988   0.9981   0.5654  
0.60       0.7693   0.7219   0.6712   0.9988   0.9981   0.5654  
0.65       0.7693   0.7219   0.6712   0.9988   0.9981   0.5654  
0.70       0.7693   0.7219   0.6712   0.9988   0.9981   0.5654  
0.75       0.7693   0.7219   0.6712   0.9988   0.9981   0.5654  
0.80       0.7693   0.7219   0.6712   0.9988   0.9981   0.5654  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7693, F1=0.7219, Normal Recall=0.6712, Normal Precision=0.9988, Attack Recall=0.9981, Attack Precision=0.5654

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
0.15       0.8016   0.8010   0.6706   0.9982   0.9981   0.6689   <--
0.20       0.8016   0.8010   0.6706   0.9982   0.9981   0.6689  
0.25       0.8016   0.8010   0.6706   0.9982   0.9981   0.6689  
0.30       0.8016   0.8010   0.6706   0.9982   0.9981   0.6689  
0.35       0.8016   0.8010   0.6706   0.9982   0.9981   0.6689  
0.40       0.8016   0.8010   0.6706   0.9982   0.9981   0.6689  
0.45       0.8016   0.8010   0.6706   0.9982   0.9981   0.6689  
0.50       0.8016   0.8010   0.6706   0.9982   0.9981   0.6689  
0.55       0.8016   0.8010   0.6706   0.9982   0.9981   0.6689  
0.60       0.8016   0.8010   0.6706   0.9982   0.9981   0.6689  
0.65       0.8016   0.8010   0.6706   0.9982   0.9981   0.6689  
0.70       0.8016   0.8010   0.6706   0.9982   0.9981   0.6689  
0.75       0.8016   0.8010   0.6706   0.9982   0.9981   0.6689  
0.80       0.8016   0.8010   0.6706   0.9982   0.9981   0.6689  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8016, F1=0.8010, Normal Recall=0.6706, Normal Precision=0.9982, Attack Recall=0.9981, Attack Precision=0.6689

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
0.15       0.8339   0.8574   0.6697   0.9972   0.9981   0.7514   <--
0.20       0.8339   0.8574   0.6697   0.9972   0.9981   0.7514  
0.25       0.8339   0.8574   0.6697   0.9972   0.9981   0.7514  
0.30       0.8339   0.8574   0.6697   0.9972   0.9981   0.7514  
0.35       0.8339   0.8574   0.6697   0.9972   0.9981   0.7514  
0.40       0.8339   0.8574   0.6697   0.9972   0.9981   0.7514  
0.45       0.8339   0.8574   0.6697   0.9972   0.9981   0.7514  
0.50       0.8339   0.8574   0.6697   0.9972   0.9981   0.7514  
0.55       0.8339   0.8574   0.6697   0.9972   0.9981   0.7514  
0.60       0.8339   0.8574   0.6697   0.9972   0.9981   0.7514  
0.65       0.8339   0.8574   0.6697   0.9972   0.9981   0.7514  
0.70       0.8339   0.8574   0.6697   0.9972   0.9981   0.7514  
0.75       0.8339   0.8574   0.6697   0.9972   0.9981   0.7514  
0.80       0.8339   0.8574   0.6697   0.9972   0.9981   0.7514  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8339, F1=0.8574, Normal Recall=0.6697, Normal Precision=0.9972, Attack Recall=0.9981, Attack Precision=0.7514

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
0.15       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517   <--
0.20       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.25       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.30       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.35       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.40       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.45       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.50       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.55       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.60       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.65       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.70       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.75       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.80       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7030, F1=0.4020, Normal Recall=0.6702, Normal Precision=0.9997, Attack Recall=0.9983, Attack Precision=0.2517

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
0.15       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312   <--
0.20       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.25       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.30       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.35       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.40       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.45       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.50       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.55       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.60       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.65       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.70       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.75       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.80       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7362, F1=0.6022, Normal Recall=0.6708, Normal Precision=0.9993, Attack Recall=0.9982, Attack Precision=0.4312

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
0.15       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645   <--
0.20       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.25       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.30       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.35       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.40       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.45       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.50       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.55       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.60       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.65       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.70       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.75       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.80       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7685, F1=0.7212, Normal Recall=0.6700, Normal Precision=0.9988, Attack Recall=0.9982, Attack Precision=0.5645

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
0.15       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681   <--
0.20       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.25       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.30       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.35       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.40       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.45       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.50       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.55       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.60       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.65       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.70       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.75       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.80       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8009, F1=0.8005, Normal Recall=0.6695, Normal Precision=0.9982, Attack Recall=0.9982, Attack Precision=0.6681

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
0.15       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507   <--
0.20       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.25       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.30       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.35       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.40       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.45       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.50       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.55       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.60       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.65       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.70       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.75       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.80       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8333, F1=0.8569, Normal Recall=0.6685, Normal Precision=0.9973, Attack Recall=0.9982, Attack Precision=0.7507

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
0.15       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517   <--
0.20       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.25       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.30       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.35       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.40       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.45       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.50       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.55       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.60       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.65       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.70       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.75       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
0.80       0.7030   0.4020   0.6702   0.9997   0.9983   0.2517  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7030, F1=0.4020, Normal Recall=0.6702, Normal Precision=0.9997, Attack Recall=0.9983, Attack Precision=0.2517

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
0.15       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312   <--
0.20       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.25       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.30       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.35       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.40       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.45       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.50       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.55       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.60       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.65       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.70       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.75       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
0.80       0.7362   0.6022   0.6708   0.9993   0.9982   0.4312  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7362, F1=0.6022, Normal Recall=0.6708, Normal Precision=0.9993, Attack Recall=0.9982, Attack Precision=0.4312

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
0.15       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645   <--
0.20       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.25       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.30       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.35       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.40       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.45       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.50       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.55       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.60       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.65       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.70       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.75       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
0.80       0.7685   0.7212   0.6700   0.9988   0.9982   0.5645  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7685, F1=0.7212, Normal Recall=0.6700, Normal Precision=0.9988, Attack Recall=0.9982, Attack Precision=0.5645

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
0.15       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681   <--
0.20       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.25       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.30       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.35       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.40       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.45       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.50       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.55       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.60       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.65       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.70       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.75       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
0.80       0.8009   0.8005   0.6695   0.9982   0.9982   0.6681  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8009, F1=0.8005, Normal Recall=0.6695, Normal Precision=0.9982, Attack Recall=0.9982, Attack Precision=0.6681

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
0.15       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507   <--
0.20       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.25       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.30       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.35       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.40       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.45       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.50       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.55       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.60       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.65       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.70       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.75       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
0.80       0.8333   0.8569   0.6685   0.9973   0.9982   0.7507  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8333, F1=0.8569, Normal Recall=0.6685, Normal Precision=0.9973, Attack Recall=0.9982, Attack Precision=0.7507

```

