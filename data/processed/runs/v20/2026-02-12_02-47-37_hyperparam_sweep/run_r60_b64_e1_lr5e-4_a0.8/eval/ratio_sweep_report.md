# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-12 10:41:34 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7626 | 0.7375 | 0.7111 | 0.6853 | 0.6605 | 0.6348 | 0.6083 | 0.5836 | 0.5568 | 0.5300 | 0.5053 |
| QAT+Prune only | 0.7625 | 0.7858 | 0.8088 | 0.8334 | 0.8569 | 0.8786 | 0.9034 | 0.9268 | 0.9510 | 0.9744 | 0.9981 |
| QAT+PTQ | 0.7627 | 0.7861 | 0.8090 | 0.8336 | 0.8571 | 0.8787 | 0.9035 | 0.9268 | 0.9510 | 0.9743 | 0.9981 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7627 | 0.7861 | 0.8090 | 0.8336 | 0.8571 | 0.8787 | 0.9035 | 0.9268 | 0.9510 | 0.9743 | 0.9981 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2798 | 0.4116 | 0.4907 | 0.5435 | 0.5805 | 0.6075 | 0.6295 | 0.6459 | 0.6593 | 0.6713 |
| QAT+Prune only | 0.0000 | 0.4824 | 0.6762 | 0.7823 | 0.8480 | 0.8915 | 0.9254 | 0.9502 | 0.9702 | 0.9859 | 0.9990 |
| QAT+PTQ | 0.0000 | 0.4827 | 0.6764 | 0.7826 | 0.8482 | 0.8916 | 0.9254 | 0.9502 | 0.9703 | 0.9859 | 0.9990 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4827 | 0.6764 | 0.7826 | 0.8482 | 0.8916 | 0.9254 | 0.9502 | 0.9703 | 0.9859 | 0.9990 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7626 | 0.7628 | 0.7625 | 0.7624 | 0.7640 | 0.7644 | 0.7628 | 0.7665 | 0.7632 | 0.7529 | 0.0000 |
| QAT+Prune only | 0.7625 | 0.7623 | 0.7615 | 0.7628 | 0.7627 | 0.7590 | 0.7615 | 0.7605 | 0.7627 | 0.7609 | 0.0000 |
| QAT+PTQ | 0.7627 | 0.7625 | 0.7617 | 0.7631 | 0.7631 | 0.7592 | 0.7616 | 0.7605 | 0.7629 | 0.7604 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7627 | 0.7625 | 0.7617 | 0.7631 | 0.7631 | 0.7592 | 0.7616 | 0.7605 | 0.7629 | 0.7604 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7626 | 0.0000 | 0.0000 | 0.0000 | 0.7626 | 1.0000 |
| 90 | 10 | 299,940 | 0.7375 | 0.1928 | 0.5098 | 0.2798 | 0.7628 | 0.9334 |
| 80 | 20 | 291,350 | 0.7111 | 0.3472 | 0.5053 | 0.4116 | 0.7625 | 0.8604 |
| 70 | 30 | 194,230 | 0.6853 | 0.4769 | 0.5053 | 0.4907 | 0.7624 | 0.7824 |
| 60 | 40 | 145,675 | 0.6605 | 0.5881 | 0.5053 | 0.5435 | 0.7640 | 0.6985 |
| 50 | 50 | 116,540 | 0.6348 | 0.6820 | 0.5053 | 0.5805 | 0.7644 | 0.6071 |
| 40 | 60 | 97,115 | 0.6083 | 0.7616 | 0.5052 | 0.6075 | 0.7628 | 0.5069 |
| 30 | 70 | 83,240 | 0.5836 | 0.8347 | 0.5052 | 0.6295 | 0.7665 | 0.3990 |
| 20 | 80 | 72,835 | 0.5568 | 0.8951 | 0.5053 | 0.6459 | 0.7632 | 0.2783 |
| 10 | 90 | 64,740 | 0.5300 | 0.9485 | 0.5053 | 0.6593 | 0.7529 | 0.1446 |
| 0 | 100 | 58,270 | 0.5053 | 1.0000 | 0.5053 | 0.6713 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7625 | 0.0000 | 0.0000 | 0.0000 | 0.7625 | 1.0000 |
| 90 | 10 | 299,940 | 0.7858 | 0.3181 | 0.9981 | 0.4824 | 0.7623 | 0.9997 |
| 80 | 20 | 291,350 | 0.8088 | 0.5113 | 0.9981 | 0.6762 | 0.7615 | 0.9994 |
| 70 | 30 | 194,230 | 0.8334 | 0.6433 | 0.9981 | 0.7823 | 0.7628 | 0.9989 |
| 60 | 40 | 145,675 | 0.8569 | 0.7371 | 0.9981 | 0.8480 | 0.7627 | 0.9983 |
| 50 | 50 | 116,540 | 0.8786 | 0.8055 | 0.9981 | 0.8915 | 0.7590 | 0.9975 |
| 40 | 60 | 97,115 | 0.9034 | 0.8626 | 0.9981 | 0.9254 | 0.7615 | 0.9962 |
| 30 | 70 | 83,240 | 0.9268 | 0.9068 | 0.9981 | 0.9502 | 0.7605 | 0.9941 |
| 20 | 80 | 72,835 | 0.9510 | 0.9439 | 0.9981 | 0.9702 | 0.7627 | 0.9900 |
| 10 | 90 | 64,740 | 0.9744 | 0.9741 | 0.9981 | 0.9859 | 0.7609 | 0.9778 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9990 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7627 | 0.0000 | 0.0000 | 0.0000 | 0.7627 | 1.0000 |
| 90 | 10 | 299,940 | 0.7861 | 0.3183 | 0.9981 | 0.4827 | 0.7625 | 0.9997 |
| 80 | 20 | 291,350 | 0.8090 | 0.5115 | 0.9981 | 0.6764 | 0.7617 | 0.9994 |
| 70 | 30 | 194,230 | 0.8336 | 0.6436 | 0.9981 | 0.7826 | 0.7631 | 0.9989 |
| 60 | 40 | 145,675 | 0.8571 | 0.7375 | 0.9981 | 0.8482 | 0.7631 | 0.9983 |
| 50 | 50 | 116,540 | 0.8787 | 0.8056 | 0.9981 | 0.8916 | 0.7592 | 0.9975 |
| 40 | 60 | 97,115 | 0.9035 | 0.8626 | 0.9981 | 0.9254 | 0.7616 | 0.9962 |
| 30 | 70 | 83,240 | 0.9268 | 0.9068 | 0.9981 | 0.9502 | 0.7605 | 0.9941 |
| 20 | 80 | 72,835 | 0.9510 | 0.9439 | 0.9981 | 0.9703 | 0.7629 | 0.9900 |
| 10 | 90 | 64,740 | 0.9743 | 0.9740 | 0.9981 | 0.9859 | 0.7604 | 0.9778 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9990 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7627 | 0.0000 | 0.0000 | 0.0000 | 0.7627 | 1.0000 |
| 90 | 10 | 299,940 | 0.7861 | 0.3183 | 0.9981 | 0.4827 | 0.7625 | 0.9997 |
| 80 | 20 | 291,350 | 0.8090 | 0.5115 | 0.9981 | 0.6764 | 0.7617 | 0.9994 |
| 70 | 30 | 194,230 | 0.8336 | 0.6436 | 0.9981 | 0.7826 | 0.7631 | 0.9989 |
| 60 | 40 | 145,675 | 0.8571 | 0.7375 | 0.9981 | 0.8482 | 0.7631 | 0.9983 |
| 50 | 50 | 116,540 | 0.8787 | 0.8056 | 0.9981 | 0.8916 | 0.7592 | 0.9975 |
| 40 | 60 | 97,115 | 0.9035 | 0.8626 | 0.9981 | 0.9254 | 0.7616 | 0.9962 |
| 30 | 70 | 83,240 | 0.9268 | 0.9068 | 0.9981 | 0.9502 | 0.7605 | 0.9941 |
| 20 | 80 | 72,835 | 0.9510 | 0.9439 | 0.9981 | 0.9703 | 0.7629 | 0.9900 |
| 10 | 90 | 64,740 | 0.9743 | 0.9740 | 0.9981 | 0.9859 | 0.7604 | 0.9778 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9990 | 0.0000 | 0.0000 |


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
0.15       0.7373   0.2788   0.7628   0.9331   0.5078   0.1922   <--
0.20       0.7373   0.2788   0.7628   0.9331   0.5078   0.1922  
0.25       0.7373   0.2788   0.7628   0.9331   0.5078   0.1922  
0.30       0.7373   0.2788   0.7628   0.9331   0.5078   0.1922  
0.35       0.7373   0.2788   0.7628   0.9331   0.5078   0.1922  
0.40       0.7373   0.2788   0.7628   0.9331   0.5078   0.1922  
0.45       0.7373   0.2788   0.7628   0.9331   0.5078   0.1922  
0.50       0.7373   0.2788   0.7628   0.9331   0.5078   0.1922  
0.55       0.7373   0.2788   0.7628   0.9331   0.5078   0.1922  
0.60       0.7373   0.2788   0.7628   0.9331   0.5078   0.1922  
0.65       0.7373   0.2788   0.7628   0.9331   0.5078   0.1922  
0.70       0.7373   0.2788   0.7628   0.9331   0.5078   0.1922  
0.75       0.7373   0.2788   0.7628   0.9331   0.5078   0.1922  
0.80       0.7373   0.2788   0.7628   0.9331   0.5078   0.1922  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7373, F1=0.2788, Normal Recall=0.7628, Normal Precision=0.9331, Attack Recall=0.5078, Attack Precision=0.1922

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
0.15       0.7114   0.4119   0.7629   0.8605   0.5053   0.3476   <--
0.20       0.7114   0.4119   0.7629   0.8605   0.5053   0.3476  
0.25       0.7114   0.4119   0.7629   0.8605   0.5053   0.3476  
0.30       0.7114   0.4119   0.7629   0.8605   0.5053   0.3476  
0.35       0.7114   0.4119   0.7629   0.8605   0.5053   0.3476  
0.40       0.7114   0.4119   0.7629   0.8605   0.5053   0.3476  
0.45       0.7114   0.4119   0.7629   0.8605   0.5053   0.3476  
0.50       0.7114   0.4119   0.7629   0.8605   0.5053   0.3476  
0.55       0.7114   0.4119   0.7629   0.8605   0.5053   0.3476  
0.60       0.7114   0.4119   0.7629   0.8605   0.5053   0.3476  
0.65       0.7114   0.4119   0.7629   0.8605   0.5053   0.3476  
0.70       0.7114   0.4119   0.7629   0.8605   0.5053   0.3476  
0.75       0.7114   0.4119   0.7629   0.8605   0.5053   0.3476  
0.80       0.7114   0.4119   0.7629   0.8605   0.5053   0.3476  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7114, F1=0.4119, Normal Recall=0.7629, Normal Precision=0.8605, Attack Recall=0.5053, Attack Precision=0.3476

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
0.15       0.6853   0.4907   0.7625   0.7824   0.5052   0.4769   <--
0.20       0.6853   0.4907   0.7625   0.7824   0.5052   0.4769  
0.25       0.6853   0.4907   0.7625   0.7824   0.5052   0.4769  
0.30       0.6853   0.4907   0.7625   0.7824   0.5052   0.4769  
0.35       0.6853   0.4907   0.7625   0.7824   0.5052   0.4769  
0.40       0.6853   0.4907   0.7625   0.7824   0.5052   0.4769  
0.45       0.6853   0.4907   0.7625   0.7824   0.5052   0.4769  
0.50       0.6853   0.4907   0.7625   0.7824   0.5052   0.4769  
0.55       0.6853   0.4907   0.7625   0.7824   0.5052   0.4769  
0.60       0.6853   0.4907   0.7625   0.7824   0.5052   0.4769  
0.65       0.6853   0.4907   0.7625   0.7824   0.5052   0.4769  
0.70       0.6853   0.4907   0.7625   0.7824   0.5052   0.4769  
0.75       0.6853   0.4907   0.7625   0.7824   0.5052   0.4769  
0.80       0.6853   0.4907   0.7625   0.7824   0.5052   0.4769  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6853, F1=0.4907, Normal Recall=0.7625, Normal Precision=0.7824, Attack Recall=0.5052, Attack Precision=0.4769

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
0.15       0.6599   0.5430   0.7629   0.6982   0.5053   0.5869   <--
0.20       0.6599   0.5430   0.7629   0.6982   0.5053   0.5869  
0.25       0.6599   0.5430   0.7629   0.6982   0.5053   0.5869  
0.30       0.6599   0.5430   0.7629   0.6982   0.5053   0.5869  
0.35       0.6599   0.5430   0.7629   0.6982   0.5053   0.5869  
0.40       0.6599   0.5430   0.7629   0.6982   0.5053   0.5869  
0.45       0.6599   0.5430   0.7629   0.6982   0.5053   0.5869  
0.50       0.6599   0.5430   0.7629   0.6982   0.5053   0.5869  
0.55       0.6599   0.5430   0.7629   0.6982   0.5053   0.5869  
0.60       0.6599   0.5430   0.7629   0.6982   0.5053   0.5869  
0.65       0.6599   0.5430   0.7629   0.6982   0.5053   0.5869  
0.70       0.6599   0.5430   0.7629   0.6982   0.5053   0.5869  
0.75       0.6599   0.5430   0.7629   0.6982   0.5053   0.5869  
0.80       0.6599   0.5430   0.7629   0.6982   0.5053   0.5869  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6599, F1=0.5430, Normal Recall=0.7629, Normal Precision=0.6982, Attack Recall=0.5053, Attack Precision=0.5869

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
0.15       0.6346   0.5803   0.7639   0.6069   0.5053   0.6815   <--
0.20       0.6346   0.5803   0.7639   0.6069   0.5053   0.6815  
0.25       0.6346   0.5803   0.7639   0.6069   0.5053   0.6815  
0.30       0.6346   0.5803   0.7639   0.6069   0.5053   0.6815  
0.35       0.6346   0.5803   0.7639   0.6069   0.5053   0.6815  
0.40       0.6346   0.5803   0.7639   0.6069   0.5053   0.6815  
0.45       0.6346   0.5803   0.7639   0.6069   0.5053   0.6815  
0.50       0.6346   0.5803   0.7639   0.6069   0.5053   0.6815  
0.55       0.6346   0.5803   0.7639   0.6069   0.5053   0.6815  
0.60       0.6346   0.5803   0.7639   0.6069   0.5053   0.6815  
0.65       0.6346   0.5803   0.7639   0.6069   0.5053   0.6815  
0.70       0.6346   0.5803   0.7639   0.6069   0.5053   0.6815  
0.75       0.6346   0.5803   0.7639   0.6069   0.5053   0.6815  
0.80       0.6346   0.5803   0.7639   0.6069   0.5053   0.6815  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6346, F1=0.5803, Normal Recall=0.7639, Normal Precision=0.6069, Attack Recall=0.5053, Attack Precision=0.6815

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
0.15       0.7859   0.4825   0.7623   0.9997   0.9982   0.3181   <--
0.20       0.7859   0.4825   0.7623   0.9997   0.9982   0.3181  
0.25       0.7859   0.4825   0.7623   0.9997   0.9982   0.3181  
0.30       0.7859   0.4825   0.7623   0.9997   0.9982   0.3181  
0.35       0.7859   0.4825   0.7623   0.9997   0.9982   0.3181  
0.40       0.7859   0.4825   0.7623   0.9997   0.9982   0.3181  
0.45       0.7859   0.4825   0.7623   0.9997   0.9982   0.3181  
0.50       0.7859   0.4825   0.7623   0.9997   0.9982   0.3181  
0.55       0.7859   0.4825   0.7623   0.9997   0.9982   0.3181  
0.60       0.7859   0.4825   0.7623   0.9997   0.9982   0.3181  
0.65       0.7859   0.4825   0.7623   0.9997   0.9982   0.3181  
0.70       0.7859   0.4825   0.7623   0.9997   0.9982   0.3181  
0.75       0.7859   0.4825   0.7623   0.9997   0.9982   0.3181  
0.80       0.7859   0.4825   0.7623   0.9997   0.9982   0.3181  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7859, F1=0.4825, Normal Recall=0.7623, Normal Precision=0.9997, Attack Recall=0.9982, Attack Precision=0.3181

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
0.15       0.8099   0.6775   0.7629   0.9994   0.9981   0.5127   <--
0.20       0.8099   0.6775   0.7629   0.9994   0.9981   0.5127  
0.25       0.8099   0.6775   0.7629   0.9994   0.9981   0.5127  
0.30       0.8099   0.6775   0.7629   0.9994   0.9981   0.5127  
0.35       0.8099   0.6775   0.7629   0.9994   0.9981   0.5127  
0.40       0.8099   0.6775   0.7629   0.9994   0.9981   0.5127  
0.45       0.8099   0.6775   0.7629   0.9994   0.9981   0.5127  
0.50       0.8099   0.6775   0.7629   0.9994   0.9981   0.5127  
0.55       0.8099   0.6775   0.7629   0.9994   0.9981   0.5127  
0.60       0.8099   0.6775   0.7629   0.9994   0.9981   0.5127  
0.65       0.8099   0.6775   0.7629   0.9994   0.9981   0.5127  
0.70       0.8099   0.6775   0.7629   0.9994   0.9981   0.5127  
0.75       0.8099   0.6775   0.7629   0.9994   0.9981   0.5127  
0.80       0.8099   0.6775   0.7629   0.9994   0.9981   0.5127  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8099, F1=0.6775, Normal Recall=0.7629, Normal Precision=0.9994, Attack Recall=0.9981, Attack Precision=0.5127

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
0.15       0.8334   0.7824   0.7629   0.9989   0.9981   0.6433   <--
0.20       0.8334   0.7824   0.7629   0.9989   0.9981   0.6433  
0.25       0.8334   0.7824   0.7629   0.9989   0.9981   0.6433  
0.30       0.8334   0.7824   0.7629   0.9989   0.9981   0.6433  
0.35       0.8334   0.7824   0.7629   0.9989   0.9981   0.6433  
0.40       0.8334   0.7824   0.7629   0.9989   0.9981   0.6433  
0.45       0.8334   0.7824   0.7629   0.9989   0.9981   0.6433  
0.50       0.8334   0.7824   0.7629   0.9989   0.9981   0.6433  
0.55       0.8334   0.7824   0.7629   0.9989   0.9981   0.6433  
0.60       0.8334   0.7824   0.7629   0.9989   0.9981   0.6433  
0.65       0.8334   0.7824   0.7629   0.9989   0.9981   0.6433  
0.70       0.8334   0.7824   0.7629   0.9989   0.9981   0.6433  
0.75       0.8334   0.7824   0.7629   0.9989   0.9981   0.6433  
0.80       0.8334   0.7824   0.7629   0.9989   0.9981   0.6433  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8334, F1=0.7824, Normal Recall=0.7629, Normal Precision=0.9989, Attack Recall=0.9981, Attack Precision=0.6433

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
0.15       0.8568   0.8479   0.7626   0.9983   0.9981   0.7370   <--
0.20       0.8568   0.8479   0.7626   0.9983   0.9981   0.7370  
0.25       0.8568   0.8479   0.7626   0.9983   0.9981   0.7370  
0.30       0.8568   0.8479   0.7626   0.9983   0.9981   0.7370  
0.35       0.8568   0.8479   0.7626   0.9983   0.9981   0.7370  
0.40       0.8568   0.8479   0.7626   0.9983   0.9981   0.7370  
0.45       0.8568   0.8479   0.7626   0.9983   0.9981   0.7370  
0.50       0.8568   0.8479   0.7626   0.9983   0.9981   0.7370  
0.55       0.8568   0.8479   0.7626   0.9983   0.9981   0.7370  
0.60       0.8568   0.8479   0.7626   0.9983   0.9981   0.7370  
0.65       0.8568   0.8479   0.7626   0.9983   0.9981   0.7370  
0.70       0.8568   0.8479   0.7626   0.9983   0.9981   0.7370  
0.75       0.8568   0.8479   0.7626   0.9983   0.9981   0.7370  
0.80       0.8568   0.8479   0.7626   0.9983   0.9981   0.7370  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8568, F1=0.8479, Normal Recall=0.7626, Normal Precision=0.9983, Attack Recall=0.9981, Attack Precision=0.7370

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
0.15       0.8795   0.8922   0.7609   0.9975   0.9981   0.8067   <--
0.20       0.8795   0.8922   0.7609   0.9975   0.9981   0.8067  
0.25       0.8795   0.8922   0.7609   0.9975   0.9981   0.8067  
0.30       0.8795   0.8922   0.7609   0.9975   0.9981   0.8067  
0.35       0.8795   0.8922   0.7609   0.9975   0.9981   0.8067  
0.40       0.8795   0.8922   0.7609   0.9975   0.9981   0.8067  
0.45       0.8795   0.8922   0.7609   0.9975   0.9981   0.8067  
0.50       0.8795   0.8922   0.7609   0.9975   0.9981   0.8067  
0.55       0.8795   0.8922   0.7609   0.9975   0.9981   0.8067  
0.60       0.8795   0.8922   0.7609   0.9975   0.9981   0.8067  
0.65       0.8795   0.8922   0.7609   0.9975   0.9981   0.8067  
0.70       0.8795   0.8922   0.7609   0.9975   0.9981   0.8067  
0.75       0.8795   0.8922   0.7609   0.9975   0.9981   0.8067  
0.80       0.8795   0.8922   0.7609   0.9975   0.9981   0.8067  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8795, F1=0.8922, Normal Recall=0.7609, Normal Precision=0.9975, Attack Recall=0.9981, Attack Precision=0.8067

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
0.15       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183   <--
0.20       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.25       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.30       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.35       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.40       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.45       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.50       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.55       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.60       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.65       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.70       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.75       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.80       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7861, F1=0.4827, Normal Recall=0.7625, Normal Precision=0.9997, Attack Recall=0.9982, Attack Precision=0.3183

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
0.15       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130   <--
0.20       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.25       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.30       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.35       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.40       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.45       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.50       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.55       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.60       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.65       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.70       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.75       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.80       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8101, F1=0.6777, Normal Recall=0.7631, Normal Precision=0.9994, Attack Recall=0.9981, Attack Precision=0.5130

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
0.15       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435   <--
0.20       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.25       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.30       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.35       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.40       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.45       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.50       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.55       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.60       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.65       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.70       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.75       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.80       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8335, F1=0.7825, Normal Recall=0.7630, Normal Precision=0.9989, Attack Recall=0.9981, Attack Precision=0.6435

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
0.15       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372   <--
0.20       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.25       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.30       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.35       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.40       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.45       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.50       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.55       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.60       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.65       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.70       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.75       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.80       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8569, F1=0.8480, Normal Recall=0.7628, Normal Precision=0.9983, Attack Recall=0.9981, Attack Precision=0.7372

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
0.15       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069   <--
0.20       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.25       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.30       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.35       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.40       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.45       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.50       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.55       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.60       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.65       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.70       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.75       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.80       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8796, F1=0.8924, Normal Recall=0.7612, Normal Precision=0.9975, Attack Recall=0.9981, Attack Precision=0.8069

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
0.15       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183   <--
0.20       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.25       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.30       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.35       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.40       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.45       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.50       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.55       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.60       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.65       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.70       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.75       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
0.80       0.7861   0.4827   0.7625   0.9997   0.9982   0.3183  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7861, F1=0.4827, Normal Recall=0.7625, Normal Precision=0.9997, Attack Recall=0.9982, Attack Precision=0.3183

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
0.15       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130   <--
0.20       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.25       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.30       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.35       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.40       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.45       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.50       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.55       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.60       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.65       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.70       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.75       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
0.80       0.8101   0.6777   0.7631   0.9994   0.9981   0.5130  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8101, F1=0.6777, Normal Recall=0.7631, Normal Precision=0.9994, Attack Recall=0.9981, Attack Precision=0.5130

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
0.15       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435   <--
0.20       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.25       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.30       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.35       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.40       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.45       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.50       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.55       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.60       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.65       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.70       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.75       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
0.80       0.8335   0.7825   0.7630   0.9989   0.9981   0.6435  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8335, F1=0.7825, Normal Recall=0.7630, Normal Precision=0.9989, Attack Recall=0.9981, Attack Precision=0.6435

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
0.15       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372   <--
0.20       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.25       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.30       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.35       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.40       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.45       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.50       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.55       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.60       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.65       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.70       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.75       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
0.80       0.8569   0.8480   0.7628   0.9983   0.9981   0.7372  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8569, F1=0.8480, Normal Recall=0.7628, Normal Precision=0.9983, Attack Recall=0.9981, Attack Precision=0.7372

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
0.15       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069   <--
0.20       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.25       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.30       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.35       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.40       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.45       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.50       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.55       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.60       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.65       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.70       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.75       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
0.80       0.8796   0.8924   0.7612   0.9975   0.9981   0.8069  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8796, F1=0.8924, Normal Recall=0.7612, Normal Precision=0.9975, Attack Recall=0.9981, Attack Precision=0.8069

```

