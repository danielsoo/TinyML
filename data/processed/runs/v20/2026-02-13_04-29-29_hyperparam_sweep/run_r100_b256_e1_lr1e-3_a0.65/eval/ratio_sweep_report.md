# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e1_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-22 09:37:04 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 1 |
| **Batch size** | 256 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3553 | 0.4197 | 0.4839 | 0.5491 | 0.6131 | 0.6753 | 0.7410 | 0.8046 | 0.8680 | 0.9340 | 0.9975 |
| QAT+Prune only | 0.5911 | 0.6321 | 0.6722 | 0.7134 | 0.7552 | 0.7947 | 0.8355 | 0.8766 | 0.9178 | 0.9578 | 0.9994 |
| QAT+PTQ | 0.5941 | 0.6349 | 0.6746 | 0.7156 | 0.7570 | 0.7962 | 0.8367 | 0.8774 | 0.9184 | 0.9581 | 0.9993 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.5941 | 0.6349 | 0.6746 | 0.7156 | 0.7570 | 0.7962 | 0.8367 | 0.8774 | 0.9184 | 0.9581 | 0.9993 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2559 | 0.4360 | 0.5703 | 0.6735 | 0.7544 | 0.8221 | 0.8772 | 0.9236 | 0.9646 | 0.9987 |
| QAT+Prune only | 0.0000 | 0.3521 | 0.5495 | 0.6766 | 0.7656 | 0.8296 | 0.8794 | 0.9189 | 0.9511 | 0.9771 | 0.9997 |
| QAT+PTQ | 0.0000 | 0.3537 | 0.5513 | 0.6783 | 0.7669 | 0.8306 | 0.8801 | 0.9194 | 0.9514 | 0.9772 | 0.9997 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3537 | 0.5513 | 0.6783 | 0.7669 | 0.8306 | 0.8801 | 0.9194 | 0.9514 | 0.9772 | 0.9997 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3553 | 0.3555 | 0.3555 | 0.3569 | 0.3568 | 0.3532 | 0.3562 | 0.3544 | 0.3500 | 0.3628 | 0.0000 |
| QAT+Prune only | 0.5911 | 0.5913 | 0.5905 | 0.5909 | 0.5924 | 0.5900 | 0.5898 | 0.5901 | 0.5915 | 0.5842 | 0.0000 |
| QAT+PTQ | 0.5941 | 0.5943 | 0.5935 | 0.5940 | 0.5954 | 0.5931 | 0.5927 | 0.5929 | 0.5946 | 0.5873 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.5941 | 0.5943 | 0.5935 | 0.5940 | 0.5954 | 0.5931 | 0.5927 | 0.5929 | 0.5946 | 0.5873 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3553 | 0.0000 | 0.0000 | 0.0000 | 0.3553 | 1.0000 |
| 90 | 10 | 299,940 | 0.4197 | 0.1467 | 0.9976 | 0.2559 | 0.3555 | 0.9992 |
| 80 | 20 | 291,350 | 0.4839 | 0.2790 | 0.9975 | 0.4360 | 0.3555 | 0.9982 |
| 70 | 30 | 194,230 | 0.5491 | 0.3993 | 0.9975 | 0.5703 | 0.3569 | 0.9970 |
| 60 | 40 | 145,675 | 0.6131 | 0.5083 | 0.9975 | 0.6735 | 0.3568 | 0.9953 |
| 50 | 50 | 116,540 | 0.6753 | 0.6066 | 0.9975 | 0.7544 | 0.3532 | 0.9929 |
| 40 | 60 | 97,115 | 0.7410 | 0.6992 | 0.9975 | 0.8221 | 0.3562 | 0.9895 |
| 30 | 70 | 83,240 | 0.8046 | 0.7829 | 0.9975 | 0.8772 | 0.3544 | 0.9837 |
| 20 | 80 | 72,835 | 0.8680 | 0.8599 | 0.9975 | 0.9236 | 0.3500 | 0.9720 |
| 10 | 90 | 64,740 | 0.9340 | 0.9337 | 0.9975 | 0.9646 | 0.3628 | 0.9411 |
| 0 | 100 | 58,270 | 0.9975 | 1.0000 | 0.9975 | 0.9987 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5911 | 0.0000 | 0.0000 | 0.0000 | 0.5911 | 1.0000 |
| 90 | 10 | 299,940 | 0.6321 | 0.2137 | 0.9994 | 0.3521 | 0.5913 | 0.9999 |
| 80 | 20 | 291,350 | 0.6722 | 0.3789 | 0.9994 | 0.5495 | 0.5905 | 0.9997 |
| 70 | 30 | 194,230 | 0.7134 | 0.5115 | 0.9994 | 0.6766 | 0.5909 | 0.9995 |
| 60 | 40 | 145,675 | 0.7552 | 0.6204 | 0.9994 | 0.7656 | 0.5924 | 0.9993 |
| 50 | 50 | 116,540 | 0.7947 | 0.7091 | 0.9994 | 0.8296 | 0.5900 | 0.9989 |
| 40 | 60 | 97,115 | 0.8355 | 0.7852 | 0.9994 | 0.8794 | 0.5898 | 0.9984 |
| 30 | 70 | 83,240 | 0.8766 | 0.8505 | 0.9994 | 0.9189 | 0.5901 | 0.9975 |
| 20 | 80 | 72,835 | 0.9178 | 0.9073 | 0.9994 | 0.9511 | 0.5915 | 0.9957 |
| 10 | 90 | 64,740 | 0.9578 | 0.9558 | 0.9994 | 0.9771 | 0.5842 | 0.9903 |
| 0 | 100 | 58,270 | 0.9994 | 1.0000 | 0.9994 | 0.9997 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5941 | 0.0000 | 0.0000 | 0.0000 | 0.5941 | 1.0000 |
| 90 | 10 | 299,940 | 0.6349 | 0.2149 | 0.9994 | 0.3537 | 0.5943 | 0.9999 |
| 80 | 20 | 291,350 | 0.6746 | 0.3806 | 0.9993 | 0.5513 | 0.5935 | 0.9997 |
| 70 | 30 | 194,230 | 0.7156 | 0.5134 | 0.9993 | 0.6783 | 0.5940 | 0.9995 |
| 60 | 40 | 145,675 | 0.7570 | 0.6222 | 0.9993 | 0.7669 | 0.5954 | 0.9992 |
| 50 | 50 | 116,540 | 0.7962 | 0.7107 | 0.9993 | 0.8306 | 0.5931 | 0.9988 |
| 40 | 60 | 97,115 | 0.8367 | 0.7863 | 0.9993 | 0.8801 | 0.5927 | 0.9983 |
| 30 | 70 | 83,240 | 0.8774 | 0.8514 | 0.9993 | 0.9194 | 0.5929 | 0.9973 |
| 20 | 80 | 72,835 | 0.9184 | 0.9079 | 0.9993 | 0.9514 | 0.5946 | 0.9954 |
| 10 | 90 | 64,740 | 0.9581 | 0.9561 | 0.9993 | 0.9772 | 0.5873 | 0.9896 |
| 0 | 100 | 58,270 | 0.9993 | 1.0000 | 0.9993 | 0.9997 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.5941 | 0.0000 | 0.0000 | 0.0000 | 0.5941 | 1.0000 |
| 90 | 10 | 299,940 | 0.6349 | 0.2149 | 0.9994 | 0.3537 | 0.5943 | 0.9999 |
| 80 | 20 | 291,350 | 0.6746 | 0.3806 | 0.9993 | 0.5513 | 0.5935 | 0.9997 |
| 70 | 30 | 194,230 | 0.7156 | 0.5134 | 0.9993 | 0.6783 | 0.5940 | 0.9995 |
| 60 | 40 | 145,675 | 0.7570 | 0.6222 | 0.9993 | 0.7669 | 0.5954 | 0.9992 |
| 50 | 50 | 116,540 | 0.7962 | 0.7107 | 0.9993 | 0.8306 | 0.5931 | 0.9988 |
| 40 | 60 | 97,115 | 0.8367 | 0.7863 | 0.9993 | 0.8801 | 0.5927 | 0.9983 |
| 30 | 70 | 83,240 | 0.8774 | 0.8514 | 0.9993 | 0.9194 | 0.5929 | 0.9973 |
| 20 | 80 | 72,835 | 0.9184 | 0.9079 | 0.9993 | 0.9514 | 0.5946 | 0.9954 |
| 10 | 90 | 64,740 | 0.9581 | 0.9561 | 0.9993 | 0.9772 | 0.5873 | 0.9896 |
| 0 | 100 | 58,270 | 0.9993 | 1.0000 | 0.9993 | 0.9997 | 0.0000 | 0.0000 |


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
0.15       0.4197   0.2559   0.3555   0.9992   0.9975   0.1467   <--
0.20       0.4197   0.2559   0.3555   0.9992   0.9975   0.1467  
0.25       0.4197   0.2559   0.3555   0.9992   0.9975   0.1467  
0.30       0.4197   0.2559   0.3555   0.9992   0.9975   0.1467  
0.35       0.4197   0.2559   0.3555   0.9992   0.9975   0.1467  
0.40       0.4197   0.2559   0.3555   0.9992   0.9975   0.1467  
0.45       0.4197   0.2559   0.3555   0.9992   0.9975   0.1467  
0.50       0.4197   0.2559   0.3555   0.9992   0.9975   0.1467  
0.55       0.4197   0.2559   0.3555   0.9992   0.9975   0.1467  
0.60       0.4197   0.2559   0.3555   0.9992   0.9975   0.1467  
0.65       0.4197   0.2559   0.3555   0.9992   0.9975   0.1467  
0.70       0.4197   0.2559   0.3555   0.9992   0.9975   0.1467  
0.75       0.4197   0.2559   0.3555   0.9992   0.9975   0.1467  
0.80       0.4197   0.2559   0.3555   0.9992   0.9975   0.1467  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4197, F1=0.2559, Normal Recall=0.3555, Normal Precision=0.9992, Attack Recall=0.9975, Attack Precision=0.1467

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
0.15       0.4838   0.4360   0.3554   0.9982   0.9975   0.2790   <--
0.20       0.4838   0.4360   0.3554   0.9982   0.9975   0.2790  
0.25       0.4838   0.4360   0.3554   0.9982   0.9975   0.2790  
0.30       0.4838   0.4360   0.3554   0.9982   0.9975   0.2790  
0.35       0.4838   0.4360   0.3554   0.9982   0.9975   0.2790  
0.40       0.4838   0.4360   0.3554   0.9982   0.9975   0.2790  
0.45       0.4838   0.4360   0.3554   0.9982   0.9975   0.2790  
0.50       0.4838   0.4360   0.3554   0.9982   0.9975   0.2790  
0.55       0.4838   0.4360   0.3554   0.9982   0.9975   0.2790  
0.60       0.4838   0.4360   0.3554   0.9982   0.9975   0.2790  
0.65       0.4838   0.4360   0.3554   0.9982   0.9975   0.2790  
0.70       0.4838   0.4360   0.3554   0.9982   0.9975   0.2790  
0.75       0.4838   0.4360   0.3554   0.9982   0.9975   0.2790  
0.80       0.4838   0.4360   0.3554   0.9982   0.9975   0.2790  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4838, F1=0.4360, Normal Recall=0.3554, Normal Precision=0.9982, Attack Recall=0.9975, Attack Precision=0.2790

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
0.15       0.5480   0.5697   0.3554   0.9970   0.9975   0.3988   <--
0.20       0.5480   0.5697   0.3554   0.9970   0.9975   0.3988  
0.25       0.5480   0.5697   0.3554   0.9970   0.9975   0.3988  
0.30       0.5480   0.5697   0.3554   0.9970   0.9975   0.3988  
0.35       0.5480   0.5697   0.3554   0.9970   0.9975   0.3988  
0.40       0.5480   0.5697   0.3554   0.9970   0.9975   0.3988  
0.45       0.5480   0.5697   0.3554   0.9970   0.9975   0.3988  
0.50       0.5480   0.5697   0.3554   0.9970   0.9975   0.3988  
0.55       0.5480   0.5697   0.3554   0.9970   0.9975   0.3988  
0.60       0.5480   0.5697   0.3554   0.9970   0.9975   0.3988  
0.65       0.5480   0.5697   0.3554   0.9970   0.9975   0.3988  
0.70       0.5480   0.5697   0.3554   0.9970   0.9975   0.3988  
0.75       0.5480   0.5697   0.3554   0.9970   0.9975   0.3988  
0.80       0.5480   0.5697   0.3554   0.9970   0.9975   0.3988  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5480, F1=0.5697, Normal Recall=0.3554, Normal Precision=0.9970, Attack Recall=0.9975, Attack Precision=0.3988

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
0.15       0.6120   0.6728   0.3550   0.9953   0.9975   0.5076   <--
0.20       0.6120   0.6728   0.3550   0.9953   0.9975   0.5076  
0.25       0.6120   0.6728   0.3550   0.9953   0.9975   0.5076  
0.30       0.6120   0.6728   0.3550   0.9953   0.9975   0.5076  
0.35       0.6120   0.6728   0.3550   0.9953   0.9975   0.5076  
0.40       0.6120   0.6728   0.3550   0.9953   0.9975   0.5076  
0.45       0.6120   0.6728   0.3550   0.9953   0.9975   0.5076  
0.50       0.6120   0.6728   0.3550   0.9953   0.9975   0.5076  
0.55       0.6120   0.6728   0.3550   0.9953   0.9975   0.5076  
0.60       0.6120   0.6728   0.3550   0.9953   0.9975   0.5076  
0.65       0.6120   0.6728   0.3550   0.9953   0.9975   0.5076  
0.70       0.6120   0.6728   0.3550   0.9953   0.9975   0.5076  
0.75       0.6120   0.6728   0.3550   0.9953   0.9975   0.5076  
0.80       0.6120   0.6728   0.3550   0.9953   0.9975   0.5076  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6120, F1=0.6728, Normal Recall=0.3550, Normal Precision=0.9953, Attack Recall=0.9975, Attack Precision=0.5076

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
0.15       0.6755   0.7545   0.3535   0.9929   0.9975   0.6067   <--
0.20       0.6755   0.7545   0.3535   0.9929   0.9975   0.6067  
0.25       0.6755   0.7545   0.3535   0.9929   0.9975   0.6067  
0.30       0.6755   0.7545   0.3535   0.9929   0.9975   0.6067  
0.35       0.6755   0.7545   0.3535   0.9929   0.9975   0.6067  
0.40       0.6755   0.7545   0.3535   0.9929   0.9975   0.6067  
0.45       0.6755   0.7545   0.3535   0.9929   0.9975   0.6067  
0.50       0.6755   0.7545   0.3535   0.9929   0.9975   0.6067  
0.55       0.6755   0.7545   0.3535   0.9929   0.9975   0.6067  
0.60       0.6755   0.7545   0.3535   0.9929   0.9975   0.6067  
0.65       0.6755   0.7545   0.3535   0.9929   0.9975   0.6067  
0.70       0.6755   0.7545   0.3535   0.9929   0.9975   0.6067  
0.75       0.6755   0.7545   0.3535   0.9929   0.9975   0.6067  
0.80       0.6755   0.7545   0.3535   0.9929   0.9975   0.6067  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6755, F1=0.7545, Normal Recall=0.3535, Normal Precision=0.9929, Attack Recall=0.9975, Attack Precision=0.6067

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
0.15       0.6322   0.3521   0.5914   0.9999   0.9994   0.2137   <--
0.20       0.6322   0.3521   0.5914   0.9999   0.9994   0.2137  
0.25       0.6322   0.3521   0.5914   0.9999   0.9994   0.2137  
0.30       0.6322   0.3521   0.5914   0.9999   0.9994   0.2137  
0.35       0.6322   0.3521   0.5914   0.9999   0.9994   0.2137  
0.40       0.6322   0.3521   0.5914   0.9999   0.9994   0.2137  
0.45       0.6322   0.3521   0.5914   0.9999   0.9994   0.2137  
0.50       0.6322   0.3521   0.5914   0.9999   0.9994   0.2137  
0.55       0.6322   0.3521   0.5914   0.9999   0.9994   0.2137  
0.60       0.6322   0.3521   0.5914   0.9999   0.9994   0.2137  
0.65       0.6322   0.3521   0.5914   0.9999   0.9994   0.2137  
0.70       0.6322   0.3521   0.5914   0.9999   0.9994   0.2137  
0.75       0.6322   0.3521   0.5914   0.9999   0.9994   0.2137  
0.80       0.6322   0.3521   0.5914   0.9999   0.9994   0.2137  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6322, F1=0.3521, Normal Recall=0.5914, Normal Precision=0.9999, Attack Recall=0.9994, Attack Precision=0.2137

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
0.15       0.6730   0.5501   0.5914   0.9997   0.9994   0.3795   <--
0.20       0.6730   0.5501   0.5914   0.9997   0.9994   0.3795  
0.25       0.6730   0.5501   0.5914   0.9997   0.9994   0.3795  
0.30       0.6730   0.5501   0.5914   0.9997   0.9994   0.3795  
0.35       0.6730   0.5501   0.5914   0.9997   0.9994   0.3795  
0.40       0.6730   0.5501   0.5914   0.9997   0.9994   0.3795  
0.45       0.6730   0.5501   0.5914   0.9997   0.9994   0.3795  
0.50       0.6730   0.5501   0.5914   0.9997   0.9994   0.3795  
0.55       0.6730   0.5501   0.5914   0.9997   0.9994   0.3795  
0.60       0.6730   0.5501   0.5914   0.9997   0.9994   0.3795  
0.65       0.6730   0.5501   0.5914   0.9997   0.9994   0.3795  
0.70       0.6730   0.5501   0.5914   0.9997   0.9994   0.3795  
0.75       0.6730   0.5501   0.5914   0.9997   0.9994   0.3795  
0.80       0.6730   0.5501   0.5914   0.9997   0.9994   0.3795  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6730, F1=0.5501, Normal Recall=0.5914, Normal Precision=0.9997, Attack Recall=0.9994, Attack Precision=0.3795

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
0.15       0.7137   0.6769   0.5913   0.9995   0.9994   0.5117   <--
0.20       0.7137   0.6769   0.5913   0.9995   0.9994   0.5117  
0.25       0.7137   0.6769   0.5913   0.9995   0.9994   0.5117  
0.30       0.7137   0.6769   0.5913   0.9995   0.9994   0.5117  
0.35       0.7137   0.6769   0.5913   0.9995   0.9994   0.5117  
0.40       0.7137   0.6769   0.5913   0.9995   0.9994   0.5117  
0.45       0.7137   0.6769   0.5913   0.9995   0.9994   0.5117  
0.50       0.7137   0.6769   0.5913   0.9995   0.9994   0.5117  
0.55       0.7137   0.6769   0.5913   0.9995   0.9994   0.5117  
0.60       0.7137   0.6769   0.5913   0.9995   0.9994   0.5117  
0.65       0.7137   0.6769   0.5913   0.9995   0.9994   0.5117  
0.70       0.7137   0.6769   0.5913   0.9995   0.9994   0.5117  
0.75       0.7137   0.6769   0.5913   0.9995   0.9994   0.5117  
0.80       0.7137   0.6769   0.5913   0.9995   0.9994   0.5117  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7137, F1=0.6769, Normal Recall=0.5913, Normal Precision=0.9995, Attack Recall=0.9994, Attack Precision=0.5117

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
0.15       0.7546   0.7651   0.5914   0.9993   0.9994   0.6199   <--
0.20       0.7546   0.7651   0.5914   0.9993   0.9994   0.6199  
0.25       0.7546   0.7651   0.5914   0.9993   0.9994   0.6199  
0.30       0.7546   0.7651   0.5914   0.9993   0.9994   0.6199  
0.35       0.7546   0.7651   0.5914   0.9993   0.9994   0.6199  
0.40       0.7546   0.7651   0.5914   0.9993   0.9994   0.6199  
0.45       0.7546   0.7651   0.5914   0.9993   0.9994   0.6199  
0.50       0.7546   0.7651   0.5914   0.9993   0.9994   0.6199  
0.55       0.7546   0.7651   0.5914   0.9993   0.9994   0.6199  
0.60       0.7546   0.7651   0.5914   0.9993   0.9994   0.6199  
0.65       0.7546   0.7651   0.5914   0.9993   0.9994   0.6199  
0.70       0.7546   0.7651   0.5914   0.9993   0.9994   0.6199  
0.75       0.7546   0.7651   0.5914   0.9993   0.9994   0.6199  
0.80       0.7546   0.7651   0.5914   0.9993   0.9994   0.6199  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7546, F1=0.7651, Normal Recall=0.5914, Normal Precision=0.9993, Attack Recall=0.9994, Attack Precision=0.6199

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
0.15       0.7943   0.8293   0.5893   0.9989   0.9994   0.7087   <--
0.20       0.7943   0.8293   0.5893   0.9989   0.9994   0.7087  
0.25       0.7943   0.8293   0.5893   0.9989   0.9994   0.7087  
0.30       0.7943   0.8293   0.5893   0.9989   0.9994   0.7087  
0.35       0.7943   0.8293   0.5893   0.9989   0.9994   0.7087  
0.40       0.7943   0.8293   0.5893   0.9989   0.9994   0.7087  
0.45       0.7943   0.8293   0.5893   0.9989   0.9994   0.7087  
0.50       0.7943   0.8293   0.5893   0.9989   0.9994   0.7087  
0.55       0.7943   0.8293   0.5893   0.9989   0.9994   0.7087  
0.60       0.7943   0.8293   0.5893   0.9989   0.9994   0.7087  
0.65       0.7943   0.8293   0.5893   0.9989   0.9994   0.7087  
0.70       0.7943   0.8293   0.5893   0.9989   0.9994   0.7087  
0.75       0.7943   0.8293   0.5893   0.9989   0.9994   0.7087  
0.80       0.7943   0.8293   0.5893   0.9989   0.9994   0.7087  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7943, F1=0.8293, Normal Recall=0.5893, Normal Precision=0.9989, Attack Recall=0.9994, Attack Precision=0.7087

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
0.15       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149   <--
0.20       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.25       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.30       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.35       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.40       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.45       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.50       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.55       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.60       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.65       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.70       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.75       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.80       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6349, F1=0.3537, Normal Recall=0.5944, Normal Precision=0.9999, Attack Recall=0.9993, Attack Precision=0.2149

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
0.15       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812   <--
0.20       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.25       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.30       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.35       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.40       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.45       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.50       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.55       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.60       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.65       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.70       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.75       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.80       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6754, F1=0.5519, Normal Recall=0.5944, Normal Precision=0.9997, Attack Recall=0.9993, Attack Precision=0.3812

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
0.15       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135   <--
0.20       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.25       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.30       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.35       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.40       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.45       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.50       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.55       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.60       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.65       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.70       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.75       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.80       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7158, F1=0.6784, Normal Recall=0.5943, Normal Precision=0.9995, Attack Recall=0.9993, Attack Precision=0.5135

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
0.15       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217   <--
0.20       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.25       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.30       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.35       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.40       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.45       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.50       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.55       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.60       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.65       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.70       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.75       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.80       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7564, F1=0.7665, Normal Recall=0.5945, Normal Precision=0.9992, Attack Recall=0.9993, Attack Precision=0.6217

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
0.15       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104   <--
0.20       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.25       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.30       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.35       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.40       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.45       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.50       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.55       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.60       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.65       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.70       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.75       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.80       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7959, F1=0.8304, Normal Recall=0.5926, Normal Precision=0.9988, Attack Recall=0.9993, Attack Precision=0.7104

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
0.15       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149   <--
0.20       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.25       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.30       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.35       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.40       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.45       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.50       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.55       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.60       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.65       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.70       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.75       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
0.80       0.6349   0.3537   0.5944   0.9999   0.9993   0.2149  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6349, F1=0.3537, Normal Recall=0.5944, Normal Precision=0.9999, Attack Recall=0.9993, Attack Precision=0.2149

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
0.15       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812   <--
0.20       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.25       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.30       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.35       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.40       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.45       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.50       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.55       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.60       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.65       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.70       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.75       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
0.80       0.6754   0.5519   0.5944   0.9997   0.9993   0.3812  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6754, F1=0.5519, Normal Recall=0.5944, Normal Precision=0.9997, Attack Recall=0.9993, Attack Precision=0.3812

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
0.15       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135   <--
0.20       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.25       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.30       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.35       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.40       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.45       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.50       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.55       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.60       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.65       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.70       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.75       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
0.80       0.7158   0.6784   0.5943   0.9995   0.9993   0.5135  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7158, F1=0.6784, Normal Recall=0.5943, Normal Precision=0.9995, Attack Recall=0.9993, Attack Precision=0.5135

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
0.15       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217   <--
0.20       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.25       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.30       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.35       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.40       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.45       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.50       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.55       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.60       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.65       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.70       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.75       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
0.80       0.7564   0.7665   0.5945   0.9992   0.9993   0.6217  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7564, F1=0.7665, Normal Recall=0.5945, Normal Precision=0.9992, Attack Recall=0.9993, Attack Precision=0.6217

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
0.15       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104   <--
0.20       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.25       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.30       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.35       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.40       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.45       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.50       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.55       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.60       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.65       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.70       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.75       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
0.80       0.7959   0.8304   0.5926   0.9988   0.9993   0.7104  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7959, F1=0.8304, Normal Recall=0.5926, Normal Precision=0.9988, Attack Recall=0.9993, Attack Precision=0.7104

```

