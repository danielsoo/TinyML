# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e1_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-21 07:10:57 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5724 | 0.6115 | 0.6515 | 0.6915 | 0.7303 | 0.7702 | 0.8112 | 0.8512 | 0.8915 | 0.9317 | 0.9716 |
| QAT+Prune only | 0.6200 | 0.6571 | 0.6947 | 0.7322 | 0.7714 | 0.8074 | 0.8462 | 0.8829 | 0.9217 | 0.9591 | 0.9971 |
| QAT+PTQ | 0.6189 | 0.6562 | 0.6938 | 0.7315 | 0.7708 | 0.8072 | 0.8460 | 0.8828 | 0.9217 | 0.9591 | 0.9971 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6189 | 0.6562 | 0.6938 | 0.7315 | 0.7708 | 0.8072 | 0.8460 | 0.8828 | 0.9217 | 0.9591 | 0.9971 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3334 | 0.5272 | 0.6540 | 0.7424 | 0.8087 | 0.8606 | 0.9014 | 0.9348 | 0.9624 | 0.9856 |
| QAT+Prune only | 0.0000 | 0.3678 | 0.5664 | 0.6908 | 0.7773 | 0.8381 | 0.8861 | 0.9226 | 0.9532 | 0.9777 | 0.9986 |
| QAT+PTQ | 0.0000 | 0.3672 | 0.5657 | 0.6902 | 0.7768 | 0.8380 | 0.8859 | 0.9225 | 0.9532 | 0.9777 | 0.9986 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3672 | 0.5657 | 0.6902 | 0.7768 | 0.8380 | 0.8859 | 0.9225 | 0.9532 | 0.9777 | 0.9986 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5724 | 0.5715 | 0.5715 | 0.5715 | 0.5694 | 0.5688 | 0.5706 | 0.5703 | 0.5711 | 0.5724 | 0.0000 |
| QAT+Prune only | 0.6200 | 0.6193 | 0.6191 | 0.6187 | 0.6210 | 0.6176 | 0.6198 | 0.6165 | 0.6202 | 0.6165 | 0.0000 |
| QAT+PTQ | 0.6189 | 0.6183 | 0.6180 | 0.6176 | 0.6200 | 0.6172 | 0.6192 | 0.6160 | 0.6198 | 0.6171 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6189 | 0.6183 | 0.6180 | 0.6176 | 0.6200 | 0.6172 | 0.6192 | 0.6160 | 0.6198 | 0.6171 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5724 | 0.0000 | 0.0000 | 0.0000 | 0.5724 | 1.0000 |
| 90 | 10 | 299,940 | 0.6115 | 0.2012 | 0.9714 | 0.3334 | 0.5715 | 0.9945 |
| 80 | 20 | 291,350 | 0.6515 | 0.3618 | 0.9716 | 0.5272 | 0.5715 | 0.9877 |
| 70 | 30 | 194,230 | 0.6915 | 0.4928 | 0.9716 | 0.6540 | 0.5715 | 0.9791 |
| 60 | 40 | 145,675 | 0.7303 | 0.6007 | 0.9716 | 0.7424 | 0.5694 | 0.9678 |
| 50 | 50 | 116,540 | 0.7702 | 0.6926 | 0.9716 | 0.8087 | 0.5688 | 0.9524 |
| 40 | 60 | 97,115 | 0.8112 | 0.7724 | 0.9716 | 0.8606 | 0.5706 | 0.9305 |
| 30 | 70 | 83,240 | 0.8512 | 0.8407 | 0.9716 | 0.9014 | 0.5703 | 0.8959 |
| 20 | 80 | 72,835 | 0.8915 | 0.9006 | 0.9716 | 0.9348 | 0.5711 | 0.8341 |
| 10 | 90 | 64,740 | 0.9317 | 0.9534 | 0.9716 | 0.9624 | 0.5724 | 0.6914 |
| 0 | 100 | 58,270 | 0.9716 | 1.0000 | 0.9716 | 0.9856 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6200 | 0.0000 | 0.0000 | 0.0000 | 0.6200 | 1.0000 |
| 90 | 10 | 299,940 | 0.6571 | 0.2255 | 0.9973 | 0.3678 | 0.6193 | 0.9995 |
| 80 | 20 | 291,350 | 0.6947 | 0.3955 | 0.9971 | 0.5664 | 0.6191 | 0.9988 |
| 70 | 30 | 194,230 | 0.7322 | 0.5285 | 0.9971 | 0.6908 | 0.6187 | 0.9980 |
| 60 | 40 | 145,675 | 0.7714 | 0.6369 | 0.9971 | 0.7773 | 0.6210 | 0.9969 |
| 50 | 50 | 116,540 | 0.8074 | 0.7228 | 0.9971 | 0.8381 | 0.6176 | 0.9954 |
| 40 | 60 | 97,115 | 0.8462 | 0.7973 | 0.9971 | 0.8861 | 0.6198 | 0.9931 |
| 30 | 70 | 83,240 | 0.8829 | 0.8585 | 0.9971 | 0.9226 | 0.6165 | 0.9892 |
| 20 | 80 | 72,835 | 0.9217 | 0.9131 | 0.9971 | 0.9532 | 0.6202 | 0.9817 |
| 10 | 90 | 64,740 | 0.9591 | 0.9590 | 0.9971 | 0.9777 | 0.6165 | 0.9596 |
| 0 | 100 | 58,270 | 0.9971 | 1.0000 | 0.9971 | 0.9986 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6189 | 0.0000 | 0.0000 | 0.0000 | 0.6189 | 1.0000 |
| 90 | 10 | 299,940 | 0.6562 | 0.2250 | 0.9973 | 0.3672 | 0.6183 | 0.9995 |
| 80 | 20 | 291,350 | 0.6938 | 0.3949 | 0.9971 | 0.5657 | 0.6180 | 0.9988 |
| 70 | 30 | 194,230 | 0.7315 | 0.5278 | 0.9971 | 0.6902 | 0.6176 | 0.9980 |
| 60 | 40 | 145,675 | 0.7708 | 0.6363 | 0.9971 | 0.7768 | 0.6200 | 0.9969 |
| 50 | 50 | 116,540 | 0.8072 | 0.7226 | 0.9971 | 0.8380 | 0.6172 | 0.9954 |
| 40 | 60 | 97,115 | 0.8460 | 0.7971 | 0.9971 | 0.8859 | 0.6192 | 0.9931 |
| 30 | 70 | 83,240 | 0.8828 | 0.8583 | 0.9971 | 0.9225 | 0.6160 | 0.9893 |
| 20 | 80 | 72,835 | 0.9217 | 0.9130 | 0.9971 | 0.9532 | 0.6198 | 0.9818 |
| 10 | 90 | 64,740 | 0.9591 | 0.9591 | 0.9971 | 0.9777 | 0.6171 | 0.9599 |
| 0 | 100 | 58,270 | 0.9971 | 1.0000 | 0.9971 | 0.9986 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6189 | 0.0000 | 0.0000 | 0.0000 | 0.6189 | 1.0000 |
| 90 | 10 | 299,940 | 0.6562 | 0.2250 | 0.9973 | 0.3672 | 0.6183 | 0.9995 |
| 80 | 20 | 291,350 | 0.6938 | 0.3949 | 0.9971 | 0.5657 | 0.6180 | 0.9988 |
| 70 | 30 | 194,230 | 0.7315 | 0.5278 | 0.9971 | 0.6902 | 0.6176 | 0.9980 |
| 60 | 40 | 145,675 | 0.7708 | 0.6363 | 0.9971 | 0.7768 | 0.6200 | 0.9969 |
| 50 | 50 | 116,540 | 0.8072 | 0.7226 | 0.9971 | 0.8380 | 0.6172 | 0.9954 |
| 40 | 60 | 97,115 | 0.8460 | 0.7971 | 0.9971 | 0.8859 | 0.6192 | 0.9931 |
| 30 | 70 | 83,240 | 0.8828 | 0.8583 | 0.9971 | 0.9225 | 0.6160 | 0.9893 |
| 20 | 80 | 72,835 | 0.9217 | 0.9130 | 0.9971 | 0.9532 | 0.6198 | 0.9818 |
| 10 | 90 | 64,740 | 0.9591 | 0.9591 | 0.9971 | 0.9777 | 0.6171 | 0.9599 |
| 0 | 100 | 58,270 | 0.9971 | 1.0000 | 0.9971 | 0.9986 | 0.0000 | 0.0000 |


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
0.15       0.6114   0.3332   0.5715   0.9943   0.9708   0.2011   <--
0.20       0.6114   0.3332   0.5715   0.9943   0.9708   0.2011  
0.25       0.6114   0.3332   0.5715   0.9943   0.9708   0.2011  
0.30       0.6114   0.3332   0.5715   0.9943   0.9708   0.2011  
0.35       0.6114   0.3332   0.5715   0.9943   0.9708   0.2011  
0.40       0.6114   0.3332   0.5715   0.9943   0.9708   0.2011  
0.45       0.6114   0.3332   0.5715   0.9943   0.9708   0.2011  
0.50       0.6114   0.3332   0.5715   0.9943   0.9708   0.2011  
0.55       0.6114   0.3332   0.5715   0.9943   0.9708   0.2011  
0.60       0.6114   0.3332   0.5715   0.9943   0.9708   0.2011  
0.65       0.6114   0.3332   0.5715   0.9943   0.9708   0.2011  
0.70       0.6114   0.3332   0.5715   0.9943   0.9708   0.2011  
0.75       0.6114   0.3332   0.5715   0.9943   0.9708   0.2011  
0.80       0.6114   0.3332   0.5715   0.9943   0.9708   0.2011  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6114, F1=0.3332, Normal Recall=0.5715, Normal Precision=0.9943, Attack Recall=0.9708, Attack Precision=0.2011

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
0.15       0.6517   0.5274   0.5717   0.9877   0.9716   0.3619   <--
0.20       0.6517   0.5274   0.5717   0.9877   0.9716   0.3619  
0.25       0.6517   0.5274   0.5717   0.9877   0.9716   0.3619  
0.30       0.6517   0.5274   0.5717   0.9877   0.9716   0.3619  
0.35       0.6517   0.5274   0.5717   0.9877   0.9716   0.3619  
0.40       0.6517   0.5274   0.5717   0.9877   0.9716   0.3619  
0.45       0.6517   0.5274   0.5717   0.9877   0.9716   0.3619  
0.50       0.6517   0.5274   0.5717   0.9877   0.9716   0.3619  
0.55       0.6517   0.5274   0.5717   0.9877   0.9716   0.3619  
0.60       0.6517   0.5274   0.5717   0.9877   0.9716   0.3619  
0.65       0.6517   0.5274   0.5717   0.9877   0.9716   0.3619  
0.70       0.6517   0.5274   0.5717   0.9877   0.9716   0.3619  
0.75       0.6517   0.5274   0.5717   0.9877   0.9716   0.3619  
0.80       0.6517   0.5274   0.5717   0.9877   0.9716   0.3619  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6517, F1=0.5274, Normal Recall=0.5717, Normal Precision=0.9877, Attack Recall=0.9716, Attack Precision=0.3619

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
0.15       0.6924   0.6546   0.5728   0.9792   0.9716   0.4936   <--
0.20       0.6924   0.6546   0.5728   0.9792   0.9716   0.4936  
0.25       0.6924   0.6546   0.5728   0.9792   0.9716   0.4936  
0.30       0.6924   0.6546   0.5728   0.9792   0.9716   0.4936  
0.35       0.6924   0.6546   0.5728   0.9792   0.9716   0.4936  
0.40       0.6924   0.6546   0.5728   0.9792   0.9716   0.4936  
0.45       0.6924   0.6546   0.5728   0.9792   0.9716   0.4936  
0.50       0.6924   0.6546   0.5728   0.9792   0.9716   0.4936  
0.55       0.6924   0.6546   0.5728   0.9792   0.9716   0.4936  
0.60       0.6924   0.6546   0.5728   0.9792   0.9716   0.4936  
0.65       0.6924   0.6546   0.5728   0.9792   0.9716   0.4936  
0.70       0.6924   0.6546   0.5728   0.9792   0.9716   0.4936  
0.75       0.6924   0.6546   0.5728   0.9792   0.9716   0.4936  
0.80       0.6924   0.6546   0.5728   0.9792   0.9716   0.4936  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6924, F1=0.6546, Normal Recall=0.5728, Normal Precision=0.9792, Attack Recall=0.9716, Attack Precision=0.4936

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
0.15       0.7316   0.7434   0.5717   0.9679   0.9716   0.6020   <--
0.20       0.7316   0.7434   0.5717   0.9679   0.9716   0.6020  
0.25       0.7316   0.7434   0.5717   0.9679   0.9716   0.6020  
0.30       0.7316   0.7434   0.5717   0.9679   0.9716   0.6020  
0.35       0.7316   0.7434   0.5717   0.9679   0.9716   0.6020  
0.40       0.7316   0.7434   0.5717   0.9679   0.9716   0.6020  
0.45       0.7316   0.7434   0.5717   0.9679   0.9716   0.6020  
0.50       0.7316   0.7434   0.5717   0.9679   0.9716   0.6020  
0.55       0.7316   0.7434   0.5717   0.9679   0.9716   0.6020  
0.60       0.7316   0.7434   0.5717   0.9679   0.9716   0.6020  
0.65       0.7316   0.7434   0.5717   0.9679   0.9716   0.6020  
0.70       0.7316   0.7434   0.5717   0.9679   0.9716   0.6020  
0.75       0.7316   0.7434   0.5717   0.9679   0.9716   0.6020  
0.80       0.7316   0.7434   0.5717   0.9679   0.9716   0.6020  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7316, F1=0.7434, Normal Recall=0.5717, Normal Precision=0.9679, Attack Recall=0.9716, Attack Precision=0.6020

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
0.15       0.7721   0.8100   0.5726   0.9527   0.9716   0.6945   <--
0.20       0.7721   0.8100   0.5726   0.9527   0.9716   0.6945  
0.25       0.7721   0.8100   0.5726   0.9527   0.9716   0.6945  
0.30       0.7721   0.8100   0.5726   0.9527   0.9716   0.6945  
0.35       0.7721   0.8100   0.5726   0.9527   0.9716   0.6945  
0.40       0.7721   0.8100   0.5726   0.9527   0.9716   0.6945  
0.45       0.7721   0.8100   0.5726   0.9527   0.9716   0.6945  
0.50       0.7721   0.8100   0.5726   0.9527   0.9716   0.6945  
0.55       0.7721   0.8100   0.5726   0.9527   0.9716   0.6945  
0.60       0.7721   0.8100   0.5726   0.9527   0.9716   0.6945  
0.65       0.7721   0.8100   0.5726   0.9527   0.9716   0.6945  
0.70       0.7721   0.8100   0.5726   0.9527   0.9716   0.6945  
0.75       0.7721   0.8100   0.5726   0.9527   0.9716   0.6945  
0.80       0.7721   0.8100   0.5726   0.9527   0.9716   0.6945  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7721, F1=0.8100, Normal Recall=0.5726, Normal Precision=0.9527, Attack Recall=0.9716, Attack Precision=0.6945

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
0.15       0.6571   0.3678   0.6193   0.9996   0.9975   0.2255   <--
0.20       0.6571   0.3678   0.6193   0.9996   0.9975   0.2255  
0.25       0.6571   0.3678   0.6193   0.9996   0.9975   0.2255  
0.30       0.6571   0.3678   0.6193   0.9996   0.9975   0.2255  
0.35       0.6571   0.3678   0.6193   0.9996   0.9975   0.2255  
0.40       0.6571   0.3678   0.6193   0.9996   0.9975   0.2255  
0.45       0.6571   0.3678   0.6193   0.9996   0.9975   0.2255  
0.50       0.6571   0.3678   0.6193   0.9996   0.9975   0.2255  
0.55       0.6571   0.3678   0.6193   0.9996   0.9975   0.2255  
0.60       0.6571   0.3678   0.6193   0.9996   0.9975   0.2255  
0.65       0.6571   0.3678   0.6193   0.9996   0.9975   0.2255  
0.70       0.6571   0.3678   0.6193   0.9996   0.9975   0.2255  
0.75       0.6571   0.3678   0.6193   0.9996   0.9975   0.2255  
0.80       0.6571   0.3678   0.6193   0.9996   0.9975   0.2255  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6571, F1=0.3678, Normal Recall=0.6193, Normal Precision=0.9996, Attack Recall=0.9975, Attack Precision=0.2255

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
0.15       0.6952   0.5668   0.6197   0.9988   0.9971   0.3960   <--
0.20       0.6952   0.5668   0.6197   0.9988   0.9971   0.3960  
0.25       0.6952   0.5668   0.6197   0.9988   0.9971   0.3960  
0.30       0.6952   0.5668   0.6197   0.9988   0.9971   0.3960  
0.35       0.6952   0.5668   0.6197   0.9988   0.9971   0.3960  
0.40       0.6952   0.5668   0.6197   0.9988   0.9971   0.3960  
0.45       0.6952   0.5668   0.6197   0.9988   0.9971   0.3960  
0.50       0.6952   0.5668   0.6197   0.9988   0.9971   0.3960  
0.55       0.6952   0.5668   0.6197   0.9988   0.9971   0.3960  
0.60       0.6952   0.5668   0.6197   0.9988   0.9971   0.3960  
0.65       0.6952   0.5668   0.6197   0.9988   0.9971   0.3960  
0.70       0.6952   0.5668   0.6197   0.9988   0.9971   0.3960  
0.75       0.6952   0.5668   0.6197   0.9988   0.9971   0.3960  
0.80       0.6952   0.5668   0.6197   0.9988   0.9971   0.3960  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6952, F1=0.5668, Normal Recall=0.6197, Normal Precision=0.9988, Attack Recall=0.9971, Attack Precision=0.3960

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
0.15       0.7332   0.6916   0.6202   0.9980   0.9971   0.5294   <--
0.20       0.7332   0.6916   0.6202   0.9980   0.9971   0.5294  
0.25       0.7332   0.6916   0.6202   0.9980   0.9971   0.5294  
0.30       0.7332   0.6916   0.6202   0.9980   0.9971   0.5294  
0.35       0.7332   0.6916   0.6202   0.9980   0.9971   0.5294  
0.40       0.7332   0.6916   0.6202   0.9980   0.9971   0.5294  
0.45       0.7332   0.6916   0.6202   0.9980   0.9971   0.5294  
0.50       0.7332   0.6916   0.6202   0.9980   0.9971   0.5294  
0.55       0.7332   0.6916   0.6202   0.9980   0.9971   0.5294  
0.60       0.7332   0.6916   0.6202   0.9980   0.9971   0.5294  
0.65       0.7332   0.6916   0.6202   0.9980   0.9971   0.5294  
0.70       0.7332   0.6916   0.6202   0.9980   0.9971   0.5294  
0.75       0.7332   0.6916   0.6202   0.9980   0.9971   0.5294  
0.80       0.7332   0.6916   0.6202   0.9980   0.9971   0.5294  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7332, F1=0.6916, Normal Recall=0.6202, Normal Precision=0.9980, Attack Recall=0.9971, Attack Precision=0.5294

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
0.15       0.7704   0.7765   0.6193   0.9969   0.9971   0.6359   <--
0.20       0.7704   0.7765   0.6193   0.9969   0.9971   0.6359  
0.25       0.7704   0.7765   0.6193   0.9969   0.9971   0.6359  
0.30       0.7704   0.7765   0.6193   0.9969   0.9971   0.6359  
0.35       0.7704   0.7765   0.6193   0.9969   0.9971   0.6359  
0.40       0.7704   0.7765   0.6193   0.9969   0.9971   0.6359  
0.45       0.7704   0.7765   0.6193   0.9969   0.9971   0.6359  
0.50       0.7704   0.7765   0.6193   0.9969   0.9971   0.6359  
0.55       0.7704   0.7765   0.6193   0.9969   0.9971   0.6359  
0.60       0.7704   0.7765   0.6193   0.9969   0.9971   0.6359  
0.65       0.7704   0.7765   0.6193   0.9969   0.9971   0.6359  
0.70       0.7704   0.7765   0.6193   0.9969   0.9971   0.6359  
0.75       0.7704   0.7765   0.6193   0.9969   0.9971   0.6359  
0.80       0.7704   0.7765   0.6193   0.9969   0.9971   0.6359  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7704, F1=0.7765, Normal Recall=0.6193, Normal Precision=0.9969, Attack Recall=0.9971, Attack Precision=0.6359

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
0.15       0.8076   0.8383   0.6181   0.9954   0.9971   0.7231   <--
0.20       0.8076   0.8383   0.6181   0.9954   0.9971   0.7231  
0.25       0.8076   0.8383   0.6181   0.9954   0.9971   0.7231  
0.30       0.8076   0.8383   0.6181   0.9954   0.9971   0.7231  
0.35       0.8076   0.8383   0.6181   0.9954   0.9971   0.7231  
0.40       0.8076   0.8383   0.6181   0.9954   0.9971   0.7231  
0.45       0.8076   0.8383   0.6181   0.9954   0.9971   0.7231  
0.50       0.8076   0.8383   0.6181   0.9954   0.9971   0.7231  
0.55       0.8076   0.8383   0.6181   0.9954   0.9971   0.7231  
0.60       0.8076   0.8383   0.6181   0.9954   0.9971   0.7231  
0.65       0.8076   0.8383   0.6181   0.9954   0.9971   0.7231  
0.70       0.8076   0.8383   0.6181   0.9954   0.9971   0.7231  
0.75       0.8076   0.8383   0.6181   0.9954   0.9971   0.7231  
0.80       0.8076   0.8383   0.6181   0.9954   0.9971   0.7231  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8076, F1=0.8383, Normal Recall=0.6181, Normal Precision=0.9954, Attack Recall=0.9971, Attack Precision=0.7231

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
0.15       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250   <--
0.20       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.25       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.30       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.35       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.40       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.45       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.50       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.55       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.60       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.65       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.70       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.75       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.80       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6562, F1=0.3672, Normal Recall=0.6183, Normal Precision=0.9996, Attack Recall=0.9975, Attack Precision=0.2250

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
0.15       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953   <--
0.20       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.25       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.30       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.35       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.40       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.45       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.50       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.55       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.60       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.65       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.70       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.75       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.80       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6944, F1=0.5662, Normal Recall=0.6187, Normal Precision=0.9988, Attack Recall=0.9971, Attack Precision=0.3953

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
0.15       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287   <--
0.20       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.25       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.30       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.35       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.40       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.45       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.50       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.55       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.60       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.65       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.70       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.75       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.80       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7325, F1=0.6910, Normal Recall=0.6191, Normal Precision=0.9980, Attack Recall=0.9971, Attack Precision=0.5287

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
0.15       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351   <--
0.20       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.25       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.30       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.35       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.40       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.45       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.50       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.55       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.60       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.65       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.70       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.75       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.80       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7697, F1=0.7760, Normal Recall=0.6181, Normal Precision=0.9969, Attack Recall=0.9971, Attack Precision=0.6351

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
0.15       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224   <--
0.20       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.25       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.30       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.35       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.40       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.45       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.50       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.55       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.60       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.65       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.70       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.75       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.80       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8070, F1=0.8378, Normal Recall=0.6169, Normal Precision=0.9954, Attack Recall=0.9971, Attack Precision=0.7224

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
0.15       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250   <--
0.20       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.25       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.30       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.35       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.40       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.45       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.50       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.55       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.60       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.65       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.70       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.75       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
0.80       0.6562   0.3672   0.6183   0.9996   0.9975   0.2250  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6562, F1=0.3672, Normal Recall=0.6183, Normal Precision=0.9996, Attack Recall=0.9975, Attack Precision=0.2250

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
0.15       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953   <--
0.20       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.25       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.30       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.35       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.40       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.45       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.50       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.55       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.60       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.65       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.70       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.75       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
0.80       0.6944   0.5662   0.6187   0.9988   0.9971   0.3953  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6944, F1=0.5662, Normal Recall=0.6187, Normal Precision=0.9988, Attack Recall=0.9971, Attack Precision=0.3953

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
0.15       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287   <--
0.20       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.25       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.30       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.35       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.40       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.45       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.50       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.55       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.60       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.65       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.70       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.75       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
0.80       0.7325   0.6910   0.6191   0.9980   0.9971   0.5287  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7325, F1=0.6910, Normal Recall=0.6191, Normal Precision=0.9980, Attack Recall=0.9971, Attack Precision=0.5287

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
0.15       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351   <--
0.20       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.25       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.30       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.35       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.40       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.45       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.50       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.55       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.60       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.65       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.70       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.75       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
0.80       0.7697   0.7760   0.6181   0.9969   0.9971   0.6351  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7697, F1=0.7760, Normal Recall=0.6181, Normal Precision=0.9969, Attack Recall=0.9971, Attack Precision=0.6351

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
0.15       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224   <--
0.20       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.25       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.30       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.35       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.40       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.45       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.50       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.55       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.60       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.65       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.70       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.75       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
0.80       0.8070   0.8378   0.6169   0.9954   0.9971   0.7224  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8070, F1=0.8378, Normal Recall=0.6169, Normal Precision=0.9954, Attack Recall=0.9971, Attack Precision=0.7224

```

