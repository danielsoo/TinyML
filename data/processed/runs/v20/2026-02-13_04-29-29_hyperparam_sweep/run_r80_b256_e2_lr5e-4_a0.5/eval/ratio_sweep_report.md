# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e2_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-18 22:35:08 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9194 | 0.8765 | 0.8343 | 0.7911 | 0.7481 | 0.7071 | 0.6639 | 0.6218 | 0.5789 | 0.5370 | 0.4939 |
| QAT+Prune only | 0.6891 | 0.7195 | 0.7497 | 0.7801 | 0.8115 | 0.8405 | 0.8721 | 0.9029 | 0.9348 | 0.9646 | 0.9957 |
| QAT+PTQ | 0.6886 | 0.7191 | 0.7493 | 0.7799 | 0.8111 | 0.8402 | 0.8721 | 0.9029 | 0.9347 | 0.9645 | 0.9957 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6886 | 0.7191 | 0.7493 | 0.7799 | 0.8111 | 0.8402 | 0.8721 | 0.9029 | 0.9347 | 0.9645 | 0.9957 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.4436 | 0.5438 | 0.5865 | 0.6107 | 0.6278 | 0.6381 | 0.6464 | 0.6524 | 0.6576 | 0.6613 |
| QAT+Prune only | 0.0000 | 0.4152 | 0.6141 | 0.7310 | 0.8087 | 0.8619 | 0.9033 | 0.9349 | 0.9607 | 0.9806 | 0.9979 |
| QAT+PTQ | 0.0000 | 0.4148 | 0.6137 | 0.7308 | 0.8083 | 0.8617 | 0.9033 | 0.9349 | 0.9606 | 0.9806 | 0.9979 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4148 | 0.6137 | 0.7308 | 0.8083 | 0.8617 | 0.9033 | 0.9349 | 0.9606 | 0.9806 | 0.9979 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9194 | 0.9192 | 0.9193 | 0.9184 | 0.9175 | 0.9203 | 0.9187 | 0.9200 | 0.9189 | 0.9243 | 0.0000 |
| QAT+Prune only | 0.6891 | 0.6888 | 0.6882 | 0.6877 | 0.6887 | 0.6853 | 0.6866 | 0.6864 | 0.6910 | 0.6846 | 0.0000 |
| QAT+PTQ | 0.6886 | 0.6883 | 0.6877 | 0.6874 | 0.6880 | 0.6846 | 0.6867 | 0.6862 | 0.6905 | 0.6835 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6886 | 0.6883 | 0.6877 | 0.6874 | 0.6880 | 0.6846 | 0.6867 | 0.6862 | 0.6905 | 0.6835 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9194 | 0.0000 | 0.0000 | 0.0000 | 0.9194 | 1.0000 |
| 90 | 10 | 299,940 | 0.8765 | 0.4037 | 0.4922 | 0.4436 | 0.9192 | 0.9422 |
| 80 | 20 | 291,350 | 0.8343 | 0.6049 | 0.4939 | 0.5438 | 0.9193 | 0.8790 |
| 70 | 30 | 194,230 | 0.7911 | 0.7219 | 0.4939 | 0.5865 | 0.9184 | 0.8090 |
| 60 | 40 | 145,675 | 0.7481 | 0.7997 | 0.4939 | 0.6107 | 0.9175 | 0.7312 |
| 50 | 50 | 116,540 | 0.7071 | 0.8610 | 0.4939 | 0.6278 | 0.9203 | 0.6452 |
| 40 | 60 | 97,115 | 0.6639 | 0.9011 | 0.4940 | 0.6381 | 0.9187 | 0.5476 |
| 30 | 70 | 83,240 | 0.6218 | 0.9351 | 0.4939 | 0.6464 | 0.9200 | 0.4379 |
| 20 | 80 | 72,835 | 0.5789 | 0.9606 | 0.4939 | 0.6524 | 0.9189 | 0.3122 |
| 10 | 90 | 64,740 | 0.5370 | 0.9833 | 0.4939 | 0.6576 | 0.9243 | 0.1687 |
| 0 | 100 | 58,270 | 0.4939 | 1.0000 | 0.4939 | 0.6613 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6891 | 0.0000 | 0.0000 | 0.0000 | 0.6891 | 1.0000 |
| 90 | 10 | 299,940 | 0.7195 | 0.2623 | 0.9958 | 0.4152 | 0.6888 | 0.9993 |
| 80 | 20 | 291,350 | 0.7497 | 0.4439 | 0.9957 | 0.6141 | 0.6882 | 0.9985 |
| 70 | 30 | 194,230 | 0.7801 | 0.5774 | 0.9957 | 0.7310 | 0.6877 | 0.9973 |
| 60 | 40 | 145,675 | 0.8115 | 0.6808 | 0.9957 | 0.8087 | 0.6887 | 0.9959 |
| 50 | 50 | 116,540 | 0.8405 | 0.7598 | 0.9957 | 0.8619 | 0.6853 | 0.9938 |
| 40 | 60 | 97,115 | 0.8721 | 0.8266 | 0.9957 | 0.9033 | 0.6866 | 0.9908 |
| 30 | 70 | 83,240 | 0.9029 | 0.8811 | 0.9957 | 0.9349 | 0.6864 | 0.9857 |
| 20 | 80 | 72,835 | 0.9348 | 0.9280 | 0.9957 | 0.9607 | 0.6910 | 0.9759 |
| 10 | 90 | 64,740 | 0.9646 | 0.9660 | 0.9957 | 0.9806 | 0.6846 | 0.9468 |
| 0 | 100 | 58,270 | 0.9957 | 1.0000 | 0.9957 | 0.9979 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6886 | 0.0000 | 0.0000 | 0.0000 | 0.6886 | 1.0000 |
| 90 | 10 | 299,940 | 0.7191 | 0.2620 | 0.9958 | 0.4148 | 0.6883 | 0.9993 |
| 80 | 20 | 291,350 | 0.7493 | 0.4436 | 0.9957 | 0.6137 | 0.6877 | 0.9984 |
| 70 | 30 | 194,230 | 0.7799 | 0.5772 | 0.9957 | 0.7308 | 0.6874 | 0.9973 |
| 60 | 40 | 145,675 | 0.8111 | 0.6803 | 0.9957 | 0.8083 | 0.6880 | 0.9959 |
| 50 | 50 | 116,540 | 0.8402 | 0.7594 | 0.9957 | 0.8617 | 0.6846 | 0.9938 |
| 40 | 60 | 97,115 | 0.8721 | 0.8266 | 0.9957 | 0.9033 | 0.6867 | 0.9908 |
| 30 | 70 | 83,240 | 0.9029 | 0.8810 | 0.9957 | 0.9349 | 0.6862 | 0.9857 |
| 20 | 80 | 72,835 | 0.9347 | 0.9279 | 0.9957 | 0.9606 | 0.6905 | 0.9758 |
| 10 | 90 | 64,740 | 0.9645 | 0.9659 | 0.9957 | 0.9806 | 0.6835 | 0.9467 |
| 0 | 100 | 58,270 | 0.9957 | 1.0000 | 0.9957 | 0.9979 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6886 | 0.0000 | 0.0000 | 0.0000 | 0.6886 | 1.0000 |
| 90 | 10 | 299,940 | 0.7191 | 0.2620 | 0.9958 | 0.4148 | 0.6883 | 0.9993 |
| 80 | 20 | 291,350 | 0.7493 | 0.4436 | 0.9957 | 0.6137 | 0.6877 | 0.9984 |
| 70 | 30 | 194,230 | 0.7799 | 0.5772 | 0.9957 | 0.7308 | 0.6874 | 0.9973 |
| 60 | 40 | 145,675 | 0.8111 | 0.6803 | 0.9957 | 0.8083 | 0.6880 | 0.9959 |
| 50 | 50 | 116,540 | 0.8402 | 0.7594 | 0.9957 | 0.8617 | 0.6846 | 0.9938 |
| 40 | 60 | 97,115 | 0.8721 | 0.8266 | 0.9957 | 0.9033 | 0.6867 | 0.9908 |
| 30 | 70 | 83,240 | 0.9029 | 0.8810 | 0.9957 | 0.9349 | 0.6862 | 0.9857 |
| 20 | 80 | 72,835 | 0.9347 | 0.9279 | 0.9957 | 0.9606 | 0.6905 | 0.9758 |
| 10 | 90 | 64,740 | 0.9645 | 0.9659 | 0.9957 | 0.9806 | 0.6835 | 0.9467 |
| 0 | 100 | 58,270 | 0.9957 | 1.0000 | 0.9957 | 0.9979 | 0.0000 | 0.0000 |


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
0.15       0.8767   0.4446   0.9192   0.9423   0.4937   0.4045   <--
0.20       0.8767   0.4446   0.9192   0.9423   0.4937   0.4045  
0.25       0.8767   0.4446   0.9192   0.9423   0.4937   0.4045  
0.30       0.8767   0.4446   0.9192   0.9423   0.4937   0.4045  
0.35       0.8767   0.4446   0.9192   0.9423   0.4937   0.4045  
0.40       0.8767   0.4446   0.9192   0.9423   0.4937   0.4045  
0.45       0.8767   0.4446   0.9192   0.9423   0.4937   0.4045  
0.50       0.8767   0.4446   0.9192   0.9423   0.4937   0.4045  
0.55       0.8767   0.4446   0.9192   0.9423   0.4937   0.4045  
0.60       0.8767   0.4446   0.9192   0.9423   0.4937   0.4045  
0.65       0.8767   0.4446   0.9192   0.9423   0.4937   0.4045  
0.70       0.8767   0.4446   0.9192   0.9423   0.4937   0.4045  
0.75       0.8767   0.4446   0.9192   0.9423   0.4937   0.4045  
0.80       0.8767   0.4446   0.9192   0.9423   0.4937   0.4045  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8767, F1=0.4446, Normal Recall=0.9192, Normal Precision=0.9423, Attack Recall=0.4937, Attack Precision=0.4045

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
0.15       0.8341   0.5436   0.9192   0.8790   0.4939   0.6044   <--
0.20       0.8341   0.5436   0.9192   0.8790   0.4939   0.6044  
0.25       0.8341   0.5436   0.9192   0.8790   0.4939   0.6044  
0.30       0.8341   0.5436   0.9192   0.8790   0.4939   0.6044  
0.35       0.8341   0.5436   0.9192   0.8790   0.4939   0.6044  
0.40       0.8341   0.5436   0.9192   0.8790   0.4939   0.6044  
0.45       0.8341   0.5436   0.9192   0.8790   0.4939   0.6044  
0.50       0.8341   0.5436   0.9192   0.8790   0.4939   0.6044  
0.55       0.8341   0.5436   0.9192   0.8790   0.4939   0.6044  
0.60       0.8341   0.5436   0.9192   0.8790   0.4939   0.6044  
0.65       0.8341   0.5436   0.9192   0.8790   0.4939   0.6044  
0.70       0.8341   0.5436   0.9192   0.8790   0.4939   0.6044  
0.75       0.8341   0.5436   0.9192   0.8790   0.4939   0.6044  
0.80       0.8341   0.5436   0.9192   0.8790   0.4939   0.6044  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8341, F1=0.5436, Normal Recall=0.9192, Normal Precision=0.8790, Attack Recall=0.4939, Attack Precision=0.6044

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
0.15       0.7917   0.5873   0.9193   0.8091   0.4940   0.7241   <--
0.20       0.7917   0.5873   0.9193   0.8091   0.4940   0.7241  
0.25       0.7917   0.5873   0.9193   0.8091   0.4940   0.7241  
0.30       0.7917   0.5873   0.9193   0.8091   0.4940   0.7241  
0.35       0.7917   0.5873   0.9193   0.8091   0.4940   0.7241  
0.40       0.7917   0.5873   0.9193   0.8091   0.4940   0.7241  
0.45       0.7917   0.5873   0.9193   0.8091   0.4940   0.7241  
0.50       0.7917   0.5873   0.9193   0.8091   0.4940   0.7241  
0.55       0.7917   0.5873   0.9193   0.8091   0.4940   0.7241  
0.60       0.7917   0.5873   0.9193   0.8091   0.4940   0.7241  
0.65       0.7917   0.5873   0.9193   0.8091   0.4940   0.7241  
0.70       0.7917   0.5873   0.9193   0.8091   0.4940   0.7241  
0.75       0.7917   0.5873   0.9193   0.8091   0.4940   0.7241  
0.80       0.7917   0.5873   0.9193   0.8091   0.4940   0.7241  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7917, F1=0.5873, Normal Recall=0.9193, Normal Precision=0.8091, Attack Recall=0.4940, Attack Precision=0.7241

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
0.15       0.7492   0.6118   0.9194   0.7316   0.4939   0.8035   <--
0.20       0.7492   0.6118   0.9194   0.7316   0.4939   0.8035  
0.25       0.7492   0.6118   0.9194   0.7316   0.4939   0.8035  
0.30       0.7492   0.6118   0.9194   0.7316   0.4939   0.8035  
0.35       0.7492   0.6118   0.9194   0.7316   0.4939   0.8035  
0.40       0.7492   0.6118   0.9194   0.7316   0.4939   0.8035  
0.45       0.7492   0.6118   0.9194   0.7316   0.4939   0.8035  
0.50       0.7492   0.6118   0.9194   0.7316   0.4939   0.8035  
0.55       0.7492   0.6118   0.9194   0.7316   0.4939   0.8035  
0.60       0.7492   0.6118   0.9194   0.7316   0.4939   0.8035  
0.65       0.7492   0.6118   0.9194   0.7316   0.4939   0.8035  
0.70       0.7492   0.6118   0.9194   0.7316   0.4939   0.8035  
0.75       0.7492   0.6118   0.9194   0.7316   0.4939   0.8035  
0.80       0.7492   0.6118   0.9194   0.7316   0.4939   0.8035  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7492, F1=0.6118, Normal Recall=0.9194, Normal Precision=0.7316, Attack Recall=0.4939, Attack Precision=0.8035

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
0.15       0.7064   0.6272   0.9189   0.6449   0.4939   0.8590   <--
0.20       0.7064   0.6272   0.9189   0.6449   0.4939   0.8590  
0.25       0.7064   0.6272   0.9189   0.6449   0.4939   0.8590  
0.30       0.7064   0.6272   0.9189   0.6449   0.4939   0.8590  
0.35       0.7064   0.6272   0.9189   0.6449   0.4939   0.8590  
0.40       0.7064   0.6272   0.9189   0.6449   0.4939   0.8590  
0.45       0.7064   0.6272   0.9189   0.6449   0.4939   0.8590  
0.50       0.7064   0.6272   0.9189   0.6449   0.4939   0.8590  
0.55       0.7064   0.6272   0.9189   0.6449   0.4939   0.8590  
0.60       0.7064   0.6272   0.9189   0.6449   0.4939   0.8590  
0.65       0.7064   0.6272   0.9189   0.6449   0.4939   0.8590  
0.70       0.7064   0.6272   0.9189   0.6449   0.4939   0.8590  
0.75       0.7064   0.6272   0.9189   0.6449   0.4939   0.8590  
0.80       0.7064   0.6272   0.9189   0.6449   0.4939   0.8590  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7064, F1=0.6272, Normal Recall=0.9189, Normal Precision=0.6449, Attack Recall=0.4939, Attack Precision=0.8590

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
0.15       0.7196   0.4153   0.6888   0.9993   0.9960   0.2623   <--
0.20       0.7196   0.4153   0.6888   0.9993   0.9960   0.2623  
0.25       0.7196   0.4153   0.6888   0.9993   0.9960   0.2623  
0.30       0.7196   0.4153   0.6888   0.9993   0.9960   0.2623  
0.35       0.7196   0.4153   0.6888   0.9993   0.9960   0.2623  
0.40       0.7196   0.4153   0.6888   0.9993   0.9960   0.2623  
0.45       0.7196   0.4153   0.6888   0.9993   0.9960   0.2623  
0.50       0.7196   0.4153   0.6888   0.9993   0.9960   0.2623  
0.55       0.7196   0.4153   0.6888   0.9993   0.9960   0.2623  
0.60       0.7196   0.4153   0.6888   0.9993   0.9960   0.2623  
0.65       0.7196   0.4153   0.6888   0.9993   0.9960   0.2623  
0.70       0.7196   0.4153   0.6888   0.9993   0.9960   0.2623  
0.75       0.7196   0.4153   0.6888   0.9993   0.9960   0.2623  
0.80       0.7196   0.4153   0.6888   0.9993   0.9960   0.2623  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7196, F1=0.4153, Normal Recall=0.6888, Normal Precision=0.9993, Attack Recall=0.9960, Attack Precision=0.2623

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
0.15       0.7507   0.6150   0.6894   0.9985   0.9957   0.4449   <--
0.20       0.7507   0.6150   0.6894   0.9985   0.9957   0.4449  
0.25       0.7507   0.6150   0.6894   0.9985   0.9957   0.4449  
0.30       0.7507   0.6150   0.6894   0.9985   0.9957   0.4449  
0.35       0.7507   0.6150   0.6894   0.9985   0.9957   0.4449  
0.40       0.7507   0.6150   0.6894   0.9985   0.9957   0.4449  
0.45       0.7507   0.6150   0.6894   0.9985   0.9957   0.4449  
0.50       0.7507   0.6150   0.6894   0.9985   0.9957   0.4449  
0.55       0.7507   0.6150   0.6894   0.9985   0.9957   0.4449  
0.60       0.7507   0.6150   0.6894   0.9985   0.9957   0.4449  
0.65       0.7507   0.6150   0.6894   0.9985   0.9957   0.4449  
0.70       0.7507   0.6150   0.6894   0.9985   0.9957   0.4449  
0.75       0.7507   0.6150   0.6894   0.9985   0.9957   0.4449  
0.80       0.7507   0.6150   0.6894   0.9985   0.9957   0.4449  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7507, F1=0.6150, Normal Recall=0.6894, Normal Precision=0.9985, Attack Recall=0.9957, Attack Precision=0.4449

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
0.15       0.7812   0.7320   0.6893   0.9974   0.9957   0.5787   <--
0.20       0.7812   0.7320   0.6893   0.9974   0.9957   0.5787  
0.25       0.7812   0.7320   0.6893   0.9974   0.9957   0.5787  
0.30       0.7812   0.7320   0.6893   0.9974   0.9957   0.5787  
0.35       0.7812   0.7320   0.6893   0.9974   0.9957   0.5787  
0.40       0.7812   0.7320   0.6893   0.9974   0.9957   0.5787  
0.45       0.7812   0.7320   0.6893   0.9974   0.9957   0.5787  
0.50       0.7812   0.7320   0.6893   0.9974   0.9957   0.5787  
0.55       0.7812   0.7320   0.6893   0.9974   0.9957   0.5787  
0.60       0.7812   0.7320   0.6893   0.9974   0.9957   0.5787  
0.65       0.7812   0.7320   0.6893   0.9974   0.9957   0.5787  
0.70       0.7812   0.7320   0.6893   0.9974   0.9957   0.5787  
0.75       0.7812   0.7320   0.6893   0.9974   0.9957   0.5787  
0.80       0.7812   0.7320   0.6893   0.9974   0.9957   0.5787  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7812, F1=0.7320, Normal Recall=0.6893, Normal Precision=0.9974, Attack Recall=0.9957, Attack Precision=0.5787

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
0.15       0.8116   0.8087   0.6888   0.9959   0.9957   0.6809   <--
0.20       0.8116   0.8087   0.6888   0.9959   0.9957   0.6809  
0.25       0.8116   0.8087   0.6888   0.9959   0.9957   0.6809  
0.30       0.8116   0.8087   0.6888   0.9959   0.9957   0.6809  
0.35       0.8116   0.8087   0.6888   0.9959   0.9957   0.6809  
0.40       0.8116   0.8087   0.6888   0.9959   0.9957   0.6809  
0.45       0.8116   0.8087   0.6888   0.9959   0.9957   0.6809  
0.50       0.8116   0.8087   0.6888   0.9959   0.9957   0.6809  
0.55       0.8116   0.8087   0.6888   0.9959   0.9957   0.6809  
0.60       0.8116   0.8087   0.6888   0.9959   0.9957   0.6809  
0.65       0.8116   0.8087   0.6888   0.9959   0.9957   0.6809  
0.70       0.8116   0.8087   0.6888   0.9959   0.9957   0.6809  
0.75       0.8116   0.8087   0.6888   0.9959   0.9957   0.6809  
0.80       0.8116   0.8087   0.6888   0.9959   0.9957   0.6809  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8116, F1=0.8087, Normal Recall=0.6888, Normal Precision=0.9959, Attack Recall=0.9957, Attack Precision=0.6809

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
0.15       0.8414   0.8626   0.6870   0.9938   0.9957   0.7609   <--
0.20       0.8414   0.8626   0.6870   0.9938   0.9957   0.7609  
0.25       0.8414   0.8626   0.6870   0.9938   0.9957   0.7609  
0.30       0.8414   0.8626   0.6870   0.9938   0.9957   0.7609  
0.35       0.8414   0.8626   0.6870   0.9938   0.9957   0.7609  
0.40       0.8414   0.8626   0.6870   0.9938   0.9957   0.7609  
0.45       0.8414   0.8626   0.6870   0.9938   0.9957   0.7609  
0.50       0.8414   0.8626   0.6870   0.9938   0.9957   0.7609  
0.55       0.8414   0.8626   0.6870   0.9938   0.9957   0.7609  
0.60       0.8414   0.8626   0.6870   0.9938   0.9957   0.7609  
0.65       0.8414   0.8626   0.6870   0.9938   0.9957   0.7609  
0.70       0.8414   0.8626   0.6870   0.9938   0.9957   0.7609  
0.75       0.8414   0.8626   0.6870   0.9938   0.9957   0.7609  
0.80       0.8414   0.8626   0.6870   0.9938   0.9957   0.7609  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8414, F1=0.8626, Normal Recall=0.6870, Normal Precision=0.9938, Attack Recall=0.9957, Attack Precision=0.7609

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
0.15       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620   <--
0.20       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.25       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.30       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.35       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.40       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.45       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.50       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.55       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.60       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.65       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.70       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.75       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.80       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7191, F1=0.4149, Normal Recall=0.6883, Normal Precision=0.9993, Attack Recall=0.9960, Attack Precision=0.2620

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
0.15       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446   <--
0.20       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.25       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.30       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.35       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.40       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.45       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.50       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.55       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.60       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.65       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.70       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.75       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.80       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7503, F1=0.6147, Normal Recall=0.6890, Normal Precision=0.9985, Attack Recall=0.9957, Attack Precision=0.4446

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
0.15       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783   <--
0.20       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.25       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.30       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.35       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.40       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.45       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.50       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.55       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.60       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.65       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.70       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.75       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.80       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7809, F1=0.7317, Normal Recall=0.6889, Normal Precision=0.9973, Attack Recall=0.9957, Attack Precision=0.5783

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
0.15       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806   <--
0.20       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.25       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.30       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.35       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.40       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.45       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.50       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.55       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.60       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.65       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.70       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.75       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.80       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8114, F1=0.8085, Normal Recall=0.6885, Normal Precision=0.9959, Attack Recall=0.9957, Attack Precision=0.6806

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
0.15       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606   <--
0.20       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.25       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.30       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.35       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.40       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.45       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.50       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.55       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.60       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.65       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.70       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.75       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.80       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8412, F1=0.8624, Normal Recall=0.6866, Normal Precision=0.9938, Attack Recall=0.9957, Attack Precision=0.7606

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
0.15       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620   <--
0.20       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.25       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.30       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.35       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.40       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.45       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.50       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.55       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.60       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.65       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.70       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.75       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
0.80       0.7191   0.4149   0.6883   0.9993   0.9960   0.2620  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7191, F1=0.4149, Normal Recall=0.6883, Normal Precision=0.9993, Attack Recall=0.9960, Attack Precision=0.2620

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
0.15       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446   <--
0.20       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.25       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.30       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.35       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.40       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.45       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.50       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.55       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.60       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.65       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.70       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.75       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
0.80       0.7503   0.6147   0.6890   0.9985   0.9957   0.4446  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7503, F1=0.6147, Normal Recall=0.6890, Normal Precision=0.9985, Attack Recall=0.9957, Attack Precision=0.4446

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
0.15       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783   <--
0.20       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.25       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.30       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.35       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.40       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.45       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.50       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.55       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.60       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.65       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.70       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.75       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
0.80       0.7809   0.7317   0.6889   0.9973   0.9957   0.5783  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7809, F1=0.7317, Normal Recall=0.6889, Normal Precision=0.9973, Attack Recall=0.9957, Attack Precision=0.5783

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
0.15       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806   <--
0.20       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.25       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.30       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.35       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.40       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.45       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.50       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.55       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.60       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.65       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.70       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.75       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
0.80       0.8114   0.8085   0.6885   0.9959   0.9957   0.6806  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8114, F1=0.8085, Normal Recall=0.6885, Normal Precision=0.9959, Attack Recall=0.9957, Attack Precision=0.6806

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
0.15       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606   <--
0.20       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.25       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.30       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.35       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.40       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.45       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.50       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.55       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.60       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.65       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.70       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.75       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
0.80       0.8412   0.8624   0.6866   0.9938   0.9957   0.7606  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8412, F1=0.8624, Normal Recall=0.6866, Normal Precision=0.9938, Attack Recall=0.9957, Attack Precision=0.7606

```

