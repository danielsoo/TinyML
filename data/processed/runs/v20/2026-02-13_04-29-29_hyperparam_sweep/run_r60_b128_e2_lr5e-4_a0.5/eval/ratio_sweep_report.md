# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e2_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-14 21:27:02 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1441 | 0.2301 | 0.3157 | 0.4002 | 0.4864 | 0.5724 | 0.6574 | 0.7440 | 0.8269 | 0.9132 | 0.9993 |
| QAT+Prune only | 0.8918 | 0.8935 | 0.8949 | 0.8975 | 0.8986 | 0.8991 | 0.9018 | 0.9040 | 0.9048 | 0.9069 | 0.9087 |
| QAT+PTQ | 0.8902 | 0.8921 | 0.8934 | 0.8960 | 0.8971 | 0.8975 | 0.9003 | 0.9026 | 0.9034 | 0.9055 | 0.9073 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8902 | 0.8921 | 0.8934 | 0.8960 | 0.8971 | 0.8975 | 0.9003 | 0.9026 | 0.9034 | 0.9055 | 0.9073 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2061 | 0.3688 | 0.4999 | 0.6088 | 0.7003 | 0.7778 | 0.8453 | 0.9023 | 0.9540 | 0.9996 |
| QAT+Prune only | 0.0000 | 0.6304 | 0.7757 | 0.8417 | 0.8775 | 0.9000 | 0.9174 | 0.9298 | 0.9386 | 0.9461 | 0.9521 |
| QAT+PTQ | 0.0000 | 0.6271 | 0.7730 | 0.8396 | 0.8758 | 0.8985 | 0.9161 | 0.9288 | 0.9376 | 0.9453 | 0.9514 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.6271 | 0.7730 | 0.8396 | 0.8758 | 0.8985 | 0.9161 | 0.9288 | 0.9376 | 0.9453 | 0.9514 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1441 | 0.1446 | 0.1449 | 0.1434 | 0.1445 | 0.1455 | 0.1445 | 0.1484 | 0.1372 | 0.1384 | 0.0000 |
| QAT+Prune only | 0.8918 | 0.8918 | 0.8914 | 0.8927 | 0.8918 | 0.8894 | 0.8915 | 0.8931 | 0.8894 | 0.8908 | 0.0000 |
| QAT+PTQ | 0.8902 | 0.8904 | 0.8900 | 0.8912 | 0.8903 | 0.8878 | 0.8897 | 0.8916 | 0.8877 | 0.8894 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8902 | 0.8904 | 0.8900 | 0.8912 | 0.8903 | 0.8878 | 0.8897 | 0.8916 | 0.8877 | 0.8894 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1441 | 0.0000 | 0.0000 | 0.0000 | 0.1441 | 1.0000 |
| 90 | 10 | 299,940 | 0.2301 | 0.1149 | 0.9993 | 0.2061 | 0.1446 | 0.9995 |
| 80 | 20 | 291,350 | 0.3157 | 0.2261 | 0.9993 | 0.3688 | 0.1449 | 0.9988 |
| 70 | 30 | 194,230 | 0.4002 | 0.3333 | 0.9993 | 0.4999 | 0.1434 | 0.9979 |
| 60 | 40 | 145,675 | 0.4864 | 0.4378 | 0.9993 | 0.6088 | 0.1445 | 0.9968 |
| 50 | 50 | 116,540 | 0.5724 | 0.5390 | 0.9993 | 0.7003 | 0.1455 | 0.9952 |
| 40 | 60 | 97,115 | 0.6574 | 0.6366 | 0.9993 | 0.7778 | 0.1445 | 0.9927 |
| 30 | 70 | 83,240 | 0.7440 | 0.7325 | 0.9993 | 0.8453 | 0.1484 | 0.9891 |
| 20 | 80 | 72,835 | 0.8269 | 0.8225 | 0.9993 | 0.9023 | 0.1372 | 0.9799 |
| 10 | 90 | 64,740 | 0.9132 | 0.9126 | 0.9993 | 0.9540 | 0.1384 | 0.9562 |
| 0 | 100 | 58,270 | 0.9993 | 1.0000 | 0.9993 | 0.9996 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8918 | 0.0000 | 0.0000 | 0.0000 | 0.8918 | 1.0000 |
| 90 | 10 | 299,940 | 0.8935 | 0.4827 | 0.9084 | 0.6304 | 0.8918 | 0.9887 |
| 80 | 20 | 291,350 | 0.8949 | 0.6766 | 0.9087 | 0.7757 | 0.8914 | 0.9750 |
| 70 | 30 | 194,230 | 0.8975 | 0.7840 | 0.9087 | 0.8417 | 0.8927 | 0.9580 |
| 60 | 40 | 145,675 | 0.8986 | 0.8485 | 0.9087 | 0.8775 | 0.8918 | 0.9361 |
| 50 | 50 | 116,540 | 0.8991 | 0.8915 | 0.9087 | 0.9000 | 0.8894 | 0.9069 |
| 40 | 60 | 97,115 | 0.9018 | 0.9263 | 0.9087 | 0.9174 | 0.8915 | 0.8668 |
| 30 | 70 | 83,240 | 0.9040 | 0.9520 | 0.9087 | 0.9298 | 0.8931 | 0.8073 |
| 20 | 80 | 72,835 | 0.9048 | 0.9705 | 0.9087 | 0.9386 | 0.8894 | 0.7088 |
| 10 | 90 | 64,740 | 0.9069 | 0.9868 | 0.9087 | 0.9461 | 0.8908 | 0.5201 |
| 0 | 100 | 58,270 | 0.9087 | 1.0000 | 0.9087 | 0.9521 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8902 | 0.0000 | 0.0000 | 0.0000 | 0.8902 | 1.0000 |
| 90 | 10 | 299,940 | 0.8921 | 0.4791 | 0.9074 | 0.6271 | 0.8904 | 0.9886 |
| 80 | 20 | 291,350 | 0.8934 | 0.6733 | 0.9073 | 0.7730 | 0.8900 | 0.9746 |
| 70 | 30 | 194,230 | 0.8960 | 0.7814 | 0.9073 | 0.8396 | 0.8912 | 0.9573 |
| 60 | 40 | 145,675 | 0.8971 | 0.8464 | 0.9073 | 0.8758 | 0.8903 | 0.9351 |
| 50 | 50 | 116,540 | 0.8975 | 0.8899 | 0.9073 | 0.8985 | 0.8878 | 0.9055 |
| 40 | 60 | 97,115 | 0.9003 | 0.9250 | 0.9073 | 0.9161 | 0.8897 | 0.8648 |
| 30 | 70 | 83,240 | 0.9026 | 0.9513 | 0.9073 | 0.9288 | 0.8916 | 0.8048 |
| 20 | 80 | 72,835 | 0.9034 | 0.9700 | 0.9073 | 0.9376 | 0.8877 | 0.7054 |
| 10 | 90 | 64,740 | 0.9055 | 0.9866 | 0.9073 | 0.9453 | 0.8894 | 0.5160 |
| 0 | 100 | 58,270 | 0.9073 | 1.0000 | 0.9073 | 0.9514 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8902 | 0.0000 | 0.0000 | 0.0000 | 0.8902 | 1.0000 |
| 90 | 10 | 299,940 | 0.8921 | 0.4791 | 0.9074 | 0.6271 | 0.8904 | 0.9886 |
| 80 | 20 | 291,350 | 0.8934 | 0.6733 | 0.9073 | 0.7730 | 0.8900 | 0.9746 |
| 70 | 30 | 194,230 | 0.8960 | 0.7814 | 0.9073 | 0.8396 | 0.8912 | 0.9573 |
| 60 | 40 | 145,675 | 0.8971 | 0.8464 | 0.9073 | 0.8758 | 0.8903 | 0.9351 |
| 50 | 50 | 116,540 | 0.8975 | 0.8899 | 0.9073 | 0.8985 | 0.8878 | 0.9055 |
| 40 | 60 | 97,115 | 0.9003 | 0.9250 | 0.9073 | 0.9161 | 0.8897 | 0.8648 |
| 30 | 70 | 83,240 | 0.9026 | 0.9513 | 0.9073 | 0.9288 | 0.8916 | 0.8048 |
| 20 | 80 | 72,835 | 0.9034 | 0.9700 | 0.9073 | 0.9376 | 0.8877 | 0.7054 |
| 10 | 90 | 64,740 | 0.9055 | 0.9866 | 0.9073 | 0.9453 | 0.8894 | 0.5160 |
| 0 | 100 | 58,270 | 0.9073 | 1.0000 | 0.9073 | 0.9514 | 0.0000 | 0.0000 |


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
0.15       0.2301   0.2061   0.1446   0.9995   0.9993   0.1149   <--
0.20       0.2301   0.2061   0.1446   0.9995   0.9993   0.1149  
0.25       0.2301   0.2061   0.1446   0.9995   0.9993   0.1149  
0.30       0.2301   0.2061   0.1446   0.9995   0.9993   0.1149  
0.35       0.2301   0.2061   0.1446   0.9995   0.9993   0.1149  
0.40       0.2301   0.2061   0.1446   0.9995   0.9993   0.1149  
0.45       0.2301   0.2061   0.1446   0.9995   0.9993   0.1149  
0.50       0.2301   0.2061   0.1446   0.9995   0.9993   0.1149  
0.55       0.2301   0.2061   0.1446   0.9995   0.9993   0.1149  
0.60       0.2301   0.2061   0.1446   0.9995   0.9993   0.1149  
0.65       0.2301   0.2061   0.1446   0.9995   0.9993   0.1149  
0.70       0.2301   0.2061   0.1446   0.9995   0.9993   0.1149  
0.75       0.2301   0.2061   0.1446   0.9995   0.9993   0.1149  
0.80       0.2301   0.2061   0.1446   0.9995   0.9993   0.1149  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2301, F1=0.2061, Normal Recall=0.1446, Normal Precision=0.9995, Attack Recall=0.9993, Attack Precision=0.1149

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
0.15       0.3155   0.3687   0.1446   0.9988   0.9993   0.2260   <--
0.20       0.3155   0.3687   0.1446   0.9988   0.9993   0.2260  
0.25       0.3155   0.3687   0.1446   0.9988   0.9993   0.2260  
0.30       0.3155   0.3687   0.1446   0.9988   0.9993   0.2260  
0.35       0.3155   0.3687   0.1446   0.9988   0.9993   0.2260  
0.40       0.3155   0.3687   0.1446   0.9988   0.9993   0.2260  
0.45       0.3155   0.3687   0.1446   0.9988   0.9993   0.2260  
0.50       0.3155   0.3687   0.1446   0.9988   0.9993   0.2260  
0.55       0.3155   0.3687   0.1446   0.9988   0.9993   0.2260  
0.60       0.3155   0.3687   0.1446   0.9988   0.9993   0.2260  
0.65       0.3155   0.3687   0.1446   0.9988   0.9993   0.2260  
0.70       0.3155   0.3687   0.1446   0.9988   0.9993   0.2260  
0.75       0.3155   0.3687   0.1446   0.9988   0.9993   0.2260  
0.80       0.3155   0.3687   0.1446   0.9988   0.9993   0.2260  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3155, F1=0.3687, Normal Recall=0.1446, Normal Precision=0.9988, Attack Recall=0.9993, Attack Precision=0.2260

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
0.15       0.4010   0.5002   0.1446   0.9979   0.9993   0.3336   <--
0.20       0.4010   0.5002   0.1446   0.9979   0.9993   0.3336  
0.25       0.4010   0.5002   0.1446   0.9979   0.9993   0.3336  
0.30       0.4010   0.5002   0.1446   0.9979   0.9993   0.3336  
0.35       0.4010   0.5002   0.1446   0.9979   0.9993   0.3336  
0.40       0.4010   0.5002   0.1446   0.9979   0.9993   0.3336  
0.45       0.4010   0.5002   0.1446   0.9979   0.9993   0.3336  
0.50       0.4010   0.5002   0.1446   0.9979   0.9993   0.3336  
0.55       0.4010   0.5002   0.1446   0.9979   0.9993   0.3336  
0.60       0.4010   0.5002   0.1446   0.9979   0.9993   0.3336  
0.65       0.4010   0.5002   0.1446   0.9979   0.9993   0.3336  
0.70       0.4010   0.5002   0.1446   0.9979   0.9993   0.3336  
0.75       0.4010   0.5002   0.1446   0.9979   0.9993   0.3336  
0.80       0.4010   0.5002   0.1446   0.9979   0.9993   0.3336  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4010, F1=0.5002, Normal Recall=0.1446, Normal Precision=0.9979, Attack Recall=0.9993, Attack Precision=0.3336

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
0.15       0.4861   0.6087   0.1439   0.9968   0.9993   0.4376   <--
0.20       0.4861   0.6087   0.1439   0.9968   0.9993   0.4376  
0.25       0.4861   0.6087   0.1439   0.9968   0.9993   0.4376  
0.30       0.4861   0.6087   0.1439   0.9968   0.9993   0.4376  
0.35       0.4861   0.6087   0.1439   0.9968   0.9993   0.4376  
0.40       0.4861   0.6087   0.1439   0.9968   0.9993   0.4376  
0.45       0.4861   0.6087   0.1439   0.9968   0.9993   0.4376  
0.50       0.4861   0.6087   0.1439   0.9968   0.9993   0.4376  
0.55       0.4861   0.6087   0.1439   0.9968   0.9993   0.4376  
0.60       0.4861   0.6087   0.1439   0.9968   0.9993   0.4376  
0.65       0.4861   0.6087   0.1439   0.9968   0.9993   0.4376  
0.70       0.4861   0.6087   0.1439   0.9968   0.9993   0.4376  
0.75       0.4861   0.6087   0.1439   0.9968   0.9993   0.4376  
0.80       0.4861   0.6087   0.1439   0.9968   0.9993   0.4376  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4861, F1=0.6087, Normal Recall=0.1439, Normal Precision=0.9968, Attack Recall=0.9993, Attack Precision=0.4376

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
0.15       0.5715   0.6999   0.1436   0.9951   0.9993   0.5385   <--
0.20       0.5715   0.6999   0.1436   0.9951   0.9993   0.5385  
0.25       0.5715   0.6999   0.1436   0.9951   0.9993   0.5385  
0.30       0.5715   0.6999   0.1436   0.9951   0.9993   0.5385  
0.35       0.5715   0.6999   0.1436   0.9951   0.9993   0.5385  
0.40       0.5715   0.6999   0.1436   0.9951   0.9993   0.5385  
0.45       0.5715   0.6999   0.1436   0.9951   0.9993   0.5385  
0.50       0.5715   0.6999   0.1436   0.9951   0.9993   0.5385  
0.55       0.5715   0.6999   0.1436   0.9951   0.9993   0.5385  
0.60       0.5715   0.6999   0.1436   0.9951   0.9993   0.5385  
0.65       0.5715   0.6999   0.1436   0.9951   0.9993   0.5385  
0.70       0.5715   0.6999   0.1436   0.9951   0.9993   0.5385  
0.75       0.5715   0.6999   0.1436   0.9951   0.9993   0.5385  
0.80       0.5715   0.6999   0.1436   0.9951   0.9993   0.5385  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5715, F1=0.6999, Normal Recall=0.1436, Normal Precision=0.9951, Attack Recall=0.9993, Attack Precision=0.5385

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
0.15       0.8933   0.6296   0.8918   0.9885   0.9067   0.4822   <--
0.20       0.8933   0.6296   0.8918   0.9885   0.9067   0.4822  
0.25       0.8933   0.6296   0.8918   0.9885   0.9067   0.4822  
0.30       0.8933   0.6296   0.8918   0.9885   0.9067   0.4822  
0.35       0.8933   0.6296   0.8918   0.9885   0.9067   0.4822  
0.40       0.8933   0.6296   0.8918   0.9885   0.9067   0.4822  
0.45       0.8933   0.6296   0.8918   0.9885   0.9067   0.4822  
0.50       0.8933   0.6296   0.8918   0.9885   0.9067   0.4822  
0.55       0.8933   0.6296   0.8918   0.9885   0.9067   0.4822  
0.60       0.8933   0.6296   0.8918   0.9885   0.9067   0.4822  
0.65       0.8933   0.6296   0.8918   0.9885   0.9067   0.4822  
0.70       0.8933   0.6296   0.8918   0.9885   0.9067   0.4822  
0.75       0.8933   0.6296   0.8918   0.9885   0.9067   0.4822  
0.80       0.8933   0.6296   0.8918   0.9885   0.9067   0.4822  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8933, F1=0.6296, Normal Recall=0.8918, Normal Precision=0.9885, Attack Recall=0.9067, Attack Precision=0.4822

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
0.15       0.8955   0.7767   0.8922   0.9750   0.9087   0.6783   <--
0.20       0.8955   0.7767   0.8922   0.9750   0.9087   0.6783  
0.25       0.8955   0.7767   0.8922   0.9750   0.9087   0.6783  
0.30       0.8955   0.7767   0.8922   0.9750   0.9087   0.6783  
0.35       0.8955   0.7767   0.8922   0.9750   0.9087   0.6783  
0.40       0.8955   0.7767   0.8922   0.9750   0.9087   0.6783  
0.45       0.8955   0.7767   0.8922   0.9750   0.9087   0.6783  
0.50       0.8955   0.7767   0.8922   0.9750   0.9087   0.6783  
0.55       0.8955   0.7767   0.8922   0.9750   0.9087   0.6783  
0.60       0.8955   0.7767   0.8922   0.9750   0.9087   0.6783  
0.65       0.8955   0.7767   0.8922   0.9750   0.9087   0.6783  
0.70       0.8955   0.7767   0.8922   0.9750   0.9087   0.6783  
0.75       0.8955   0.7767   0.8922   0.9750   0.9087   0.6783  
0.80       0.8955   0.7767   0.8922   0.9750   0.9087   0.6783  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8955, F1=0.7767, Normal Recall=0.8922, Normal Precision=0.9750, Attack Recall=0.9087, Attack Precision=0.6783

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
0.15       0.8974   0.8416   0.8925   0.9580   0.9087   0.7837   <--
0.20       0.8974   0.8416   0.8925   0.9580   0.9087   0.7837  
0.25       0.8974   0.8416   0.8925   0.9580   0.9087   0.7837  
0.30       0.8974   0.8416   0.8925   0.9580   0.9087   0.7837  
0.35       0.8974   0.8416   0.8925   0.9580   0.9087   0.7837  
0.40       0.8974   0.8416   0.8925   0.9580   0.9087   0.7837  
0.45       0.8974   0.8416   0.8925   0.9580   0.9087   0.7837  
0.50       0.8974   0.8416   0.8925   0.9580   0.9087   0.7837  
0.55       0.8974   0.8416   0.8925   0.9580   0.9087   0.7837  
0.60       0.8974   0.8416   0.8925   0.9580   0.9087   0.7837  
0.65       0.8974   0.8416   0.8925   0.9580   0.9087   0.7837  
0.70       0.8974   0.8416   0.8925   0.9580   0.9087   0.7837  
0.75       0.8974   0.8416   0.8925   0.9580   0.9087   0.7837  
0.80       0.8974   0.8416   0.8925   0.9580   0.9087   0.7837  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8974, F1=0.8416, Normal Recall=0.8925, Normal Precision=0.9580, Attack Recall=0.9087, Attack Precision=0.7837

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
0.15       0.8986   0.8775   0.8918   0.9361   0.9087   0.8485   <--
0.20       0.8986   0.8775   0.8918   0.9361   0.9087   0.8485  
0.25       0.8986   0.8775   0.8918   0.9361   0.9087   0.8485  
0.30       0.8986   0.8775   0.8918   0.9361   0.9087   0.8485  
0.35       0.8986   0.8775   0.8918   0.9361   0.9087   0.8485  
0.40       0.8986   0.8775   0.8918   0.9361   0.9087   0.8485  
0.45       0.8986   0.8775   0.8918   0.9361   0.9087   0.8485  
0.50       0.8986   0.8775   0.8918   0.9361   0.9087   0.8485  
0.55       0.8986   0.8775   0.8918   0.9361   0.9087   0.8485  
0.60       0.8986   0.8775   0.8918   0.9361   0.9087   0.8485  
0.65       0.8986   0.8775   0.8918   0.9361   0.9087   0.8485  
0.70       0.8986   0.8775   0.8918   0.9361   0.9087   0.8485  
0.75       0.8986   0.8775   0.8918   0.9361   0.9087   0.8485  
0.80       0.8986   0.8775   0.8918   0.9361   0.9087   0.8485  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8986, F1=0.8775, Normal Recall=0.8918, Normal Precision=0.9361, Attack Recall=0.9087, Attack Precision=0.8485

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
0.15       0.9000   0.9008   0.8913   0.9070   0.9087   0.8931   <--
0.20       0.9000   0.9008   0.8913   0.9070   0.9087   0.8931  
0.25       0.9000   0.9008   0.8913   0.9070   0.9087   0.8931  
0.30       0.9000   0.9008   0.8913   0.9070   0.9087   0.8931  
0.35       0.9000   0.9008   0.8913   0.9070   0.9087   0.8931  
0.40       0.9000   0.9008   0.8913   0.9070   0.9087   0.8931  
0.45       0.9000   0.9008   0.8913   0.9070   0.9087   0.8931  
0.50       0.9000   0.9008   0.8913   0.9070   0.9087   0.8931  
0.55       0.9000   0.9008   0.8913   0.9070   0.9087   0.8931  
0.60       0.9000   0.9008   0.8913   0.9070   0.9087   0.8931  
0.65       0.9000   0.9008   0.8913   0.9070   0.9087   0.8931  
0.70       0.9000   0.9008   0.8913   0.9070   0.9087   0.8931  
0.75       0.9000   0.9008   0.8913   0.9070   0.9087   0.8931  
0.80       0.9000   0.9008   0.8913   0.9070   0.9087   0.8931  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9000, F1=0.9008, Normal Recall=0.8913, Normal Precision=0.9070, Attack Recall=0.9087, Attack Precision=0.8931

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
0.15       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785   <--
0.20       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.25       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.30       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.35       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.40       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.45       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.50       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.55       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.60       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.65       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.70       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.75       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.80       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8919, F1=0.6261, Normal Recall=0.8904, Normal Precision=0.9883, Attack Recall=0.9054, Attack Precision=0.4785

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
0.15       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750   <--
0.20       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.25       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.30       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.35       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.40       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.45       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.50       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.55       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.60       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.65       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.70       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.75       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.80       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8941, F1=0.7741, Normal Recall=0.8908, Normal Precision=0.9746, Attack Recall=0.9073, Attack Precision=0.6750

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
0.15       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809   <--
0.20       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.25       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.30       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.35       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.40       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.45       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.50       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.55       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.60       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.65       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.70       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.75       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.80       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8958, F1=0.8394, Normal Recall=0.8909, Normal Precision=0.9573, Attack Recall=0.9073, Attack Precision=0.7809

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
0.15       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465   <--
0.20       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.25       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.30       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.35       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.40       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.45       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.50       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.55       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.60       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.65       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.70       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.75       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.80       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8971, F1=0.8758, Normal Recall=0.8903, Normal Precision=0.9351, Attack Recall=0.9073, Attack Precision=0.8465

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
0.15       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916   <--
0.20       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.25       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.30       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.35       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.40       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.45       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.50       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.55       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.60       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.65       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.70       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.75       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.80       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8985, F1=0.8994, Normal Recall=0.8897, Normal Precision=0.9056, Attack Recall=0.9073, Attack Precision=0.8916

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
0.15       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785   <--
0.20       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.25       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.30       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.35       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.40       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.45       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.50       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.55       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.60       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.65       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.70       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.75       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
0.80       0.8919   0.6261   0.8904   0.9883   0.9054   0.4785  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8919, F1=0.6261, Normal Recall=0.8904, Normal Precision=0.9883, Attack Recall=0.9054, Attack Precision=0.4785

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
0.15       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750   <--
0.20       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.25       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.30       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.35       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.40       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.45       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.50       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.55       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.60       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.65       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.70       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.75       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
0.80       0.8941   0.7741   0.8908   0.9746   0.9073   0.6750  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8941, F1=0.7741, Normal Recall=0.8908, Normal Precision=0.9746, Attack Recall=0.9073, Attack Precision=0.6750

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
0.15       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809   <--
0.20       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.25       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.30       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.35       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.40       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.45       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.50       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.55       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.60       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.65       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.70       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.75       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
0.80       0.8958   0.8394   0.8909   0.9573   0.9073   0.7809  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8958, F1=0.8394, Normal Recall=0.8909, Normal Precision=0.9573, Attack Recall=0.9073, Attack Precision=0.7809

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
0.15       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465   <--
0.20       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.25       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.30       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.35       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.40       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.45       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.50       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.55       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.60       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.65       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.70       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.75       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
0.80       0.8971   0.8758   0.8903   0.9351   0.9073   0.8465  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8971, F1=0.8758, Normal Recall=0.8903, Normal Precision=0.9351, Attack Recall=0.9073, Attack Precision=0.8465

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
0.15       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916   <--
0.20       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.25       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.30       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.35       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.40       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.45       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.50       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.55       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.60       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.65       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.70       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.75       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
0.80       0.8985   0.8994   0.8897   0.9056   0.9073   0.8916  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8985, F1=0.8994, Normal Recall=0.8897, Normal Precision=0.9056, Attack Recall=0.9073, Attack Precision=0.8916

```

