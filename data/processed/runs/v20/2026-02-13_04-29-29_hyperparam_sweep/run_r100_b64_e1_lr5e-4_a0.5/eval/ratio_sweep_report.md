# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e1_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-19 12:19:09 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9104 | 0.9056 | 0.9002 | 0.8958 | 0.8904 | 0.8854 | 0.8804 | 0.8755 | 0.8703 | 0.8654 | 0.8606 |
| QAT+Prune only | 0.7172 | 0.7419 | 0.7661 | 0.7921 | 0.8166 | 0.8396 | 0.8664 | 0.8909 | 0.9156 | 0.9406 | 0.9656 |
| QAT+PTQ | 0.7179 | 0.7428 | 0.7671 | 0.7932 | 0.8177 | 0.8409 | 0.8676 | 0.8923 | 0.9171 | 0.9422 | 0.9672 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7179 | 0.7428 | 0.7671 | 0.7932 | 0.8177 | 0.8409 | 0.8676 | 0.8923 | 0.9171 | 0.9422 | 0.9672 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6462 | 0.7752 | 0.8321 | 0.8627 | 0.8825 | 0.8962 | 0.9063 | 0.9139 | 0.9201 | 0.9251 |
| QAT+Prune only | 0.0000 | 0.4281 | 0.6228 | 0.7359 | 0.8081 | 0.8575 | 0.8966 | 0.9253 | 0.9482 | 0.9670 | 0.9825 |
| QAT+PTQ | 0.0000 | 0.4293 | 0.6243 | 0.7372 | 0.8094 | 0.8587 | 0.8976 | 0.9263 | 0.9492 | 0.9679 | 0.9833 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4293 | 0.6243 | 0.7372 | 0.8094 | 0.8587 | 0.8976 | 0.9263 | 0.9492 | 0.9679 | 0.9833 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9104 | 0.9104 | 0.9101 | 0.9109 | 0.9103 | 0.9103 | 0.9101 | 0.9103 | 0.9092 | 0.9093 | 0.0000 |
| QAT+Prune only | 0.7172 | 0.7170 | 0.7162 | 0.7177 | 0.7172 | 0.7136 | 0.7176 | 0.7168 | 0.7158 | 0.7161 | 0.0000 |
| QAT+PTQ | 0.7179 | 0.7178 | 0.7171 | 0.7186 | 0.7181 | 0.7146 | 0.7183 | 0.7176 | 0.7169 | 0.7178 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7179 | 0.7178 | 0.7171 | 0.7186 | 0.7181 | 0.7146 | 0.7183 | 0.7176 | 0.7169 | 0.7178 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9104 | 0.0000 | 0.0000 | 0.0000 | 0.9104 | 1.0000 |
| 90 | 10 | 299,940 | 0.9056 | 0.5167 | 0.8624 | 0.6462 | 0.9104 | 0.9835 |
| 80 | 20 | 291,350 | 0.9002 | 0.7053 | 0.8606 | 0.7752 | 0.9101 | 0.9631 |
| 70 | 30 | 194,230 | 0.8958 | 0.8054 | 0.8606 | 0.8321 | 0.9109 | 0.9384 |
| 60 | 40 | 145,675 | 0.8904 | 0.8648 | 0.8606 | 0.8627 | 0.9103 | 0.9073 |
| 50 | 50 | 116,540 | 0.8854 | 0.9056 | 0.8606 | 0.8825 | 0.9103 | 0.8672 |
| 40 | 60 | 97,115 | 0.8804 | 0.9349 | 0.8606 | 0.8962 | 0.9101 | 0.8131 |
| 30 | 70 | 83,240 | 0.8755 | 0.9573 | 0.8606 | 0.9063 | 0.9103 | 0.7367 |
| 20 | 80 | 72,835 | 0.8703 | 0.9743 | 0.8606 | 0.9139 | 0.9092 | 0.6198 |
| 10 | 90 | 64,740 | 0.8654 | 0.9884 | 0.8606 | 0.9201 | 0.9093 | 0.4201 |
| 0 | 100 | 58,270 | 0.8606 | 1.0000 | 0.8606 | 0.9251 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7172 | 0.0000 | 0.0000 | 0.0000 | 0.7172 | 1.0000 |
| 90 | 10 | 299,940 | 0.7419 | 0.2750 | 0.9659 | 0.4281 | 0.7170 | 0.9947 |
| 80 | 20 | 291,350 | 0.7661 | 0.4597 | 0.9656 | 0.6228 | 0.7162 | 0.9881 |
| 70 | 30 | 194,230 | 0.7921 | 0.5945 | 0.9656 | 0.7359 | 0.7177 | 0.9798 |
| 60 | 40 | 145,675 | 0.8166 | 0.6948 | 0.9656 | 0.8081 | 0.7172 | 0.9690 |
| 50 | 50 | 116,540 | 0.8396 | 0.7713 | 0.9656 | 0.8575 | 0.7136 | 0.9540 |
| 40 | 60 | 97,115 | 0.8664 | 0.8368 | 0.9656 | 0.8966 | 0.7176 | 0.9328 |
| 30 | 70 | 83,240 | 0.8909 | 0.8884 | 0.9656 | 0.9253 | 0.7168 | 0.8992 |
| 20 | 80 | 72,835 | 0.9156 | 0.9315 | 0.9656 | 0.9482 | 0.7158 | 0.8386 |
| 10 | 90 | 64,740 | 0.9406 | 0.9684 | 0.9656 | 0.9670 | 0.7161 | 0.6979 |
| 0 | 100 | 58,270 | 0.9656 | 1.0000 | 0.9656 | 0.9825 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7179 | 0.0000 | 0.0000 | 0.0000 | 0.7179 | 1.0000 |
| 90 | 10 | 299,940 | 0.7428 | 0.2758 | 0.9673 | 0.4293 | 0.7178 | 0.9950 |
| 80 | 20 | 291,350 | 0.7671 | 0.4609 | 0.9672 | 0.6243 | 0.7171 | 0.9887 |
| 70 | 30 | 194,230 | 0.7932 | 0.5956 | 0.9672 | 0.7372 | 0.7186 | 0.9808 |
| 60 | 40 | 145,675 | 0.8177 | 0.6958 | 0.9672 | 0.8094 | 0.7181 | 0.9704 |
| 50 | 50 | 116,540 | 0.8409 | 0.7721 | 0.9672 | 0.8587 | 0.7146 | 0.9561 |
| 40 | 60 | 97,115 | 0.8676 | 0.8374 | 0.9672 | 0.8976 | 0.7183 | 0.9358 |
| 30 | 70 | 83,240 | 0.8923 | 0.8888 | 0.9672 | 0.9263 | 0.7176 | 0.9035 |
| 20 | 80 | 72,835 | 0.9171 | 0.9318 | 0.9672 | 0.9492 | 0.7169 | 0.8451 |
| 10 | 90 | 64,740 | 0.9422 | 0.9686 | 0.9672 | 0.9679 | 0.7178 | 0.7083 |
| 0 | 100 | 58,270 | 0.9672 | 1.0000 | 0.9672 | 0.9833 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7179 | 0.0000 | 0.0000 | 0.0000 | 0.7179 | 1.0000 |
| 90 | 10 | 299,940 | 0.7428 | 0.2758 | 0.9673 | 0.4293 | 0.7178 | 0.9950 |
| 80 | 20 | 291,350 | 0.7671 | 0.4609 | 0.9672 | 0.6243 | 0.7171 | 0.9887 |
| 70 | 30 | 194,230 | 0.7932 | 0.5956 | 0.9672 | 0.7372 | 0.7186 | 0.9808 |
| 60 | 40 | 145,675 | 0.8177 | 0.6958 | 0.9672 | 0.8094 | 0.7181 | 0.9704 |
| 50 | 50 | 116,540 | 0.8409 | 0.7721 | 0.9672 | 0.8587 | 0.7146 | 0.9561 |
| 40 | 60 | 97,115 | 0.8676 | 0.8374 | 0.9672 | 0.8976 | 0.7183 | 0.9358 |
| 30 | 70 | 83,240 | 0.8923 | 0.8888 | 0.9672 | 0.9263 | 0.7176 | 0.9035 |
| 20 | 80 | 72,835 | 0.9171 | 0.9318 | 0.9672 | 0.9492 | 0.7169 | 0.8451 |
| 10 | 90 | 64,740 | 0.9422 | 0.9686 | 0.9672 | 0.9679 | 0.7178 | 0.7083 |
| 0 | 100 | 58,270 | 0.9672 | 1.0000 | 0.9672 | 0.9833 | 0.0000 | 0.0000 |


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
0.15       0.9055   0.6460   0.9104   0.9834   0.8620   0.5165   <--
0.20       0.9055   0.6460   0.9104   0.9834   0.8620   0.5165  
0.25       0.9055   0.6460   0.9104   0.9834   0.8620   0.5165  
0.30       0.9055   0.6460   0.9104   0.9834   0.8620   0.5165  
0.35       0.9055   0.6460   0.9104   0.9834   0.8620   0.5165  
0.40       0.9055   0.6460   0.9104   0.9834   0.8620   0.5165  
0.45       0.9055   0.6460   0.9104   0.9834   0.8620   0.5165  
0.50       0.9055   0.6460   0.9104   0.9834   0.8620   0.5165  
0.55       0.9055   0.6460   0.9104   0.9834   0.8620   0.5165  
0.60       0.9055   0.6460   0.9104   0.9834   0.8620   0.5165  
0.65       0.9055   0.6460   0.9104   0.9834   0.8620   0.5165  
0.70       0.9055   0.6460   0.9104   0.9834   0.8620   0.5165  
0.75       0.9055   0.6460   0.9104   0.9834   0.8620   0.5165  
0.80       0.9055   0.6460   0.9104   0.9834   0.8620   0.5165  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9055, F1=0.6460, Normal Recall=0.9104, Normal Precision=0.9834, Attack Recall=0.8620, Attack Precision=0.5165

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
0.15       0.9004   0.7756   0.9104   0.9631   0.8606   0.7060   <--
0.20       0.9004   0.7756   0.9104   0.9631   0.8606   0.7060  
0.25       0.9004   0.7756   0.9104   0.9631   0.8606   0.7060  
0.30       0.9004   0.7756   0.9104   0.9631   0.8606   0.7060  
0.35       0.9004   0.7756   0.9104   0.9631   0.8606   0.7060  
0.40       0.9004   0.7756   0.9104   0.9631   0.8606   0.7060  
0.45       0.9004   0.7756   0.9104   0.9631   0.8606   0.7060  
0.50       0.9004   0.7756   0.9104   0.9631   0.8606   0.7060  
0.55       0.9004   0.7756   0.9104   0.9631   0.8606   0.7060  
0.60       0.9004   0.7756   0.9104   0.9631   0.8606   0.7060  
0.65       0.9004   0.7756   0.9104   0.9631   0.8606   0.7060  
0.70       0.9004   0.7756   0.9104   0.9631   0.8606   0.7060  
0.75       0.9004   0.7756   0.9104   0.9631   0.8606   0.7060  
0.80       0.9004   0.7756   0.9104   0.9631   0.8606   0.7060  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9004, F1=0.7756, Normal Recall=0.9104, Normal Precision=0.9631, Attack Recall=0.8606, Attack Precision=0.7060

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
0.15       0.8961   0.8325   0.9114   0.9385   0.8606   0.8063   <--
0.20       0.8961   0.8325   0.9114   0.9385   0.8606   0.8063  
0.25       0.8961   0.8325   0.9114   0.9385   0.8606   0.8063  
0.30       0.8961   0.8325   0.9114   0.9385   0.8606   0.8063  
0.35       0.8961   0.8325   0.9114   0.9385   0.8606   0.8063  
0.40       0.8961   0.8325   0.9114   0.9385   0.8606   0.8063  
0.45       0.8961   0.8325   0.9114   0.9385   0.8606   0.8063  
0.50       0.8961   0.8325   0.9114   0.9385   0.8606   0.8063  
0.55       0.8961   0.8325   0.9114   0.9385   0.8606   0.8063  
0.60       0.8961   0.8325   0.9114   0.9385   0.8606   0.8063  
0.65       0.8961   0.8325   0.9114   0.9385   0.8606   0.8063  
0.70       0.8961   0.8325   0.9114   0.9385   0.8606   0.8063  
0.75       0.8961   0.8325   0.9114   0.9385   0.8606   0.8063  
0.80       0.8961   0.8325   0.9114   0.9385   0.8606   0.8063  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8961, F1=0.8325, Normal Recall=0.9114, Normal Precision=0.9385, Attack Recall=0.8606, Attack Precision=0.8063

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
0.15       0.8906   0.8629   0.9106   0.9074   0.8606   0.8652   <--
0.20       0.8906   0.8629   0.9106   0.9074   0.8606   0.8652  
0.25       0.8906   0.8629   0.9106   0.9074   0.8606   0.8652  
0.30       0.8906   0.8629   0.9106   0.9074   0.8606   0.8652  
0.35       0.8906   0.8629   0.9106   0.9074   0.8606   0.8652  
0.40       0.8906   0.8629   0.9106   0.9074   0.8606   0.8652  
0.45       0.8906   0.8629   0.9106   0.9074   0.8606   0.8652  
0.50       0.8906   0.8629   0.9106   0.9074   0.8606   0.8652  
0.55       0.8906   0.8629   0.9106   0.9074   0.8606   0.8652  
0.60       0.8906   0.8629   0.9106   0.9074   0.8606   0.8652  
0.65       0.8906   0.8629   0.9106   0.9074   0.8606   0.8652  
0.70       0.8906   0.8629   0.9106   0.9074   0.8606   0.8652  
0.75       0.8906   0.8629   0.9106   0.9074   0.8606   0.8652  
0.80       0.8906   0.8629   0.9106   0.9074   0.8606   0.8652  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8906, F1=0.8629, Normal Recall=0.9106, Normal Precision=0.9074, Attack Recall=0.8606, Attack Precision=0.8652

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
0.15       0.8852   0.8823   0.9099   0.8671   0.8606   0.9052   <--
0.20       0.8852   0.8823   0.9099   0.8671   0.8606   0.9052  
0.25       0.8852   0.8823   0.9099   0.8671   0.8606   0.9052  
0.30       0.8852   0.8823   0.9099   0.8671   0.8606   0.9052  
0.35       0.8852   0.8823   0.9099   0.8671   0.8606   0.9052  
0.40       0.8852   0.8823   0.9099   0.8671   0.8606   0.9052  
0.45       0.8852   0.8823   0.9099   0.8671   0.8606   0.9052  
0.50       0.8852   0.8823   0.9099   0.8671   0.8606   0.9052  
0.55       0.8852   0.8823   0.9099   0.8671   0.8606   0.9052  
0.60       0.8852   0.8823   0.9099   0.8671   0.8606   0.9052  
0.65       0.8852   0.8823   0.9099   0.8671   0.8606   0.9052  
0.70       0.8852   0.8823   0.9099   0.8671   0.8606   0.9052  
0.75       0.8852   0.8823   0.9099   0.8671   0.8606   0.9052  
0.80       0.8852   0.8823   0.9099   0.8671   0.8606   0.9052  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8852, F1=0.8823, Normal Recall=0.9099, Normal Precision=0.8671, Attack Recall=0.8606, Attack Precision=0.9052

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
0.15       0.7418   0.4277   0.7170   0.9946   0.9649   0.2748   <--
0.20       0.7418   0.4277   0.7170   0.9946   0.9649   0.2748  
0.25       0.7418   0.4277   0.7170   0.9946   0.9649   0.2748  
0.30       0.7418   0.4277   0.7170   0.9946   0.9649   0.2748  
0.35       0.7418   0.4277   0.7170   0.9946   0.9649   0.2748  
0.40       0.7418   0.4277   0.7170   0.9946   0.9649   0.2748  
0.45       0.7418   0.4277   0.7170   0.9946   0.9649   0.2748  
0.50       0.7418   0.4277   0.7170   0.9946   0.9649   0.2748  
0.55       0.7418   0.4277   0.7170   0.9946   0.9649   0.2748  
0.60       0.7418   0.4277   0.7170   0.9946   0.9649   0.2748  
0.65       0.7418   0.4277   0.7170   0.9946   0.9649   0.2748  
0.70       0.7418   0.4277   0.7170   0.9946   0.9649   0.2748  
0.75       0.7418   0.4277   0.7170   0.9946   0.9649   0.2748  
0.80       0.7418   0.4277   0.7170   0.9946   0.9649   0.2748  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7418, F1=0.4277, Normal Recall=0.7170, Normal Precision=0.9946, Attack Recall=0.9649, Attack Precision=0.2748

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
0.15       0.7670   0.6238   0.7174   0.9881   0.9656   0.4607   <--
0.20       0.7670   0.6238   0.7174   0.9881   0.9656   0.4607  
0.25       0.7670   0.6238   0.7174   0.9881   0.9656   0.4607  
0.30       0.7670   0.6238   0.7174   0.9881   0.9656   0.4607  
0.35       0.7670   0.6238   0.7174   0.9881   0.9656   0.4607  
0.40       0.7670   0.6238   0.7174   0.9881   0.9656   0.4607  
0.45       0.7670   0.6238   0.7174   0.9881   0.9656   0.4607  
0.50       0.7670   0.6238   0.7174   0.9881   0.9656   0.4607  
0.55       0.7670   0.6238   0.7174   0.9881   0.9656   0.4607  
0.60       0.7670   0.6238   0.7174   0.9881   0.9656   0.4607  
0.65       0.7670   0.6238   0.7174   0.9881   0.9656   0.4607  
0.70       0.7670   0.6238   0.7174   0.9881   0.9656   0.4607  
0.75       0.7670   0.6238   0.7174   0.9881   0.9656   0.4607  
0.80       0.7670   0.6238   0.7174   0.9881   0.9656   0.4607  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7670, F1=0.6238, Normal Recall=0.7174, Normal Precision=0.9881, Attack Recall=0.9656, Attack Precision=0.4607

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
0.15       0.7911   0.7350   0.7163   0.9798   0.9656   0.5933   <--
0.20       0.7911   0.7350   0.7163   0.9798   0.9656   0.5933  
0.25       0.7911   0.7350   0.7163   0.9798   0.9656   0.5933  
0.30       0.7911   0.7350   0.7163   0.9798   0.9656   0.5933  
0.35       0.7911   0.7350   0.7163   0.9798   0.9656   0.5933  
0.40       0.7911   0.7350   0.7163   0.9798   0.9656   0.5933  
0.45       0.7911   0.7350   0.7163   0.9798   0.9656   0.5933  
0.50       0.7911   0.7350   0.7163   0.9798   0.9656   0.5933  
0.55       0.7911   0.7350   0.7163   0.9798   0.9656   0.5933  
0.60       0.7911   0.7350   0.7163   0.9798   0.9656   0.5933  
0.65       0.7911   0.7350   0.7163   0.9798   0.9656   0.5933  
0.70       0.7911   0.7350   0.7163   0.9798   0.9656   0.5933  
0.75       0.7911   0.7350   0.7163   0.9798   0.9656   0.5933  
0.80       0.7911   0.7350   0.7163   0.9798   0.9656   0.5933  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7911, F1=0.7350, Normal Recall=0.7163, Normal Precision=0.9798, Attack Recall=0.9656, Attack Precision=0.5933

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
0.15       0.8163   0.8078   0.7167   0.9690   0.9656   0.6944   <--
0.20       0.8163   0.8078   0.7167   0.9690   0.9656   0.6944  
0.25       0.8163   0.8078   0.7167   0.9690   0.9656   0.6944  
0.30       0.8163   0.8078   0.7167   0.9690   0.9656   0.6944  
0.35       0.8163   0.8078   0.7167   0.9690   0.9656   0.6944  
0.40       0.8163   0.8078   0.7167   0.9690   0.9656   0.6944  
0.45       0.8163   0.8078   0.7167   0.9690   0.9656   0.6944  
0.50       0.8163   0.8078   0.7167   0.9690   0.9656   0.6944  
0.55       0.8163   0.8078   0.7167   0.9690   0.9656   0.6944  
0.60       0.8163   0.8078   0.7167   0.9690   0.9656   0.6944  
0.65       0.8163   0.8078   0.7167   0.9690   0.9656   0.6944  
0.70       0.8163   0.8078   0.7167   0.9690   0.9656   0.6944  
0.75       0.8163   0.8078   0.7167   0.9690   0.9656   0.6944  
0.80       0.8163   0.8078   0.7167   0.9690   0.9656   0.6944  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8163, F1=0.8078, Normal Recall=0.7167, Normal Precision=0.9690, Attack Recall=0.9656, Attack Precision=0.6944

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
0.15       0.8399   0.8578   0.7143   0.9540   0.9656   0.7717   <--
0.20       0.8399   0.8578   0.7143   0.9540   0.9656   0.7717  
0.25       0.8399   0.8578   0.7143   0.9540   0.9656   0.7717  
0.30       0.8399   0.8578   0.7143   0.9540   0.9656   0.7717  
0.35       0.8399   0.8578   0.7143   0.9540   0.9656   0.7717  
0.40       0.8399   0.8578   0.7143   0.9540   0.9656   0.7717  
0.45       0.8399   0.8578   0.7143   0.9540   0.9656   0.7717  
0.50       0.8399   0.8578   0.7143   0.9540   0.9656   0.7717  
0.55       0.8399   0.8578   0.7143   0.9540   0.9656   0.7717  
0.60       0.8399   0.8578   0.7143   0.9540   0.9656   0.7717  
0.65       0.8399   0.8578   0.7143   0.9540   0.9656   0.7717  
0.70       0.8399   0.8578   0.7143   0.9540   0.9656   0.7717  
0.75       0.8399   0.8578   0.7143   0.9540   0.9656   0.7717  
0.80       0.8399   0.8578   0.7143   0.9540   0.9656   0.7717  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8399, F1=0.8578, Normal Recall=0.7143, Normal Precision=0.9540, Attack Recall=0.9656, Attack Precision=0.7717

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
0.15       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757   <--
0.20       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.25       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.30       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.35       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.40       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.45       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.50       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.55       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.60       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.65       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.70       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.75       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.80       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7427, F1=0.4291, Normal Recall=0.7178, Normal Precision=0.9949, Attack Recall=0.9668, Attack Precision=0.2757

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
0.15       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618   <--
0.20       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.25       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.30       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.35       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.40       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.45       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.50       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.55       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.60       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.65       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.70       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.75       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.80       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7680, F1=0.6251, Normal Recall=0.7182, Normal Precision=0.9887, Attack Recall=0.9672, Attack Precision=0.4618

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
0.15       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943   <--
0.20       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.25       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.30       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.35       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.40       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.45       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.50       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.55       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.60       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.65       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.70       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.75       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.80       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7921, F1=0.7362, Normal Recall=0.7171, Normal Precision=0.9807, Attack Recall=0.9672, Attack Precision=0.5943

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
0.15       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953   <--
0.20       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.25       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.30       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.35       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.40       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.45       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.50       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.55       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.60       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.65       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.70       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.75       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.80       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8173, F1=0.8090, Normal Recall=0.7174, Normal Precision=0.9704, Attack Recall=0.9672, Attack Precision=0.6953

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
0.15       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724   <--
0.20       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.25       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.30       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.35       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.40       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.45       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.50       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.55       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.60       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.65       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.70       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.75       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.80       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8411, F1=0.8589, Normal Recall=0.7150, Normal Precision=0.9561, Attack Recall=0.9672, Attack Precision=0.7724

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
0.15       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757   <--
0.20       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.25       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.30       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.35       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.40       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.45       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.50       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.55       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.60       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.65       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.70       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.75       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
0.80       0.7427   0.4291   0.7178   0.9949   0.9668   0.2757  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7427, F1=0.4291, Normal Recall=0.7178, Normal Precision=0.9949, Attack Recall=0.9668, Attack Precision=0.2757

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
0.15       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618   <--
0.20       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.25       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.30       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.35       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.40       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.45       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.50       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.55       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.60       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.65       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.70       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.75       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
0.80       0.7680   0.6251   0.7182   0.9887   0.9672   0.4618  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7680, F1=0.6251, Normal Recall=0.7182, Normal Precision=0.9887, Attack Recall=0.9672, Attack Precision=0.4618

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
0.15       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943   <--
0.20       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.25       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.30       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.35       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.40       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.45       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.50       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.55       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.60       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.65       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.70       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.75       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
0.80       0.7921   0.7362   0.7171   0.9807   0.9672   0.5943  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7921, F1=0.7362, Normal Recall=0.7171, Normal Precision=0.9807, Attack Recall=0.9672, Attack Precision=0.5943

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
0.15       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953   <--
0.20       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.25       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.30       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.35       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.40       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.45       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.50       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.55       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.60       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.65       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.70       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.75       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
0.80       0.8173   0.8090   0.7174   0.9704   0.9672   0.6953  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8173, F1=0.8090, Normal Recall=0.7174, Normal Precision=0.9704, Attack Recall=0.9672, Attack Precision=0.6953

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
0.15       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724   <--
0.20       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.25       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.30       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.35       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.40       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.45       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.50       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.55       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.60       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.65       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.70       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.75       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
0.80       0.8411   0.8589   0.7150   0.9561   0.9672   0.7724  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8411, F1=0.8589, Normal Recall=0.7150, Normal Precision=0.9561, Attack Recall=0.9672, Attack Precision=0.7724

```

