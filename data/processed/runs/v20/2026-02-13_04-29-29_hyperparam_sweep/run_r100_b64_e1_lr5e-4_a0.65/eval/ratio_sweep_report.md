# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e1_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-19 13:48:30 |

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
| Original (TFLite) | 0.8628 | 0.8489 | 0.8335 | 0.8201 | 0.8045 | 0.7900 | 0.7755 | 0.7623 | 0.7468 | 0.7320 | 0.7178 |
| QAT+Prune only | 0.9189 | 0.9248 | 0.9307 | 0.9375 | 0.9433 | 0.9491 | 0.9561 | 0.9621 | 0.9679 | 0.9740 | 0.9802 |
| QAT+PTQ | 0.9175 | 0.9232 | 0.9290 | 0.9356 | 0.9414 | 0.9468 | 0.9537 | 0.9594 | 0.9652 | 0.9710 | 0.9772 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9175 | 0.9232 | 0.9290 | 0.9356 | 0.9414 | 0.9468 | 0.9537 | 0.9594 | 0.9652 | 0.9710 | 0.9772 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.4881 | 0.6330 | 0.7053 | 0.7460 | 0.7737 | 0.7932 | 0.8087 | 0.8194 | 0.8282 | 0.8357 |
| QAT+Prune only | 0.0000 | 0.7227 | 0.8497 | 0.9039 | 0.9326 | 0.9506 | 0.9640 | 0.9731 | 0.9799 | 0.9855 | 0.9900 |
| QAT+PTQ | 0.0000 | 0.7180 | 0.8463 | 0.9010 | 0.9302 | 0.9484 | 0.9621 | 0.9712 | 0.9782 | 0.9838 | 0.9885 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.7180 | 0.8463 | 0.9010 | 0.9302 | 0.9484 | 0.9621 | 0.9712 | 0.9782 | 0.9838 | 0.9885 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8628 | 0.8631 | 0.8625 | 0.8639 | 0.8624 | 0.8623 | 0.8620 | 0.8663 | 0.8631 | 0.8599 | 0.0000 |
| QAT+Prune only | 0.9189 | 0.9186 | 0.9183 | 0.9191 | 0.9187 | 0.9180 | 0.9198 | 0.9197 | 0.9183 | 0.9174 | 0.0000 |
| QAT+PTQ | 0.9175 | 0.9172 | 0.9169 | 0.9178 | 0.9175 | 0.9164 | 0.9185 | 0.9177 | 0.9171 | 0.9147 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9175 | 0.9172 | 0.9169 | 0.9178 | 0.9175 | 0.9164 | 0.9185 | 0.9177 | 0.9171 | 0.9147 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8628 | 0.0000 | 0.0000 | 0.0000 | 0.8628 | 1.0000 |
| 90 | 10 | 299,940 | 0.8489 | 0.3690 | 0.7205 | 0.4881 | 0.8631 | 0.9653 |
| 80 | 20 | 291,350 | 0.8335 | 0.5661 | 0.7178 | 0.6330 | 0.8625 | 0.9244 |
| 70 | 30 | 194,230 | 0.8201 | 0.6933 | 0.7178 | 0.7053 | 0.8639 | 0.8772 |
| 60 | 40 | 145,675 | 0.8045 | 0.7766 | 0.7178 | 0.7460 | 0.8624 | 0.8209 |
| 50 | 50 | 116,540 | 0.7900 | 0.8391 | 0.7178 | 0.7737 | 0.8623 | 0.7534 |
| 40 | 60 | 97,115 | 0.7755 | 0.8864 | 0.7178 | 0.7932 | 0.8620 | 0.6706 |
| 30 | 70 | 83,240 | 0.7623 | 0.9261 | 0.7178 | 0.8087 | 0.8663 | 0.5681 |
| 20 | 80 | 72,835 | 0.7468 | 0.9545 | 0.7178 | 0.8194 | 0.8631 | 0.4333 |
| 10 | 90 | 64,740 | 0.7320 | 0.9788 | 0.7178 | 0.8282 | 0.8599 | 0.2529 |
| 0 | 100 | 58,270 | 0.7178 | 1.0000 | 0.7178 | 0.8357 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9189 | 0.0000 | 0.0000 | 0.0000 | 0.9189 | 1.0000 |
| 90 | 10 | 299,940 | 0.9248 | 0.5724 | 0.9802 | 0.7227 | 0.9186 | 0.9976 |
| 80 | 20 | 291,350 | 0.9307 | 0.7499 | 0.9802 | 0.8497 | 0.9183 | 0.9947 |
| 70 | 30 | 194,230 | 0.9375 | 0.8386 | 0.9802 | 0.9039 | 0.9191 | 0.9909 |
| 60 | 40 | 145,675 | 0.9433 | 0.8894 | 0.9802 | 0.9326 | 0.9187 | 0.9859 |
| 50 | 50 | 116,540 | 0.9491 | 0.9228 | 0.9802 | 0.9506 | 0.9180 | 0.9789 |
| 40 | 60 | 97,115 | 0.9561 | 0.9483 | 0.9802 | 0.9640 | 0.9198 | 0.9688 |
| 30 | 70 | 83,240 | 0.9621 | 0.9661 | 0.9802 | 0.9731 | 0.9197 | 0.9523 |
| 20 | 80 | 72,835 | 0.9679 | 0.9796 | 0.9802 | 0.9799 | 0.9183 | 0.9208 |
| 10 | 90 | 64,740 | 0.9740 | 0.9907 | 0.9802 | 0.9855 | 0.9174 | 0.8377 |
| 0 | 100 | 58,270 | 0.9802 | 1.0000 | 0.9802 | 0.9900 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9175 | 0.0000 | 0.0000 | 0.0000 | 0.9175 | 1.0000 |
| 90 | 10 | 299,940 | 0.9232 | 0.5675 | 0.9773 | 0.7180 | 0.9172 | 0.9973 |
| 80 | 20 | 291,350 | 0.9290 | 0.7462 | 0.9772 | 0.8463 | 0.9169 | 0.9938 |
| 70 | 30 | 194,230 | 0.9356 | 0.8359 | 0.9772 | 0.9010 | 0.9178 | 0.9895 |
| 60 | 40 | 145,675 | 0.9414 | 0.8876 | 0.9772 | 0.9302 | 0.9175 | 0.9837 |
| 50 | 50 | 116,540 | 0.9468 | 0.9212 | 0.9772 | 0.9484 | 0.9164 | 0.9758 |
| 40 | 60 | 97,115 | 0.9537 | 0.9473 | 0.9772 | 0.9621 | 0.9185 | 0.9642 |
| 30 | 70 | 83,240 | 0.9594 | 0.9652 | 0.9772 | 0.9712 | 0.9177 | 0.9453 |
| 20 | 80 | 72,835 | 0.9652 | 0.9792 | 0.9772 | 0.9782 | 0.9171 | 0.9097 |
| 10 | 90 | 64,740 | 0.9710 | 0.9904 | 0.9772 | 0.9838 | 0.9147 | 0.8171 |
| 0 | 100 | 58,270 | 0.9772 | 1.0000 | 0.9772 | 0.9885 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9175 | 0.0000 | 0.0000 | 0.0000 | 0.9175 | 1.0000 |
| 90 | 10 | 299,940 | 0.9232 | 0.5675 | 0.9773 | 0.7180 | 0.9172 | 0.9973 |
| 80 | 20 | 291,350 | 0.9290 | 0.7462 | 0.9772 | 0.8463 | 0.9169 | 0.9938 |
| 70 | 30 | 194,230 | 0.9356 | 0.8359 | 0.9772 | 0.9010 | 0.9178 | 0.9895 |
| 60 | 40 | 145,675 | 0.9414 | 0.8876 | 0.9772 | 0.9302 | 0.9175 | 0.9837 |
| 50 | 50 | 116,540 | 0.9468 | 0.9212 | 0.9772 | 0.9484 | 0.9164 | 0.9758 |
| 40 | 60 | 97,115 | 0.9537 | 0.9473 | 0.9772 | 0.9621 | 0.9185 | 0.9642 |
| 30 | 70 | 83,240 | 0.9594 | 0.9652 | 0.9772 | 0.9712 | 0.9177 | 0.9453 |
| 20 | 80 | 72,835 | 0.9652 | 0.9792 | 0.9772 | 0.9782 | 0.9171 | 0.9097 |
| 10 | 90 | 64,740 | 0.9710 | 0.9904 | 0.9772 | 0.9838 | 0.9147 | 0.8171 |
| 0 | 100 | 58,270 | 0.9772 | 1.0000 | 0.9772 | 0.9885 | 0.0000 | 0.0000 |


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
0.15       0.8487   0.4871   0.8631   0.9650   0.7186   0.3685   <--
0.20       0.8487   0.4871   0.8631   0.9650   0.7186   0.3685  
0.25       0.8487   0.4871   0.8631   0.9650   0.7186   0.3685  
0.30       0.8487   0.4871   0.8631   0.9650   0.7186   0.3685  
0.35       0.8487   0.4871   0.8631   0.9650   0.7186   0.3685  
0.40       0.8487   0.4871   0.8631   0.9650   0.7186   0.3685  
0.45       0.8487   0.4871   0.8631   0.9650   0.7186   0.3685  
0.50       0.8487   0.4871   0.8631   0.9650   0.7186   0.3685  
0.55       0.8487   0.4871   0.8631   0.9650   0.7186   0.3685  
0.60       0.8487   0.4871   0.8631   0.9650   0.7186   0.3685  
0.65       0.8487   0.4871   0.8631   0.9650   0.7186   0.3685  
0.70       0.8487   0.4871   0.8631   0.9650   0.7186   0.3685  
0.75       0.8487   0.4871   0.8631   0.9650   0.7186   0.3685  
0.80       0.8487   0.4871   0.8631   0.9650   0.7186   0.3685  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8487, F1=0.4871, Normal Recall=0.8631, Normal Precision=0.9650, Attack Recall=0.7186, Attack Precision=0.3685

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
0.15       0.8342   0.6339   0.8633   0.9244   0.7178   0.5675   <--
0.20       0.8342   0.6339   0.8633   0.9244   0.7178   0.5675  
0.25       0.8342   0.6339   0.8633   0.9244   0.7178   0.5675  
0.30       0.8342   0.6339   0.8633   0.9244   0.7178   0.5675  
0.35       0.8342   0.6339   0.8633   0.9244   0.7178   0.5675  
0.40       0.8342   0.6339   0.8633   0.9244   0.7178   0.5675  
0.45       0.8342   0.6339   0.8633   0.9244   0.7178   0.5675  
0.50       0.8342   0.6339   0.8633   0.9244   0.7178   0.5675  
0.55       0.8342   0.6339   0.8633   0.9244   0.7178   0.5675  
0.60       0.8342   0.6339   0.8633   0.9244   0.7178   0.5675  
0.65       0.8342   0.6339   0.8633   0.9244   0.7178   0.5675  
0.70       0.8342   0.6339   0.8633   0.9244   0.7178   0.5675  
0.75       0.8342   0.6339   0.8633   0.9244   0.7178   0.5675  
0.80       0.8342   0.6339   0.8633   0.9244   0.7178   0.5675  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8342, F1=0.6339, Normal Recall=0.8633, Normal Precision=0.9244, Attack Recall=0.7178, Attack Precision=0.5675

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
0.15       0.8198   0.7051   0.8636   0.8771   0.7178   0.6928   <--
0.20       0.8198   0.7051   0.8636   0.8771   0.7178   0.6928  
0.25       0.8198   0.7051   0.8636   0.8771   0.7178   0.6928  
0.30       0.8198   0.7051   0.8636   0.8771   0.7178   0.6928  
0.35       0.8198   0.7051   0.8636   0.8771   0.7178   0.6928  
0.40       0.8198   0.7051   0.8636   0.8771   0.7178   0.6928  
0.45       0.8198   0.7051   0.8636   0.8771   0.7178   0.6928  
0.50       0.8198   0.7051   0.8636   0.8771   0.7178   0.6928  
0.55       0.8198   0.7051   0.8636   0.8771   0.7178   0.6928  
0.60       0.8198   0.7051   0.8636   0.8771   0.7178   0.6928  
0.65       0.8198   0.7051   0.8636   0.8771   0.7178   0.6928  
0.70       0.8198   0.7051   0.8636   0.8771   0.7178   0.6928  
0.75       0.8198   0.7051   0.8636   0.8771   0.7178   0.6928  
0.80       0.8198   0.7051   0.8636   0.8771   0.7178   0.6928  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8198, F1=0.7051, Normal Recall=0.8636, Normal Precision=0.8771, Attack Recall=0.7178, Attack Precision=0.6928

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
0.15       0.8051   0.7466   0.8633   0.8210   0.7178   0.7778   <--
0.20       0.8051   0.7466   0.8633   0.8210   0.7178   0.7778  
0.25       0.8051   0.7466   0.8633   0.8210   0.7178   0.7778  
0.30       0.8051   0.7466   0.8633   0.8210   0.7178   0.7778  
0.35       0.8051   0.7466   0.8633   0.8210   0.7178   0.7778  
0.40       0.8051   0.7466   0.8633   0.8210   0.7178   0.7778  
0.45       0.8051   0.7466   0.8633   0.8210   0.7178   0.7778  
0.50       0.8051   0.7466   0.8633   0.8210   0.7178   0.7778  
0.55       0.8051   0.7466   0.8633   0.8210   0.7178   0.7778  
0.60       0.8051   0.7466   0.8633   0.8210   0.7178   0.7778  
0.65       0.8051   0.7466   0.8633   0.8210   0.7178   0.7778  
0.70       0.8051   0.7466   0.8633   0.8210   0.7178   0.7778  
0.75       0.8051   0.7466   0.8633   0.8210   0.7178   0.7778  
0.80       0.8051   0.7466   0.8633   0.8210   0.7178   0.7778  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8051, F1=0.7466, Normal Recall=0.8633, Normal Precision=0.8210, Attack Recall=0.7178, Attack Precision=0.7778

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
0.15       0.7904   0.7740   0.8631   0.7536   0.7178   0.8399   <--
0.20       0.7904   0.7740   0.8631   0.7536   0.7178   0.8399  
0.25       0.7904   0.7740   0.8631   0.7536   0.7178   0.8399  
0.30       0.7904   0.7740   0.8631   0.7536   0.7178   0.8399  
0.35       0.7904   0.7740   0.8631   0.7536   0.7178   0.8399  
0.40       0.7904   0.7740   0.8631   0.7536   0.7178   0.8399  
0.45       0.7904   0.7740   0.8631   0.7536   0.7178   0.8399  
0.50       0.7904   0.7740   0.8631   0.7536   0.7178   0.8399  
0.55       0.7904   0.7740   0.8631   0.7536   0.7178   0.8399  
0.60       0.7904   0.7740   0.8631   0.7536   0.7178   0.8399  
0.65       0.7904   0.7740   0.8631   0.7536   0.7178   0.8399  
0.70       0.7904   0.7740   0.8631   0.7536   0.7178   0.8399  
0.75       0.7904   0.7740   0.8631   0.7536   0.7178   0.8399  
0.80       0.7904   0.7740   0.8631   0.7536   0.7178   0.8399  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7904, F1=0.7740, Normal Recall=0.8631, Normal Precision=0.7536, Attack Recall=0.7178, Attack Precision=0.8399

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
0.15       0.9249   0.7233   0.9186   0.9978   0.9814   0.5727   <--
0.20       0.9249   0.7233   0.9186   0.9978   0.9814   0.5727  
0.25       0.9249   0.7233   0.9186   0.9978   0.9814   0.5727  
0.30       0.9249   0.7233   0.9186   0.9978   0.9814   0.5727  
0.35       0.9249   0.7233   0.9186   0.9978   0.9814   0.5727  
0.40       0.9249   0.7233   0.9186   0.9978   0.9814   0.5727  
0.45       0.9249   0.7233   0.9186   0.9978   0.9814   0.5727  
0.50       0.9249   0.7233   0.9186   0.9978   0.9814   0.5727  
0.55       0.9249   0.7233   0.9186   0.9978   0.9814   0.5727  
0.60       0.9249   0.7233   0.9186   0.9978   0.9814   0.5727  
0.65       0.9249   0.7233   0.9186   0.9978   0.9814   0.5727  
0.70       0.9249   0.7233   0.9186   0.9978   0.9814   0.5727  
0.75       0.9249   0.7233   0.9186   0.9978   0.9814   0.5727  
0.80       0.9249   0.7233   0.9186   0.9978   0.9814   0.5727  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9249, F1=0.7233, Normal Recall=0.9186, Normal Precision=0.9978, Attack Recall=0.9814, Attack Precision=0.5727

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
0.15       0.9313   0.8509   0.9190   0.9947   0.9802   0.7516   <--
0.20       0.9313   0.8509   0.9190   0.9947   0.9802   0.7516  
0.25       0.9313   0.8509   0.9190   0.9947   0.9802   0.7516  
0.30       0.9313   0.8509   0.9190   0.9947   0.9802   0.7516  
0.35       0.9313   0.8509   0.9190   0.9947   0.9802   0.7516  
0.40       0.9313   0.8509   0.9190   0.9947   0.9802   0.7516  
0.45       0.9313   0.8509   0.9190   0.9947   0.9802   0.7516  
0.50       0.9313   0.8509   0.9190   0.9947   0.9802   0.7516  
0.55       0.9313   0.8509   0.9190   0.9947   0.9802   0.7516  
0.60       0.9313   0.8509   0.9190   0.9947   0.9802   0.7516  
0.65       0.9313   0.8509   0.9190   0.9947   0.9802   0.7516  
0.70       0.9313   0.8509   0.9190   0.9947   0.9802   0.7516  
0.75       0.9313   0.8509   0.9190   0.9947   0.9802   0.7516  
0.80       0.9313   0.8509   0.9190   0.9947   0.9802   0.7516  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9313, F1=0.8509, Normal Recall=0.9190, Normal Precision=0.9947, Attack Recall=0.9802, Attack Precision=0.7516

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
0.15       0.9375   0.9040   0.9192   0.9909   0.9802   0.8387   <--
0.20       0.9375   0.9040   0.9192   0.9909   0.9802   0.8387  
0.25       0.9375   0.9040   0.9192   0.9909   0.9802   0.8387  
0.30       0.9375   0.9040   0.9192   0.9909   0.9802   0.8387  
0.35       0.9375   0.9040   0.9192   0.9909   0.9802   0.8387  
0.40       0.9375   0.9040   0.9192   0.9909   0.9802   0.8387  
0.45       0.9375   0.9040   0.9192   0.9909   0.9802   0.8387  
0.50       0.9375   0.9040   0.9192   0.9909   0.9802   0.8387  
0.55       0.9375   0.9040   0.9192   0.9909   0.9802   0.8387  
0.60       0.9375   0.9040   0.9192   0.9909   0.9802   0.8387  
0.65       0.9375   0.9040   0.9192   0.9909   0.9802   0.8387  
0.70       0.9375   0.9040   0.9192   0.9909   0.9802   0.8387  
0.75       0.9375   0.9040   0.9192   0.9909   0.9802   0.8387  
0.80       0.9375   0.9040   0.9192   0.9909   0.9802   0.8387  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9375, F1=0.9040, Normal Recall=0.9192, Normal Precision=0.9909, Attack Recall=0.9802, Attack Precision=0.8387

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
0.15       0.9436   0.9329   0.9191   0.9859   0.9802   0.8899   <--
0.20       0.9436   0.9329   0.9191   0.9859   0.9802   0.8899  
0.25       0.9436   0.9329   0.9191   0.9859   0.9802   0.8899  
0.30       0.9436   0.9329   0.9191   0.9859   0.9802   0.8899  
0.35       0.9436   0.9329   0.9191   0.9859   0.9802   0.8899  
0.40       0.9436   0.9329   0.9191   0.9859   0.9802   0.8899  
0.45       0.9436   0.9329   0.9191   0.9859   0.9802   0.8899  
0.50       0.9436   0.9329   0.9191   0.9859   0.9802   0.8899  
0.55       0.9436   0.9329   0.9191   0.9859   0.9802   0.8899  
0.60       0.9436   0.9329   0.9191   0.9859   0.9802   0.8899  
0.65       0.9436   0.9329   0.9191   0.9859   0.9802   0.8899  
0.70       0.9436   0.9329   0.9191   0.9859   0.9802   0.8899  
0.75       0.9436   0.9329   0.9191   0.9859   0.9802   0.8899  
0.80       0.9436   0.9329   0.9191   0.9859   0.9802   0.8899  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9436, F1=0.9329, Normal Recall=0.9191, Normal Precision=0.9859, Attack Recall=0.9802, Attack Precision=0.8899

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
0.15       0.9497   0.9512   0.9191   0.9790   0.9802   0.9238   <--
0.20       0.9497   0.9512   0.9191   0.9790   0.9802   0.9238  
0.25       0.9497   0.9512   0.9191   0.9790   0.9802   0.9238  
0.30       0.9497   0.9512   0.9191   0.9790   0.9802   0.9238  
0.35       0.9497   0.9512   0.9191   0.9790   0.9802   0.9238  
0.40       0.9497   0.9512   0.9191   0.9790   0.9802   0.9238  
0.45       0.9497   0.9512   0.9191   0.9790   0.9802   0.9238  
0.50       0.9497   0.9512   0.9191   0.9790   0.9802   0.9238  
0.55       0.9497   0.9512   0.9191   0.9790   0.9802   0.9238  
0.60       0.9497   0.9512   0.9191   0.9790   0.9802   0.9238  
0.65       0.9497   0.9512   0.9191   0.9790   0.9802   0.9238  
0.70       0.9497   0.9512   0.9191   0.9790   0.9802   0.9238  
0.75       0.9497   0.9512   0.9191   0.9790   0.9802   0.9238  
0.80       0.9497   0.9512   0.9191   0.9790   0.9802   0.9238  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9497, F1=0.9512, Normal Recall=0.9191, Normal Precision=0.9790, Attack Recall=0.9802, Attack Precision=0.9238

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
0.15       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677   <--
0.20       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.25       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.30       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.35       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.40       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.45       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.50       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.55       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.60       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.65       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.70       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.75       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.80       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9233, F1=0.7184, Normal Recall=0.9172, Normal Precision=0.9974, Attack Recall=0.9781, Attack Precision=0.5677

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
0.15       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479   <--
0.20       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.25       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.30       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.35       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.40       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.45       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.50       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.55       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.60       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.65       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.70       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.75       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.80       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9296, F1=0.8473, Normal Recall=0.9177, Normal Precision=0.9938, Attack Recall=0.9772, Attack Precision=0.7479

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
0.15       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361   <--
0.20       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.25       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.30       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.35       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.40       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.45       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.50       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.55       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.60       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.65       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.70       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.75       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.80       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9357, F1=0.9012, Normal Recall=0.9179, Normal Precision=0.9895, Attack Recall=0.9772, Attack Precision=0.8361

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
0.15       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880   <--
0.20       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.25       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.30       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.35       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.40       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.45       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.50       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.55       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.60       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.65       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.70       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.75       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.80       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9416, F1=0.9305, Normal Recall=0.9178, Normal Precision=0.9837, Attack Recall=0.9772, Attack Precision=0.8880

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
0.15       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222   <--
0.20       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.25       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.30       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.35       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.40       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.45       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.50       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.55       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.60       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.65       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.70       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.75       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.80       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9474, F1=0.9489, Normal Recall=0.9175, Normal Precision=0.9758, Attack Recall=0.9772, Attack Precision=0.9222

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
0.15       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677   <--
0.20       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.25       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.30       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.35       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.40       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.45       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.50       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.55       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.60       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.65       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.70       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.75       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
0.80       0.9233   0.7184   0.9172   0.9974   0.9781   0.5677  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9233, F1=0.7184, Normal Recall=0.9172, Normal Precision=0.9974, Attack Recall=0.9781, Attack Precision=0.5677

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
0.15       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479   <--
0.20       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.25       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.30       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.35       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.40       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.45       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.50       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.55       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.60       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.65       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.70       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.75       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
0.80       0.9296   0.8473   0.9177   0.9938   0.9772   0.7479  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9296, F1=0.8473, Normal Recall=0.9177, Normal Precision=0.9938, Attack Recall=0.9772, Attack Precision=0.7479

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
0.15       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361   <--
0.20       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.25       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.30       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.35       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.40       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.45       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.50       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.55       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.60       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.65       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.70       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.75       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
0.80       0.9357   0.9012   0.9179   0.9895   0.9772   0.8361  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9357, F1=0.9012, Normal Recall=0.9179, Normal Precision=0.9895, Attack Recall=0.9772, Attack Precision=0.8361

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
0.15       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880   <--
0.20       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.25       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.30       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.35       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.40       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.45       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.50       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.55       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.60       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.65       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.70       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.75       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
0.80       0.9416   0.9305   0.9178   0.9837   0.9772   0.8880  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9416, F1=0.9305, Normal Recall=0.9178, Normal Precision=0.9837, Attack Recall=0.9772, Attack Precision=0.8880

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
0.15       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222   <--
0.20       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.25       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.30       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.35       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.40       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.45       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.50       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.55       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.60       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.65       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.70       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.75       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
0.80       0.9474   0.9489   0.9175   0.9758   0.9772   0.9222  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9474, F1=0.9489, Normal Recall=0.9175, Normal Precision=0.9758, Attack Recall=0.9772, Attack Precision=0.9222

```

