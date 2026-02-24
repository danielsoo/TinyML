# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e2_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-20 01:28:02 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 2 |
| **Batch size** | 64 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3170 | 0.3679 | 0.4176 | 0.4672 | 0.5178 | 0.5676 | 0.6174 | 0.6657 | 0.7177 | 0.7669 | 0.8162 |
| QAT+Prune only | 0.8341 | 0.8390 | 0.8426 | 0.8473 | 0.8511 | 0.8551 | 0.8589 | 0.8632 | 0.8682 | 0.8707 | 0.8757 |
| QAT+PTQ | 0.8333 | 0.8383 | 0.8421 | 0.8470 | 0.8509 | 0.8550 | 0.8591 | 0.8638 | 0.8689 | 0.8714 | 0.8767 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8333 | 0.8383 | 0.8421 | 0.8470 | 0.8509 | 0.8550 | 0.8591 | 0.8638 | 0.8689 | 0.8714 | 0.8767 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2052 | 0.3592 | 0.4789 | 0.5752 | 0.6537 | 0.7191 | 0.7737 | 0.8222 | 0.8630 | 0.8988 |
| QAT+Prune only | 0.0000 | 0.5213 | 0.6900 | 0.7749 | 0.8247 | 0.8580 | 0.8816 | 0.8996 | 0.9140 | 0.9242 | 0.9338 |
| QAT+PTQ | 0.0000 | 0.5205 | 0.6895 | 0.7746 | 0.8247 | 0.8581 | 0.8819 | 0.9001 | 0.9145 | 0.9246 | 0.9343 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5205 | 0.6895 | 0.7746 | 0.8247 | 0.8581 | 0.8819 | 0.9001 | 0.9145 | 0.9246 | 0.9343 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3170 | 0.3181 | 0.3180 | 0.3177 | 0.3190 | 0.3191 | 0.3191 | 0.3147 | 0.3237 | 0.3231 | 0.0000 |
| QAT+Prune only | 0.8341 | 0.8349 | 0.8344 | 0.8352 | 0.8346 | 0.8345 | 0.8337 | 0.8339 | 0.8383 | 0.8256 | 0.0000 |
| QAT+PTQ | 0.8333 | 0.8340 | 0.8335 | 0.8343 | 0.8338 | 0.8333 | 0.8328 | 0.8337 | 0.8378 | 0.8239 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8333 | 0.8340 | 0.8335 | 0.8343 | 0.8338 | 0.8333 | 0.8328 | 0.8337 | 0.8378 | 0.8239 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3170 | 0.0000 | 0.0000 | 0.0000 | 0.3170 | 1.0000 |
| 90 | 10 | 299,940 | 0.3679 | 0.1173 | 0.8158 | 0.2052 | 0.3181 | 0.9396 |
| 80 | 20 | 291,350 | 0.4176 | 0.2303 | 0.8162 | 0.3592 | 0.3180 | 0.8737 |
| 70 | 30 | 194,230 | 0.4672 | 0.3389 | 0.8162 | 0.4789 | 0.3177 | 0.8013 |
| 60 | 40 | 145,675 | 0.5178 | 0.4441 | 0.8162 | 0.5752 | 0.3190 | 0.7224 |
| 50 | 50 | 116,540 | 0.5676 | 0.5452 | 0.8162 | 0.6537 | 0.3191 | 0.6345 |
| 40 | 60 | 97,115 | 0.6174 | 0.6426 | 0.8162 | 0.7191 | 0.3191 | 0.5365 |
| 30 | 70 | 83,240 | 0.6657 | 0.7354 | 0.8162 | 0.7737 | 0.3147 | 0.4232 |
| 20 | 80 | 72,835 | 0.7177 | 0.8284 | 0.8162 | 0.8222 | 0.3237 | 0.3057 |
| 10 | 90 | 64,740 | 0.7669 | 0.9156 | 0.8162 | 0.8630 | 0.3231 | 0.1634 |
| 0 | 100 | 58,270 | 0.8162 | 1.0000 | 0.8162 | 0.8988 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8341 | 0.0000 | 0.0000 | 0.0000 | 0.8341 | 1.0000 |
| 90 | 10 | 299,940 | 0.8390 | 0.3710 | 0.8763 | 0.5213 | 0.8349 | 0.9838 |
| 80 | 20 | 291,350 | 0.8426 | 0.5693 | 0.8757 | 0.6900 | 0.8344 | 0.9641 |
| 70 | 30 | 194,230 | 0.8473 | 0.6948 | 0.8757 | 0.7749 | 0.8352 | 0.9401 |
| 60 | 40 | 145,675 | 0.8511 | 0.7793 | 0.8757 | 0.8247 | 0.8346 | 0.9097 |
| 50 | 50 | 116,540 | 0.8551 | 0.8411 | 0.8757 | 0.8580 | 0.8345 | 0.8704 |
| 40 | 60 | 97,115 | 0.8589 | 0.8876 | 0.8757 | 0.8816 | 0.8337 | 0.8173 |
| 30 | 70 | 83,240 | 0.8632 | 0.9248 | 0.8757 | 0.8996 | 0.8339 | 0.7420 |
| 20 | 80 | 72,835 | 0.8682 | 0.9559 | 0.8757 | 0.9140 | 0.8383 | 0.6278 |
| 10 | 90 | 64,740 | 0.8707 | 0.9784 | 0.8757 | 0.9242 | 0.8256 | 0.4247 |
| 0 | 100 | 58,270 | 0.8757 | 1.0000 | 0.8757 | 0.9338 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8333 | 0.0000 | 0.0000 | 0.0000 | 0.8333 | 1.0000 |
| 90 | 10 | 299,940 | 0.8383 | 0.3700 | 0.8773 | 0.5205 | 0.8340 | 0.9839 |
| 80 | 20 | 291,350 | 0.8421 | 0.5682 | 0.8767 | 0.6895 | 0.8335 | 0.9643 |
| 70 | 30 | 194,230 | 0.8470 | 0.6939 | 0.8767 | 0.7746 | 0.8343 | 0.9404 |
| 60 | 40 | 145,675 | 0.8509 | 0.7786 | 0.8767 | 0.8247 | 0.8338 | 0.9102 |
| 50 | 50 | 116,540 | 0.8550 | 0.8402 | 0.8767 | 0.8581 | 0.8333 | 0.8711 |
| 40 | 60 | 97,115 | 0.8591 | 0.8872 | 0.8767 | 0.8819 | 0.8328 | 0.8182 |
| 30 | 70 | 83,240 | 0.8638 | 0.9248 | 0.8767 | 0.9001 | 0.8337 | 0.7434 |
| 20 | 80 | 72,835 | 0.8689 | 0.9558 | 0.8767 | 0.9145 | 0.8378 | 0.6294 |
| 10 | 90 | 64,740 | 0.8714 | 0.9782 | 0.8767 | 0.9246 | 0.8239 | 0.4260 |
| 0 | 100 | 58,270 | 0.8767 | 1.0000 | 0.8767 | 0.9343 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8333 | 0.0000 | 0.0000 | 0.0000 | 0.8333 | 1.0000 |
| 90 | 10 | 299,940 | 0.8383 | 0.3700 | 0.8773 | 0.5205 | 0.8340 | 0.9839 |
| 80 | 20 | 291,350 | 0.8421 | 0.5682 | 0.8767 | 0.6895 | 0.8335 | 0.9643 |
| 70 | 30 | 194,230 | 0.8470 | 0.6939 | 0.8767 | 0.7746 | 0.8343 | 0.9404 |
| 60 | 40 | 145,675 | 0.8509 | 0.7786 | 0.8767 | 0.8247 | 0.8338 | 0.9102 |
| 50 | 50 | 116,540 | 0.8550 | 0.8402 | 0.8767 | 0.8581 | 0.8333 | 0.8711 |
| 40 | 60 | 97,115 | 0.8591 | 0.8872 | 0.8767 | 0.8819 | 0.8328 | 0.8182 |
| 30 | 70 | 83,240 | 0.8638 | 0.9248 | 0.8767 | 0.9001 | 0.8337 | 0.7434 |
| 20 | 80 | 72,835 | 0.8689 | 0.9558 | 0.8767 | 0.9145 | 0.8378 | 0.6294 |
| 10 | 90 | 64,740 | 0.8714 | 0.9782 | 0.8767 | 0.9246 | 0.8239 | 0.4260 |
| 0 | 100 | 58,270 | 0.8767 | 1.0000 | 0.8767 | 0.9343 | 0.0000 | 0.0000 |


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
0.15       0.3680   0.2054   0.3181   0.9399   0.8170   0.1175   <--
0.20       0.3680   0.2054   0.3181   0.9399   0.8170   0.1175  
0.25       0.3680   0.2054   0.3181   0.9399   0.8170   0.1175  
0.30       0.3680   0.2054   0.3181   0.9399   0.8170   0.1175  
0.35       0.3680   0.2054   0.3181   0.9399   0.8170   0.1175  
0.40       0.3680   0.2054   0.3181   0.9399   0.8170   0.1175  
0.45       0.3680   0.2054   0.3181   0.9399   0.8170   0.1175  
0.50       0.3680   0.2054   0.3181   0.9399   0.8170   0.1175  
0.55       0.3680   0.2054   0.3181   0.9399   0.8170   0.1175  
0.60       0.3680   0.2054   0.3181   0.9399   0.8170   0.1175  
0.65       0.3680   0.2054   0.3181   0.9399   0.8170   0.1175  
0.70       0.3680   0.2054   0.3181   0.9399   0.8170   0.1175  
0.75       0.3680   0.2054   0.3181   0.9399   0.8170   0.1175  
0.80       0.3680   0.2054   0.3181   0.9399   0.8170   0.1175  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3680, F1=0.2054, Normal Recall=0.3181, Normal Precision=0.9399, Attack Recall=0.8170, Attack Precision=0.1175

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
0.15       0.4178   0.3593   0.3182   0.8738   0.8162   0.2303   <--
0.20       0.4178   0.3593   0.3182   0.8738   0.8162   0.2303  
0.25       0.4178   0.3593   0.3182   0.8738   0.8162   0.2303  
0.30       0.4178   0.3593   0.3182   0.8738   0.8162   0.2303  
0.35       0.4178   0.3593   0.3182   0.8738   0.8162   0.2303  
0.40       0.4178   0.3593   0.3182   0.8738   0.8162   0.2303  
0.45       0.4178   0.3593   0.3182   0.8738   0.8162   0.2303  
0.50       0.4178   0.3593   0.3182   0.8738   0.8162   0.2303  
0.55       0.4178   0.3593   0.3182   0.8738   0.8162   0.2303  
0.60       0.4178   0.3593   0.3182   0.8738   0.8162   0.2303  
0.65       0.4178   0.3593   0.3182   0.8738   0.8162   0.2303  
0.70       0.4178   0.3593   0.3182   0.8738   0.8162   0.2303  
0.75       0.4178   0.3593   0.3182   0.8738   0.8162   0.2303  
0.80       0.4178   0.3593   0.3182   0.8738   0.8162   0.2303  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4178, F1=0.3593, Normal Recall=0.3182, Normal Precision=0.8738, Attack Recall=0.8162, Attack Precision=0.2303

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
0.15       0.4670   0.4788   0.3174   0.8011   0.8162   0.3388   <--
0.20       0.4670   0.4788   0.3174   0.8011   0.8162   0.3388  
0.25       0.4670   0.4788   0.3174   0.8011   0.8162   0.3388  
0.30       0.4670   0.4788   0.3174   0.8011   0.8162   0.3388  
0.35       0.4670   0.4788   0.3174   0.8011   0.8162   0.3388  
0.40       0.4670   0.4788   0.3174   0.8011   0.8162   0.3388  
0.45       0.4670   0.4788   0.3174   0.8011   0.8162   0.3388  
0.50       0.4670   0.4788   0.3174   0.8011   0.8162   0.3388  
0.55       0.4670   0.4788   0.3174   0.8011   0.8162   0.3388  
0.60       0.4670   0.4788   0.3174   0.8011   0.8162   0.3388  
0.65       0.4670   0.4788   0.3174   0.8011   0.8162   0.3388  
0.70       0.4670   0.4788   0.3174   0.8011   0.8162   0.3388  
0.75       0.4670   0.4788   0.3174   0.8011   0.8162   0.3388  
0.80       0.4670   0.4788   0.3174   0.8011   0.8162   0.3388  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4670, F1=0.4788, Normal Recall=0.3174, Normal Precision=0.8011, Attack Recall=0.8162, Attack Precision=0.3388

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
0.15       0.5173   0.5749   0.3180   0.7218   0.8162   0.4438   <--
0.20       0.5173   0.5749   0.3180   0.7218   0.8162   0.4438  
0.25       0.5173   0.5749   0.3180   0.7218   0.8162   0.4438  
0.30       0.5173   0.5749   0.3180   0.7218   0.8162   0.4438  
0.35       0.5173   0.5749   0.3180   0.7218   0.8162   0.4438  
0.40       0.5173   0.5749   0.3180   0.7218   0.8162   0.4438  
0.45       0.5173   0.5749   0.3180   0.7218   0.8162   0.4438  
0.50       0.5173   0.5749   0.3180   0.7218   0.8162   0.4438  
0.55       0.5173   0.5749   0.3180   0.7218   0.8162   0.4438  
0.60       0.5173   0.5749   0.3180   0.7218   0.8162   0.4438  
0.65       0.5173   0.5749   0.3180   0.7218   0.8162   0.4438  
0.70       0.5173   0.5749   0.3180   0.7218   0.8162   0.4438  
0.75       0.5173   0.5749   0.3180   0.7218   0.8162   0.4438  
0.80       0.5173   0.5749   0.3180   0.7218   0.8162   0.4438  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5173, F1=0.5749, Normal Recall=0.3180, Normal Precision=0.7218, Attack Recall=0.8162, Attack Precision=0.4438

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
0.15       0.5667   0.6532   0.3172   0.6331   0.8162   0.5445   <--
0.20       0.5667   0.6532   0.3172   0.6331   0.8162   0.5445  
0.25       0.5667   0.6532   0.3172   0.6331   0.8162   0.5445  
0.30       0.5667   0.6532   0.3172   0.6331   0.8162   0.5445  
0.35       0.5667   0.6532   0.3172   0.6331   0.8162   0.5445  
0.40       0.5667   0.6532   0.3172   0.6331   0.8162   0.5445  
0.45       0.5667   0.6532   0.3172   0.6331   0.8162   0.5445  
0.50       0.5667   0.6532   0.3172   0.6331   0.8162   0.5445  
0.55       0.5667   0.6532   0.3172   0.6331   0.8162   0.5445  
0.60       0.5667   0.6532   0.3172   0.6331   0.8162   0.5445  
0.65       0.5667   0.6532   0.3172   0.6331   0.8162   0.5445  
0.70       0.5667   0.6532   0.3172   0.6331   0.8162   0.5445  
0.75       0.5667   0.6532   0.3172   0.6331   0.8162   0.5445  
0.80       0.5667   0.6532   0.3172   0.6331   0.8162   0.5445  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5667, F1=0.6532, Normal Recall=0.3172, Normal Precision=0.6331, Attack Recall=0.8162, Attack Precision=0.5445

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
0.15       0.8390   0.5213   0.8349   0.9838   0.8763   0.3710   <--
0.20       0.8390   0.5213   0.8349   0.9838   0.8763   0.3710  
0.25       0.8390   0.5213   0.8349   0.9838   0.8763   0.3710  
0.30       0.8390   0.5213   0.8349   0.9838   0.8763   0.3710  
0.35       0.8390   0.5213   0.8349   0.9838   0.8763   0.3710  
0.40       0.8390   0.5213   0.8349   0.9838   0.8763   0.3710  
0.45       0.8390   0.5213   0.8349   0.9838   0.8763   0.3710  
0.50       0.8390   0.5213   0.8349   0.9838   0.8763   0.3710  
0.55       0.8390   0.5213   0.8349   0.9838   0.8763   0.3710  
0.60       0.8390   0.5213   0.8349   0.9838   0.8763   0.3710  
0.65       0.8390   0.5213   0.8349   0.9838   0.8763   0.3710  
0.70       0.8390   0.5213   0.8349   0.9838   0.8763   0.3710  
0.75       0.8390   0.5213   0.8349   0.9838   0.8763   0.3710  
0.80       0.8390   0.5213   0.8349   0.9838   0.8763   0.3710  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8390, F1=0.5213, Normal Recall=0.8349, Normal Precision=0.9838, Attack Recall=0.8763, Attack Precision=0.3710

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
0.15       0.8434   0.6910   0.8353   0.9641   0.8757   0.5707   <--
0.20       0.8434   0.6910   0.8353   0.9641   0.8757   0.5707  
0.25       0.8434   0.6910   0.8353   0.9641   0.8757   0.5707  
0.30       0.8434   0.6910   0.8353   0.9641   0.8757   0.5707  
0.35       0.8434   0.6910   0.8353   0.9641   0.8757   0.5707  
0.40       0.8434   0.6910   0.8353   0.9641   0.8757   0.5707  
0.45       0.8434   0.6910   0.8353   0.9641   0.8757   0.5707  
0.50       0.8434   0.6910   0.8353   0.9641   0.8757   0.5707  
0.55       0.8434   0.6910   0.8353   0.9641   0.8757   0.5707  
0.60       0.8434   0.6910   0.8353   0.9641   0.8757   0.5707  
0.65       0.8434   0.6910   0.8353   0.9641   0.8757   0.5707  
0.70       0.8434   0.6910   0.8353   0.9641   0.8757   0.5707  
0.75       0.8434   0.6910   0.8353   0.9641   0.8757   0.5707  
0.80       0.8434   0.6910   0.8353   0.9641   0.8757   0.5707  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8434, F1=0.6910, Normal Recall=0.8353, Normal Precision=0.9641, Attack Recall=0.8757, Attack Precision=0.5707

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
0.15       0.8470   0.7744   0.8346   0.9400   0.8757   0.6942   <--
0.20       0.8470   0.7744   0.8346   0.9400   0.8757   0.6942  
0.25       0.8470   0.7744   0.8346   0.9400   0.8757   0.6942  
0.30       0.8470   0.7744   0.8346   0.9400   0.8757   0.6942  
0.35       0.8470   0.7744   0.8346   0.9400   0.8757   0.6942  
0.40       0.8470   0.7744   0.8346   0.9400   0.8757   0.6942  
0.45       0.8470   0.7744   0.8346   0.9400   0.8757   0.6942  
0.50       0.8470   0.7744   0.8346   0.9400   0.8757   0.6942  
0.55       0.8470   0.7744   0.8346   0.9400   0.8757   0.6942  
0.60       0.8470   0.7744   0.8346   0.9400   0.8757   0.6942  
0.65       0.8470   0.7744   0.8346   0.9400   0.8757   0.6942  
0.70       0.8470   0.7744   0.8346   0.9400   0.8757   0.6942  
0.75       0.8470   0.7744   0.8346   0.9400   0.8757   0.6942  
0.80       0.8470   0.7744   0.8346   0.9400   0.8757   0.6942  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8470, F1=0.7744, Normal Recall=0.8346, Normal Precision=0.9400, Attack Recall=0.8757, Attack Precision=0.6942

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
0.15       0.8506   0.8242   0.8338   0.9096   0.8757   0.7784   <--
0.20       0.8506   0.8242   0.8338   0.9096   0.8757   0.7784  
0.25       0.8506   0.8242   0.8338   0.9096   0.8757   0.7784  
0.30       0.8506   0.8242   0.8338   0.9096   0.8757   0.7784  
0.35       0.8506   0.8242   0.8338   0.9096   0.8757   0.7784  
0.40       0.8506   0.8242   0.8338   0.9096   0.8757   0.7784  
0.45       0.8506   0.8242   0.8338   0.9096   0.8757   0.7784  
0.50       0.8506   0.8242   0.8338   0.9096   0.8757   0.7784  
0.55       0.8506   0.8242   0.8338   0.9096   0.8757   0.7784  
0.60       0.8506   0.8242   0.8338   0.9096   0.8757   0.7784  
0.65       0.8506   0.8242   0.8338   0.9096   0.8757   0.7784  
0.70       0.8506   0.8242   0.8338   0.9096   0.8757   0.7784  
0.75       0.8506   0.8242   0.8338   0.9096   0.8757   0.7784  
0.80       0.8506   0.8242   0.8338   0.9096   0.8757   0.7784  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8506, F1=0.8242, Normal Recall=0.8338, Normal Precision=0.9096, Attack Recall=0.8757, Attack Precision=0.7784

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
0.15       0.8540   0.8571   0.8322   0.8701   0.8757   0.8392   <--
0.20       0.8540   0.8571   0.8322   0.8701   0.8757   0.8392  
0.25       0.8540   0.8571   0.8322   0.8701   0.8757   0.8392  
0.30       0.8540   0.8571   0.8322   0.8701   0.8757   0.8392  
0.35       0.8540   0.8571   0.8322   0.8701   0.8757   0.8392  
0.40       0.8540   0.8571   0.8322   0.8701   0.8757   0.8392  
0.45       0.8540   0.8571   0.8322   0.8701   0.8757   0.8392  
0.50       0.8540   0.8571   0.8322   0.8701   0.8757   0.8392  
0.55       0.8540   0.8571   0.8322   0.8701   0.8757   0.8392  
0.60       0.8540   0.8571   0.8322   0.8701   0.8757   0.8392  
0.65       0.8540   0.8571   0.8322   0.8701   0.8757   0.8392  
0.70       0.8540   0.8571   0.8322   0.8701   0.8757   0.8392  
0.75       0.8540   0.8571   0.8322   0.8701   0.8757   0.8392  
0.80       0.8540   0.8571   0.8322   0.8701   0.8757   0.8392  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8540, F1=0.8571, Normal Recall=0.8322, Normal Precision=0.8701, Attack Recall=0.8757, Attack Precision=0.8392

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
0.15       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699   <--
0.20       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.25       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.30       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.35       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.40       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.45       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.50       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.55       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.60       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.65       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.70       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.75       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.80       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8383, F1=0.5204, Normal Recall=0.8340, Normal Precision=0.9839, Attack Recall=0.8772, Attack Precision=0.3699

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
0.15       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697   <--
0.20       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.25       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.30       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.35       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.40       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.45       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.50       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.55       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.60       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.65       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.70       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.75       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.80       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8429, F1=0.6906, Normal Recall=0.8344, Normal Precision=0.9644, Attack Recall=0.8767, Attack Precision=0.5697

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
0.15       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932   <--
0.20       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.25       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.30       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.35       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.40       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.45       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.50       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.55       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.60       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.65       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.70       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.75       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.80       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8466, F1=0.7742, Normal Recall=0.8337, Normal Precision=0.9404, Attack Recall=0.8767, Attack Precision=0.6932

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
0.15       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777   <--
0.20       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.25       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.30       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.35       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.40       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.45       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.50       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.55       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.60       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.65       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.70       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.75       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.80       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8505, F1=0.8242, Normal Recall=0.8330, Normal Precision=0.9102, Attack Recall=0.8767, Attack Precision=0.7777

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
0.15       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386   <--
0.20       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.25       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.30       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.35       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.40       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.45       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.50       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.55       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.60       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.65       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.70       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.75       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.80       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8539, F1=0.8572, Normal Recall=0.8312, Normal Precision=0.8708, Attack Recall=0.8767, Attack Precision=0.8386

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
0.15       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699   <--
0.20       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.25       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.30       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.35       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.40       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.45       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.50       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.55       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.60       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.65       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.70       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.75       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
0.80       0.8383   0.5204   0.8340   0.9839   0.8772   0.3699  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8383, F1=0.5204, Normal Recall=0.8340, Normal Precision=0.9839, Attack Recall=0.8772, Attack Precision=0.3699

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
0.15       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697   <--
0.20       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.25       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.30       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.35       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.40       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.45       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.50       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.55       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.60       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.65       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.70       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.75       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
0.80       0.8429   0.6906   0.8344   0.9644   0.8767   0.5697  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8429, F1=0.6906, Normal Recall=0.8344, Normal Precision=0.9644, Attack Recall=0.8767, Attack Precision=0.5697

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
0.15       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932   <--
0.20       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.25       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.30       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.35       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.40       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.45       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.50       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.55       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.60       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.65       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.70       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.75       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
0.80       0.8466   0.7742   0.8337   0.9404   0.8767   0.6932  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8466, F1=0.7742, Normal Recall=0.8337, Normal Precision=0.9404, Attack Recall=0.8767, Attack Precision=0.6932

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
0.15       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777   <--
0.20       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.25       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.30       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.35       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.40       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.45       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.50       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.55       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.60       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.65       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.70       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.75       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
0.80       0.8505   0.8242   0.8330   0.9102   0.8767   0.7777  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8505, F1=0.8242, Normal Recall=0.8330, Normal Precision=0.9102, Attack Recall=0.8767, Attack Precision=0.7777

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
0.15       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386   <--
0.20       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.25       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.30       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.35       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.40       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.45       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.50       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.55       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.60       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.65       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.70       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.75       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
0.80       0.8539   0.8572   0.8312   0.8708   0.8767   0.8386  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8539, F1=0.8572, Normal Recall=0.8312, Normal Precision=0.8708, Attack Recall=0.8767, Attack Precision=0.8386

```

