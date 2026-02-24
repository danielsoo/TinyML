# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e2_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-22 15:10:20 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9276 | 0.9342 | 0.9401 | 0.9469 | 0.9528 | 0.9588 | 0.9654 | 0.9720 | 0.9779 | 0.9839 | 0.9902 |
| QAT+Prune only | 0.0365 | 0.1323 | 0.2285 | 0.3254 | 0.4215 | 0.5178 | 0.6140 | 0.7107 | 0.8070 | 0.9038 | 1.0000 |
| QAT+PTQ | 0.0359 | 0.1318 | 0.2280 | 0.3250 | 0.4211 | 0.5176 | 0.6137 | 0.7105 | 0.8069 | 0.9037 | 1.0000 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.0359 | 0.1318 | 0.2280 | 0.3250 | 0.4211 | 0.5176 | 0.6137 | 0.7105 | 0.8069 | 0.9037 | 1.0000 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.7505 | 0.8686 | 0.9180 | 0.9437 | 0.9600 | 0.9717 | 0.9802 | 0.9862 | 0.9910 | 0.9951 |
| QAT+Prune only | 0.0000 | 0.1873 | 0.3415 | 0.4708 | 0.5803 | 0.6747 | 0.7566 | 0.8287 | 0.8924 | 0.9493 | 1.0000 |
| QAT+PTQ | 0.0000 | 0.1872 | 0.3413 | 0.4706 | 0.5802 | 0.6746 | 0.7565 | 0.8287 | 0.8923 | 0.9492 | 1.0000 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.1872 | 0.3413 | 0.4706 | 0.5802 | 0.6746 | 0.7565 | 0.8287 | 0.8923 | 0.9492 | 1.0000 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9276 | 0.9280 | 0.9275 | 0.9284 | 0.9278 | 0.9273 | 0.9281 | 0.9294 | 0.9285 | 0.9271 | 0.0000 |
| QAT+Prune only | 0.0365 | 0.0359 | 0.0357 | 0.0363 | 0.0358 | 0.0355 | 0.0351 | 0.0357 | 0.0351 | 0.0377 | 0.0000 |
| QAT+PTQ | 0.0359 | 0.0353 | 0.0350 | 0.0357 | 0.0352 | 0.0351 | 0.0343 | 0.0352 | 0.0344 | 0.0372 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.0359 | 0.0353 | 0.0350 | 0.0357 | 0.0352 | 0.0351 | 0.0343 | 0.0352 | 0.0344 | 0.0372 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9276 | 0.0000 | 0.0000 | 0.0000 | 0.9276 | 1.0000 |
| 90 | 10 | 299,940 | 0.9342 | 0.6043 | 0.9900 | 0.7505 | 0.9280 | 0.9988 |
| 80 | 20 | 291,350 | 0.9401 | 0.7736 | 0.9902 | 0.8686 | 0.9275 | 0.9974 |
| 70 | 30 | 194,230 | 0.9469 | 0.8556 | 0.9902 | 0.9180 | 0.9284 | 0.9955 |
| 60 | 40 | 145,675 | 0.9528 | 0.9014 | 0.9902 | 0.9437 | 0.9278 | 0.9930 |
| 50 | 50 | 116,540 | 0.9588 | 0.9316 | 0.9902 | 0.9600 | 0.9273 | 0.9895 |
| 40 | 60 | 97,115 | 0.9654 | 0.9538 | 0.9902 | 0.9717 | 0.9281 | 0.9844 |
| 30 | 70 | 83,240 | 0.9720 | 0.9704 | 0.9902 | 0.9802 | 0.9294 | 0.9759 |
| 20 | 80 | 72,835 | 0.9779 | 0.9823 | 0.9902 | 0.9862 | 0.9285 | 0.9594 |
| 10 | 90 | 64,740 | 0.9839 | 0.9919 | 0.9902 | 0.9910 | 0.9271 | 0.9130 |
| 0 | 100 | 58,270 | 0.9902 | 1.0000 | 0.9902 | 0.9951 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0365 | 0.0000 | 0.0000 | 0.0000 | 0.0365 | 1.0000 |
| 90 | 10 | 299,940 | 0.1323 | 0.1033 | 1.0000 | 0.1873 | 0.0359 | 1.0000 |
| 80 | 20 | 291,350 | 0.2285 | 0.2059 | 1.0000 | 0.3415 | 0.0357 | 1.0000 |
| 70 | 30 | 194,230 | 0.3254 | 0.3078 | 1.0000 | 0.4708 | 0.0363 | 1.0000 |
| 60 | 40 | 145,675 | 0.4215 | 0.4088 | 1.0000 | 0.5803 | 0.0358 | 1.0000 |
| 50 | 50 | 116,540 | 0.5178 | 0.5090 | 1.0000 | 0.6747 | 0.0355 | 1.0000 |
| 40 | 60 | 97,115 | 0.6140 | 0.6085 | 1.0000 | 0.7566 | 0.0351 | 1.0000 |
| 30 | 70 | 83,240 | 0.7107 | 0.7076 | 1.0000 | 0.8287 | 0.0357 | 1.0000 |
| 20 | 80 | 72,835 | 0.8070 | 0.8057 | 1.0000 | 0.8924 | 0.0351 | 1.0000 |
| 10 | 90 | 64,740 | 0.9038 | 0.9034 | 1.0000 | 0.9493 | 0.0377 | 1.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0359 | 0.0000 | 0.0000 | 0.0000 | 0.0359 | 1.0000 |
| 90 | 10 | 299,940 | 0.1318 | 0.1033 | 1.0000 | 0.1872 | 0.0353 | 1.0000 |
| 80 | 20 | 291,350 | 0.2280 | 0.2058 | 1.0000 | 0.3413 | 0.0350 | 1.0000 |
| 70 | 30 | 194,230 | 0.3250 | 0.3077 | 1.0000 | 0.4706 | 0.0357 | 1.0000 |
| 60 | 40 | 145,675 | 0.4211 | 0.4086 | 1.0000 | 0.5802 | 0.0352 | 1.0000 |
| 50 | 50 | 116,540 | 0.5176 | 0.5089 | 1.0000 | 0.6746 | 0.0351 | 1.0000 |
| 40 | 60 | 97,115 | 0.6137 | 0.6084 | 1.0000 | 0.7565 | 0.0343 | 1.0000 |
| 30 | 70 | 83,240 | 0.7105 | 0.7075 | 1.0000 | 0.8287 | 0.0352 | 1.0000 |
| 20 | 80 | 72,835 | 0.8069 | 0.8055 | 1.0000 | 0.8923 | 0.0344 | 1.0000 |
| 10 | 90 | 64,740 | 0.9037 | 0.9034 | 1.0000 | 0.9492 | 0.0372 | 1.0000 |
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
| 100 | 0 | 100,000 | 0.0359 | 0.0000 | 0.0000 | 0.0000 | 0.0359 | 1.0000 |
| 90 | 10 | 299,940 | 0.1318 | 0.1033 | 1.0000 | 0.1872 | 0.0353 | 1.0000 |
| 80 | 20 | 291,350 | 0.2280 | 0.2058 | 1.0000 | 0.3413 | 0.0350 | 1.0000 |
| 70 | 30 | 194,230 | 0.3250 | 0.3077 | 1.0000 | 0.4706 | 0.0357 | 1.0000 |
| 60 | 40 | 145,675 | 0.4211 | 0.4086 | 1.0000 | 0.5802 | 0.0352 | 1.0000 |
| 50 | 50 | 116,540 | 0.5176 | 0.5089 | 1.0000 | 0.6746 | 0.0351 | 1.0000 |
| 40 | 60 | 97,115 | 0.6137 | 0.6084 | 1.0000 | 0.7565 | 0.0343 | 1.0000 |
| 30 | 70 | 83,240 | 0.7105 | 0.7075 | 1.0000 | 0.8287 | 0.0352 | 1.0000 |
| 20 | 80 | 72,835 | 0.8069 | 0.8055 | 1.0000 | 0.8923 | 0.0344 | 1.0000 |
| 10 | 90 | 64,740 | 0.9037 | 0.9034 | 1.0000 | 0.9492 | 0.0372 | 1.0000 |
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
0.15       0.9343   0.7511   0.9280   0.9990   0.9913   0.6046   <--
0.20       0.9343   0.7511   0.9280   0.9990   0.9913   0.6046  
0.25       0.9343   0.7511   0.9280   0.9990   0.9913   0.6046  
0.30       0.9343   0.7511   0.9280   0.9990   0.9913   0.6046  
0.35       0.9343   0.7511   0.9280   0.9990   0.9913   0.6046  
0.40       0.9343   0.7511   0.9280   0.9990   0.9913   0.6046  
0.45       0.9343   0.7511   0.9280   0.9990   0.9913   0.6046  
0.50       0.9343   0.7511   0.9280   0.9990   0.9913   0.6046  
0.55       0.9343   0.7511   0.9280   0.9990   0.9913   0.6046  
0.60       0.9343   0.7511   0.9280   0.9990   0.9913   0.6046  
0.65       0.9343   0.7511   0.9280   0.9990   0.9913   0.6046  
0.70       0.9343   0.7511   0.9280   0.9990   0.9913   0.6046  
0.75       0.9343   0.7511   0.9280   0.9990   0.9913   0.6046  
0.80       0.9343   0.7511   0.9280   0.9990   0.9913   0.6046  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9343, F1=0.7511, Normal Recall=0.9280, Normal Precision=0.9990, Attack Recall=0.9913, Attack Precision=0.6046

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
0.15       0.9406   0.8697   0.9283   0.9974   0.9902   0.7753   <--
0.20       0.9406   0.8697   0.9283   0.9974   0.9902   0.7753  
0.25       0.9406   0.8697   0.9283   0.9974   0.9902   0.7753  
0.30       0.9406   0.8697   0.9283   0.9974   0.9902   0.7753  
0.35       0.9406   0.8697   0.9283   0.9974   0.9902   0.7753  
0.40       0.9406   0.8697   0.9283   0.9974   0.9902   0.7753  
0.45       0.9406   0.8697   0.9283   0.9974   0.9902   0.7753  
0.50       0.9406   0.8697   0.9283   0.9974   0.9902   0.7753  
0.55       0.9406   0.8697   0.9283   0.9974   0.9902   0.7753  
0.60       0.9406   0.8697   0.9283   0.9974   0.9902   0.7753  
0.65       0.9406   0.8697   0.9283   0.9974   0.9902   0.7753  
0.70       0.9406   0.8697   0.9283   0.9974   0.9902   0.7753  
0.75       0.9406   0.8697   0.9283   0.9974   0.9902   0.7753  
0.80       0.9406   0.8697   0.9283   0.9974   0.9902   0.7753  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9406, F1=0.8697, Normal Recall=0.9283, Normal Precision=0.9974, Attack Recall=0.9902, Attack Precision=0.7753

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
0.15       0.9469   0.9180   0.9284   0.9955   0.9902   0.8556   <--
0.20       0.9469   0.9180   0.9284   0.9955   0.9902   0.8556  
0.25       0.9469   0.9180   0.9284   0.9955   0.9902   0.8556  
0.30       0.9469   0.9180   0.9284   0.9955   0.9902   0.8556  
0.35       0.9469   0.9180   0.9284   0.9955   0.9902   0.8556  
0.40       0.9469   0.9180   0.9284   0.9955   0.9902   0.8556  
0.45       0.9469   0.9180   0.9284   0.9955   0.9902   0.8556  
0.50       0.9469   0.9180   0.9284   0.9955   0.9902   0.8556  
0.55       0.9469   0.9180   0.9284   0.9955   0.9902   0.8556  
0.60       0.9469   0.9180   0.9284   0.9955   0.9902   0.8556  
0.65       0.9469   0.9180   0.9284   0.9955   0.9902   0.8556  
0.70       0.9469   0.9180   0.9284   0.9955   0.9902   0.8556  
0.75       0.9469   0.9180   0.9284   0.9955   0.9902   0.8556  
0.80       0.9469   0.9180   0.9284   0.9955   0.9902   0.8556  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9469, F1=0.9180, Normal Recall=0.9284, Normal Precision=0.9955, Attack Recall=0.9902, Attack Precision=0.8556

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
0.15       0.9529   0.9438   0.9280   0.9930   0.9902   0.9016   <--
0.20       0.9529   0.9438   0.9280   0.9930   0.9902   0.9016  
0.25       0.9529   0.9438   0.9280   0.9930   0.9902   0.9016  
0.30       0.9529   0.9438   0.9280   0.9930   0.9902   0.9016  
0.35       0.9529   0.9438   0.9280   0.9930   0.9902   0.9016  
0.40       0.9529   0.9438   0.9280   0.9930   0.9902   0.9016  
0.45       0.9529   0.9438   0.9280   0.9930   0.9902   0.9016  
0.50       0.9529   0.9438   0.9280   0.9930   0.9902   0.9016  
0.55       0.9529   0.9438   0.9280   0.9930   0.9902   0.9016  
0.60       0.9529   0.9438   0.9280   0.9930   0.9902   0.9016  
0.65       0.9529   0.9438   0.9280   0.9930   0.9902   0.9016  
0.70       0.9529   0.9438   0.9280   0.9930   0.9902   0.9016  
0.75       0.9529   0.9438   0.9280   0.9930   0.9902   0.9016  
0.80       0.9529   0.9438   0.9280   0.9930   0.9902   0.9016  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9529, F1=0.9438, Normal Recall=0.9280, Normal Precision=0.9930, Attack Recall=0.9902, Attack Precision=0.9016

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
0.15       0.9590   0.9603   0.9279   0.9895   0.9902   0.9321   <--
0.20       0.9590   0.9603   0.9279   0.9895   0.9902   0.9321  
0.25       0.9590   0.9603   0.9279   0.9895   0.9902   0.9321  
0.30       0.9590   0.9603   0.9279   0.9895   0.9902   0.9321  
0.35       0.9590   0.9603   0.9279   0.9895   0.9902   0.9321  
0.40       0.9590   0.9603   0.9279   0.9895   0.9902   0.9321  
0.45       0.9590   0.9603   0.9279   0.9895   0.9902   0.9321  
0.50       0.9590   0.9603   0.9279   0.9895   0.9902   0.9321  
0.55       0.9590   0.9603   0.9279   0.9895   0.9902   0.9321  
0.60       0.9590   0.9603   0.9279   0.9895   0.9902   0.9321  
0.65       0.9590   0.9603   0.9279   0.9895   0.9902   0.9321  
0.70       0.9590   0.9603   0.9279   0.9895   0.9902   0.9321  
0.75       0.9590   0.9603   0.9279   0.9895   0.9902   0.9321  
0.80       0.9590   0.9603   0.9279   0.9895   0.9902   0.9321  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9590, F1=0.9603, Normal Recall=0.9279, Normal Precision=0.9895, Attack Recall=0.9902, Attack Precision=0.9321

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
0.15       0.1323   0.1873   0.0359   1.0000   1.0000   0.1033   <--
0.20       0.1323   0.1873   0.0359   1.0000   1.0000   0.1033  
0.25       0.1323   0.1873   0.0359   1.0000   1.0000   0.1033  
0.30       0.1323   0.1873   0.0359   1.0000   1.0000   0.1033  
0.35       0.1323   0.1873   0.0359   1.0000   1.0000   0.1033  
0.40       0.1323   0.1873   0.0359   1.0000   1.0000   0.1033  
0.45       0.1323   0.1873   0.0359   1.0000   1.0000   0.1033  
0.50       0.1323   0.1873   0.0359   1.0000   1.0000   0.1033  
0.55       0.1323   0.1873   0.0359   1.0000   1.0000   0.1033  
0.60       0.1323   0.1873   0.0359   1.0000   1.0000   0.1033  
0.65       0.1323   0.1873   0.0359   1.0000   1.0000   0.1033  
0.70       0.1323   0.1873   0.0359   1.0000   1.0000   0.1033  
0.75       0.1323   0.1873   0.0359   1.0000   1.0000   0.1033  
0.80       0.1323   0.1873   0.0359   1.0000   1.0000   0.1033  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.1323, F1=0.1873, Normal Recall=0.0359, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1033

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
0.15       0.2286   0.3415   0.0358   1.0000   1.0000   0.2059   <--
0.20       0.2286   0.3415   0.0358   1.0000   1.0000   0.2059  
0.25       0.2286   0.3415   0.0358   1.0000   1.0000   0.2059  
0.30       0.2286   0.3415   0.0358   1.0000   1.0000   0.2059  
0.35       0.2286   0.3415   0.0358   1.0000   1.0000   0.2059  
0.40       0.2286   0.3415   0.0358   1.0000   1.0000   0.2059  
0.45       0.2286   0.3415   0.0358   1.0000   1.0000   0.2059  
0.50       0.2286   0.3415   0.0358   1.0000   1.0000   0.2059  
0.55       0.2286   0.3415   0.0358   1.0000   1.0000   0.2059  
0.60       0.2286   0.3415   0.0358   1.0000   1.0000   0.2059  
0.65       0.2286   0.3415   0.0358   1.0000   1.0000   0.2059  
0.70       0.2286   0.3415   0.0358   1.0000   1.0000   0.2059  
0.75       0.2286   0.3415   0.0358   1.0000   1.0000   0.2059  
0.80       0.2286   0.3415   0.0358   1.0000   1.0000   0.2059  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2286, F1=0.3415, Normal Recall=0.0358, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.2059

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
0.15       0.3255   0.4708   0.0364   1.0000   1.0000   0.3078   <--
0.20       0.3255   0.4708   0.0364   1.0000   1.0000   0.3078  
0.25       0.3255   0.4708   0.0364   1.0000   1.0000   0.3078  
0.30       0.3255   0.4708   0.0364   1.0000   1.0000   0.3078  
0.35       0.3255   0.4708   0.0364   1.0000   1.0000   0.3078  
0.40       0.3255   0.4708   0.0364   1.0000   1.0000   0.3078  
0.45       0.3255   0.4708   0.0364   1.0000   1.0000   0.3078  
0.50       0.3255   0.4708   0.0364   1.0000   1.0000   0.3078  
0.55       0.3255   0.4708   0.0364   1.0000   1.0000   0.3078  
0.60       0.3255   0.4708   0.0364   1.0000   1.0000   0.3078  
0.65       0.3255   0.4708   0.0364   1.0000   1.0000   0.3078  
0.70       0.3255   0.4708   0.0364   1.0000   1.0000   0.3078  
0.75       0.3255   0.4708   0.0364   1.0000   1.0000   0.3078  
0.80       0.3255   0.4708   0.0364   1.0000   1.0000   0.3078  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3255, F1=0.4708, Normal Recall=0.0364, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3078

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
0.15       0.4219   0.5805   0.0364   1.0000   1.0000   0.4089   <--
0.20       0.4219   0.5805   0.0364   1.0000   1.0000   0.4089  
0.25       0.4219   0.5805   0.0364   1.0000   1.0000   0.4089  
0.30       0.4219   0.5805   0.0364   1.0000   1.0000   0.4089  
0.35       0.4219   0.5805   0.0364   1.0000   1.0000   0.4089  
0.40       0.4219   0.5805   0.0364   1.0000   1.0000   0.4089  
0.45       0.4219   0.5805   0.0364   1.0000   1.0000   0.4089  
0.50       0.4219   0.5805   0.0364   1.0000   1.0000   0.4089  
0.55       0.4219   0.5805   0.0364   1.0000   1.0000   0.4089  
0.60       0.4219   0.5805   0.0364   1.0000   1.0000   0.4089  
0.65       0.4219   0.5805   0.0364   1.0000   1.0000   0.4089  
0.70       0.4219   0.5805   0.0364   1.0000   1.0000   0.4089  
0.75       0.4219   0.5805   0.0364   1.0000   1.0000   0.4089  
0.80       0.4219   0.5805   0.0364   1.0000   1.0000   0.4089  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4219, F1=0.5805, Normal Recall=0.0364, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4089

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
0.15       0.5181   0.6748   0.0362   1.0000   1.0000   0.5092   <--
0.20       0.5181   0.6748   0.0362   1.0000   1.0000   0.5092  
0.25       0.5181   0.6748   0.0362   1.0000   1.0000   0.5092  
0.30       0.5181   0.6748   0.0362   1.0000   1.0000   0.5092  
0.35       0.5181   0.6748   0.0362   1.0000   1.0000   0.5092  
0.40       0.5181   0.6748   0.0362   1.0000   1.0000   0.5092  
0.45       0.5181   0.6748   0.0362   1.0000   1.0000   0.5092  
0.50       0.5181   0.6748   0.0362   1.0000   1.0000   0.5092  
0.55       0.5181   0.6748   0.0362   1.0000   1.0000   0.5092  
0.60       0.5181   0.6748   0.0362   1.0000   1.0000   0.5092  
0.65       0.5181   0.6748   0.0362   1.0000   1.0000   0.5092  
0.70       0.5181   0.6748   0.0362   1.0000   1.0000   0.5092  
0.75       0.5181   0.6748   0.0362   1.0000   1.0000   0.5092  
0.80       0.5181   0.6748   0.0362   1.0000   1.0000   0.5092  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5181, F1=0.6748, Normal Recall=0.0362, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5092

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
0.15       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033   <--
0.20       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.25       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.30       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.35       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.40       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.45       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.50       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.55       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.60       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.65       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.70       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.75       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.80       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.1318, F1=0.1872, Normal Recall=0.0353, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1033

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
0.15       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058   <--
0.20       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.25       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.30       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.35       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.40       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.45       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.50       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.55       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.60       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.65       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.70       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.75       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.80       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2281, F1=0.3413, Normal Recall=0.0351, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.2058

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
0.15       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077   <--
0.20       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.25       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.30       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.35       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.40       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.45       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.50       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.55       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.60       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.65       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.70       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.75       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.80       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3250, F1=0.4706, Normal Recall=0.0358, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3077

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
0.15       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088   <--
0.20       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.25       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.30       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.35       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.40       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.45       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.50       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.55       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.60       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.65       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.70       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.75       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.80       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4215, F1=0.5803, Normal Recall=0.0358, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4088

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
0.15       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090   <--
0.20       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.25       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.30       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.35       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.40       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.45       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.50       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.55       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.60       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.65       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.70       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.75       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.80       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5178, F1=0.6747, Normal Recall=0.0355, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5090

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
0.15       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033   <--
0.20       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.25       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.30       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.35       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.40       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.45       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.50       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.55       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.60       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.65       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.70       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.75       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
0.80       0.1318   0.1872   0.0353   1.0000   1.0000   0.1033  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.1318, F1=0.1872, Normal Recall=0.0353, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1033

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
0.15       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058   <--
0.20       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.25       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.30       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.35       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.40       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.45       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.50       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.55       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.60       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.65       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.70       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.75       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
0.80       0.2281   0.3413   0.0351   1.0000   1.0000   0.2058  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2281, F1=0.3413, Normal Recall=0.0351, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.2058

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
0.15       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077   <--
0.20       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.25       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.30       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.35       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.40       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.45       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.50       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.55       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.60       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.65       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.70       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.75       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
0.80       0.3250   0.4706   0.0358   1.0000   1.0000   0.3077  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3250, F1=0.4706, Normal Recall=0.0358, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3077

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
0.15       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088   <--
0.20       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.25       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.30       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.35       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.40       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.45       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.50       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.55       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.60       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.65       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.70       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.75       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
0.80       0.4215   0.5803   0.0358   1.0000   1.0000   0.4088  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4215, F1=0.5803, Normal Recall=0.0358, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4088

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
0.15       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090   <--
0.20       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.25       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.30       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.35       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.40       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.45       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.50       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.55       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.60       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.65       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.70       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.75       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
0.80       0.5178   0.6747   0.0355   1.0000   1.0000   0.5090  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5178, F1=0.6747, Normal Recall=0.0355, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5090

```

