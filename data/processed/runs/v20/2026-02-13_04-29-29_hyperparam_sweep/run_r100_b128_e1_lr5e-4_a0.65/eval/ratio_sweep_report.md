# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e1_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-21 03:03:43 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8314 | 0.8460 | 0.8615 | 0.8780 | 0.8942 | 0.9102 | 0.9259 | 0.9427 | 0.9584 | 0.9751 | 0.9910 |
| QAT+Prune only | 0.9306 | 0.9159 | 0.9004 | 0.8856 | 0.8708 | 0.8550 | 0.8406 | 0.8260 | 0.8099 | 0.7949 | 0.7799 |
| QAT+PTQ | 0.9300 | 0.9152 | 0.8996 | 0.8848 | 0.8699 | 0.8539 | 0.8394 | 0.8247 | 0.8084 | 0.7934 | 0.7782 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9300 | 0.9152 | 0.8996 | 0.8848 | 0.8699 | 0.8539 | 0.8394 | 0.8247 | 0.8084 | 0.7934 | 0.7782 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.5628 | 0.7411 | 0.8297 | 0.8823 | 0.9169 | 0.9414 | 0.9603 | 0.9744 | 0.9862 | 0.9955 |
| QAT+Prune only | 0.0000 | 0.6497 | 0.7580 | 0.8036 | 0.8285 | 0.8432 | 0.8545 | 0.8625 | 0.8678 | 0.8725 | 0.8763 |
| QAT+PTQ | 0.0000 | 0.6475 | 0.7561 | 0.8021 | 0.8271 | 0.8420 | 0.8533 | 0.8614 | 0.8667 | 0.8715 | 0.8753 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.6475 | 0.7561 | 0.8021 | 0.8271 | 0.8420 | 0.8533 | 0.8614 | 0.8667 | 0.8715 | 0.8753 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8314 | 0.8299 | 0.8291 | 0.8295 | 0.8297 | 0.8293 | 0.8283 | 0.8298 | 0.8278 | 0.8313 | 0.0000 |
| QAT+Prune only | 0.9306 | 0.9309 | 0.9305 | 0.9310 | 0.9315 | 0.9302 | 0.9317 | 0.9335 | 0.9298 | 0.9305 | 0.0000 |
| QAT+PTQ | 0.9300 | 0.9304 | 0.9300 | 0.9305 | 0.9310 | 0.9297 | 0.9312 | 0.9332 | 0.9294 | 0.9305 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9300 | 0.9304 | 0.9300 | 0.9305 | 0.9310 | 0.9297 | 0.9312 | 0.9332 | 0.9294 | 0.9305 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8314 | 0.0000 | 0.0000 | 0.0000 | 0.8314 | 1.0000 |
| 90 | 10 | 299,940 | 0.8460 | 0.3930 | 0.9912 | 0.5628 | 0.8299 | 0.9988 |
| 80 | 20 | 291,350 | 0.8615 | 0.5919 | 0.9910 | 0.7411 | 0.8291 | 0.9973 |
| 70 | 30 | 194,230 | 0.8780 | 0.7136 | 0.9910 | 0.8297 | 0.8295 | 0.9954 |
| 60 | 40 | 145,675 | 0.8942 | 0.7951 | 0.9910 | 0.8823 | 0.8297 | 0.9928 |
| 50 | 50 | 116,540 | 0.9102 | 0.8531 | 0.9910 | 0.9169 | 0.8293 | 0.9893 |
| 40 | 60 | 97,115 | 0.9259 | 0.8964 | 0.9910 | 0.9414 | 0.8283 | 0.9840 |
| 30 | 70 | 83,240 | 0.9427 | 0.9314 | 0.9910 | 0.9603 | 0.8298 | 0.9754 |
| 20 | 80 | 72,835 | 0.9584 | 0.9584 | 0.9910 | 0.9744 | 0.8278 | 0.9584 |
| 10 | 90 | 64,740 | 0.9751 | 0.9814 | 0.9910 | 0.9862 | 0.8313 | 0.9114 |
| 0 | 100 | 58,270 | 0.9910 | 1.0000 | 0.9910 | 0.9955 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9306 | 0.0000 | 0.0000 | 0.0000 | 0.9306 | 1.0000 |
| 90 | 10 | 299,940 | 0.9159 | 0.5565 | 0.7805 | 0.6497 | 0.9309 | 0.9745 |
| 80 | 20 | 291,350 | 0.9004 | 0.7372 | 0.7799 | 0.7580 | 0.9305 | 0.9442 |
| 70 | 30 | 194,230 | 0.8856 | 0.8288 | 0.7799 | 0.8036 | 0.9310 | 0.9080 |
| 60 | 40 | 145,675 | 0.8708 | 0.8836 | 0.7799 | 0.8285 | 0.9315 | 0.8639 |
| 50 | 50 | 116,540 | 0.8550 | 0.9178 | 0.7799 | 0.8432 | 0.9302 | 0.8086 |
| 40 | 60 | 97,115 | 0.8406 | 0.9448 | 0.7799 | 0.8545 | 0.9317 | 0.7383 |
| 30 | 70 | 83,240 | 0.8260 | 0.9648 | 0.7799 | 0.8625 | 0.9335 | 0.6451 |
| 20 | 80 | 72,835 | 0.8099 | 0.9780 | 0.7799 | 0.8678 | 0.9298 | 0.5136 |
| 10 | 90 | 64,740 | 0.7949 | 0.9902 | 0.7799 | 0.8725 | 0.9305 | 0.3196 |
| 0 | 100 | 58,270 | 0.7799 | 1.0000 | 0.7799 | 0.8763 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9300 | 0.0000 | 0.0000 | 0.0000 | 0.9300 | 1.0000 |
| 90 | 10 | 299,940 | 0.9152 | 0.5541 | 0.7788 | 0.6475 | 0.9304 | 0.9743 |
| 80 | 20 | 291,350 | 0.8996 | 0.7353 | 0.7782 | 0.7561 | 0.9300 | 0.9437 |
| 70 | 30 | 194,230 | 0.8848 | 0.8275 | 0.7782 | 0.8021 | 0.9305 | 0.9073 |
| 60 | 40 | 145,675 | 0.8699 | 0.8826 | 0.7782 | 0.8271 | 0.9310 | 0.8629 |
| 50 | 50 | 116,540 | 0.8539 | 0.9171 | 0.7782 | 0.8420 | 0.9297 | 0.8074 |
| 40 | 60 | 97,115 | 0.8394 | 0.9444 | 0.7782 | 0.8533 | 0.9312 | 0.7368 |
| 30 | 70 | 83,240 | 0.8247 | 0.9645 | 0.7782 | 0.8614 | 0.9332 | 0.6433 |
| 20 | 80 | 72,835 | 0.8084 | 0.9778 | 0.7782 | 0.8667 | 0.9294 | 0.5116 |
| 10 | 90 | 64,740 | 0.7934 | 0.9902 | 0.7782 | 0.8715 | 0.9305 | 0.3179 |
| 0 | 100 | 58,270 | 0.7782 | 1.0000 | 0.7782 | 0.8753 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9300 | 0.0000 | 0.0000 | 0.0000 | 0.9300 | 1.0000 |
| 90 | 10 | 299,940 | 0.9152 | 0.5541 | 0.7788 | 0.6475 | 0.9304 | 0.9743 |
| 80 | 20 | 291,350 | 0.8996 | 0.7353 | 0.7782 | 0.7561 | 0.9300 | 0.9437 |
| 70 | 30 | 194,230 | 0.8848 | 0.8275 | 0.7782 | 0.8021 | 0.9305 | 0.9073 |
| 60 | 40 | 145,675 | 0.8699 | 0.8826 | 0.7782 | 0.8271 | 0.9310 | 0.8629 |
| 50 | 50 | 116,540 | 0.8539 | 0.9171 | 0.7782 | 0.8420 | 0.9297 | 0.8074 |
| 40 | 60 | 97,115 | 0.8394 | 0.9444 | 0.7782 | 0.8533 | 0.9312 | 0.7368 |
| 30 | 70 | 83,240 | 0.8247 | 0.9645 | 0.7782 | 0.8614 | 0.9332 | 0.6433 |
| 20 | 80 | 72,835 | 0.8084 | 0.9778 | 0.7782 | 0.8667 | 0.9294 | 0.5116 |
| 10 | 90 | 64,740 | 0.7934 | 0.9902 | 0.7782 | 0.8715 | 0.9305 | 0.3179 |
| 0 | 100 | 58,270 | 0.7782 | 1.0000 | 0.7782 | 0.8753 | 0.0000 | 0.0000 |


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
0.15       0.8461   0.5633   0.8299   0.9990   0.9923   0.3933   <--
0.20       0.8461   0.5633   0.8299   0.9990   0.9923   0.3933  
0.25       0.8461   0.5633   0.8299   0.9990   0.9923   0.3933  
0.30       0.8461   0.5633   0.8299   0.9990   0.9923   0.3933  
0.35       0.8461   0.5633   0.8299   0.9990   0.9923   0.3933  
0.40       0.8461   0.5633   0.8299   0.9990   0.9923   0.3933  
0.45       0.8461   0.5633   0.8299   0.9990   0.9923   0.3933  
0.50       0.8461   0.5633   0.8299   0.9990   0.9923   0.3933  
0.55       0.8461   0.5633   0.8299   0.9990   0.9923   0.3933  
0.60       0.8461   0.5633   0.8299   0.9990   0.9923   0.3933  
0.65       0.8461   0.5633   0.8299   0.9990   0.9923   0.3933  
0.70       0.8461   0.5633   0.8299   0.9990   0.9923   0.3933  
0.75       0.8461   0.5633   0.8299   0.9990   0.9923   0.3933  
0.80       0.8461   0.5633   0.8299   0.9990   0.9923   0.3933  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8461, F1=0.5633, Normal Recall=0.8299, Normal Precision=0.9990, Attack Recall=0.9923, Attack Precision=0.3933

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
0.15       0.8624   0.7423   0.8302   0.9973   0.9910   0.5934   <--
0.20       0.8624   0.7423   0.8302   0.9973   0.9910   0.5934  
0.25       0.8624   0.7423   0.8302   0.9973   0.9910   0.5934  
0.30       0.8624   0.7423   0.8302   0.9973   0.9910   0.5934  
0.35       0.8624   0.7423   0.8302   0.9973   0.9910   0.5934  
0.40       0.8624   0.7423   0.8302   0.9973   0.9910   0.5934  
0.45       0.8624   0.7423   0.8302   0.9973   0.9910   0.5934  
0.50       0.8624   0.7423   0.8302   0.9973   0.9910   0.5934  
0.55       0.8624   0.7423   0.8302   0.9973   0.9910   0.5934  
0.60       0.8624   0.7423   0.8302   0.9973   0.9910   0.5934  
0.65       0.8624   0.7423   0.8302   0.9973   0.9910   0.5934  
0.70       0.8624   0.7423   0.8302   0.9973   0.9910   0.5934  
0.75       0.8624   0.7423   0.8302   0.9973   0.9910   0.5934  
0.80       0.8624   0.7423   0.8302   0.9973   0.9910   0.5934  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8624, F1=0.7423, Normal Recall=0.8302, Normal Precision=0.9973, Attack Recall=0.9910, Attack Precision=0.5934

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
0.15       0.8794   0.8314   0.8316   0.9954   0.9910   0.7160   <--
0.20       0.8794   0.8314   0.8316   0.9954   0.9910   0.7160  
0.25       0.8794   0.8314   0.8316   0.9954   0.9910   0.7160  
0.30       0.8794   0.8314   0.8316   0.9954   0.9910   0.7160  
0.35       0.8794   0.8314   0.8316   0.9954   0.9910   0.7160  
0.40       0.8794   0.8314   0.8316   0.9954   0.9910   0.7160  
0.45       0.8794   0.8314   0.8316   0.9954   0.9910   0.7160  
0.50       0.8794   0.8314   0.8316   0.9954   0.9910   0.7160  
0.55       0.8794   0.8314   0.8316   0.9954   0.9910   0.7160  
0.60       0.8794   0.8314   0.8316   0.9954   0.9910   0.7160  
0.65       0.8794   0.8314   0.8316   0.9954   0.9910   0.7160  
0.70       0.8794   0.8314   0.8316   0.9954   0.9910   0.7160  
0.75       0.8794   0.8314   0.8316   0.9954   0.9910   0.7160  
0.80       0.8794   0.8314   0.8316   0.9954   0.9910   0.7160  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8794, F1=0.8314, Normal Recall=0.8316, Normal Precision=0.9954, Attack Recall=0.9910, Attack Precision=0.7160

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
0.15       0.8956   0.8836   0.8320   0.9929   0.9910   0.7972   <--
0.20       0.8956   0.8836   0.8320   0.9929   0.9910   0.7972  
0.25       0.8956   0.8836   0.8320   0.9929   0.9910   0.7972  
0.30       0.8956   0.8836   0.8320   0.9929   0.9910   0.7972  
0.35       0.8956   0.8836   0.8320   0.9929   0.9910   0.7972  
0.40       0.8956   0.8836   0.8320   0.9929   0.9910   0.7972  
0.45       0.8956   0.8836   0.8320   0.9929   0.9910   0.7972  
0.50       0.8956   0.8836   0.8320   0.9929   0.9910   0.7972  
0.55       0.8956   0.8836   0.8320   0.9929   0.9910   0.7972  
0.60       0.8956   0.8836   0.8320   0.9929   0.9910   0.7972  
0.65       0.8956   0.8836   0.8320   0.9929   0.9910   0.7972  
0.70       0.8956   0.8836   0.8320   0.9929   0.9910   0.7972  
0.75       0.8956   0.8836   0.8320   0.9929   0.9910   0.7972  
0.80       0.8956   0.8836   0.8320   0.9929   0.9910   0.7972  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8956, F1=0.8836, Normal Recall=0.8320, Normal Precision=0.9929, Attack Recall=0.9910, Attack Precision=0.7972

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
0.15       0.9114   0.9179   0.8317   0.9893   0.9910   0.8549   <--
0.20       0.9114   0.9179   0.8317   0.9893   0.9910   0.8549  
0.25       0.9114   0.9179   0.8317   0.9893   0.9910   0.8549  
0.30       0.9114   0.9179   0.8317   0.9893   0.9910   0.8549  
0.35       0.9114   0.9179   0.8317   0.9893   0.9910   0.8549  
0.40       0.9114   0.9179   0.8317   0.9893   0.9910   0.8549  
0.45       0.9114   0.9179   0.8317   0.9893   0.9910   0.8549  
0.50       0.9114   0.9179   0.8317   0.9893   0.9910   0.8549  
0.55       0.9114   0.9179   0.8317   0.9893   0.9910   0.8549  
0.60       0.9114   0.9179   0.8317   0.9893   0.9910   0.8549  
0.65       0.9114   0.9179   0.8317   0.9893   0.9910   0.8549  
0.70       0.9114   0.9179   0.8317   0.9893   0.9910   0.8549  
0.75       0.9114   0.9179   0.8317   0.9893   0.9910   0.8549  
0.80       0.9114   0.9179   0.8317   0.9893   0.9910   0.8549  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9114, F1=0.9179, Normal Recall=0.8317, Normal Precision=0.9893, Attack Recall=0.9910, Attack Precision=0.8549

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
0.15       0.9155   0.6478   0.9309   0.9741   0.7771   0.5555   <--
0.20       0.9155   0.6478   0.9309   0.9741   0.7771   0.5555  
0.25       0.9155   0.6478   0.9309   0.9741   0.7771   0.5555  
0.30       0.9155   0.6478   0.9309   0.9741   0.7771   0.5555  
0.35       0.9155   0.6478   0.9309   0.9741   0.7771   0.5555  
0.40       0.9155   0.6478   0.9309   0.9741   0.7771   0.5555  
0.45       0.9155   0.6478   0.9309   0.9741   0.7771   0.5555  
0.50       0.9155   0.6478   0.9309   0.9741   0.7771   0.5555  
0.55       0.9155   0.6478   0.9309   0.9741   0.7771   0.5555  
0.60       0.9155   0.6478   0.9309   0.9741   0.7771   0.5555  
0.65       0.9155   0.6478   0.9309   0.9741   0.7771   0.5555  
0.70       0.9155   0.6478   0.9309   0.9741   0.7771   0.5555  
0.75       0.9155   0.6478   0.9309   0.9741   0.7771   0.5555  
0.80       0.9155   0.6478   0.9309   0.9741   0.7771   0.5555  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9155, F1=0.6478, Normal Recall=0.9309, Normal Precision=0.9741, Attack Recall=0.7771, Attack Precision=0.5555

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
0.15       0.9008   0.7587   0.9310   0.9442   0.7799   0.7386   <--
0.20       0.9008   0.7587   0.9310   0.9442   0.7799   0.7386  
0.25       0.9008   0.7587   0.9310   0.9442   0.7799   0.7386  
0.30       0.9008   0.7587   0.9310   0.9442   0.7799   0.7386  
0.35       0.9008   0.7587   0.9310   0.9442   0.7799   0.7386  
0.40       0.9008   0.7587   0.9310   0.9442   0.7799   0.7386  
0.45       0.9008   0.7587   0.9310   0.9442   0.7799   0.7386  
0.50       0.9008   0.7587   0.9310   0.9442   0.7799   0.7386  
0.55       0.9008   0.7587   0.9310   0.9442   0.7799   0.7386  
0.60       0.9008   0.7587   0.9310   0.9442   0.7799   0.7386  
0.65       0.9008   0.7587   0.9310   0.9442   0.7799   0.7386  
0.70       0.9008   0.7587   0.9310   0.9442   0.7799   0.7386  
0.75       0.9008   0.7587   0.9310   0.9442   0.7799   0.7386  
0.80       0.9008   0.7587   0.9310   0.9442   0.7799   0.7386  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9008, F1=0.7587, Normal Recall=0.9310, Normal Precision=0.9442, Attack Recall=0.7799, Attack Precision=0.7386

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
0.15       0.8857   0.8037   0.9311   0.9080   0.7799   0.8291   <--
0.20       0.8857   0.8037   0.9311   0.9080   0.7799   0.8291  
0.25       0.8857   0.8037   0.9311   0.9080   0.7799   0.8291  
0.30       0.8857   0.8037   0.9311   0.9080   0.7799   0.8291  
0.35       0.8857   0.8037   0.9311   0.9080   0.7799   0.8291  
0.40       0.8857   0.8037   0.9311   0.9080   0.7799   0.8291  
0.45       0.8857   0.8037   0.9311   0.9080   0.7799   0.8291  
0.50       0.8857   0.8037   0.9311   0.9080   0.7799   0.8291  
0.55       0.8857   0.8037   0.9311   0.9080   0.7799   0.8291  
0.60       0.8857   0.8037   0.9311   0.9080   0.7799   0.8291  
0.65       0.8857   0.8037   0.9311   0.9080   0.7799   0.8291  
0.70       0.8857   0.8037   0.9311   0.9080   0.7799   0.8291  
0.75       0.8857   0.8037   0.9311   0.9080   0.7799   0.8291  
0.80       0.8857   0.8037   0.9311   0.9080   0.7799   0.8291  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8857, F1=0.8037, Normal Recall=0.9311, Normal Precision=0.9080, Attack Recall=0.7799, Attack Precision=0.8291

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
0.15       0.8703   0.8279   0.9306   0.8638   0.7799   0.8822   <--
0.20       0.8703   0.8279   0.9306   0.8638   0.7799   0.8822  
0.25       0.8703   0.8279   0.9306   0.8638   0.7799   0.8822  
0.30       0.8703   0.8279   0.9306   0.8638   0.7799   0.8822  
0.35       0.8703   0.8279   0.9306   0.8638   0.7799   0.8822  
0.40       0.8703   0.8279   0.9306   0.8638   0.7799   0.8822  
0.45       0.8703   0.8279   0.9306   0.8638   0.7799   0.8822  
0.50       0.8703   0.8279   0.9306   0.8638   0.7799   0.8822  
0.55       0.8703   0.8279   0.9306   0.8638   0.7799   0.8822  
0.60       0.8703   0.8279   0.9306   0.8638   0.7799   0.8822  
0.65       0.8703   0.8279   0.9306   0.8638   0.7799   0.8822  
0.70       0.8703   0.8279   0.9306   0.8638   0.7799   0.8822  
0.75       0.8703   0.8279   0.9306   0.8638   0.7799   0.8822  
0.80       0.8703   0.8279   0.9306   0.8638   0.7799   0.8822  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8703, F1=0.8279, Normal Recall=0.9306, Normal Precision=0.8638, Attack Recall=0.7799, Attack Precision=0.8822

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
0.15       0.8553   0.8435   0.9308   0.8087   0.7799   0.9185   <--
0.20       0.8553   0.8435   0.9308   0.8087   0.7799   0.9185  
0.25       0.8553   0.8435   0.9308   0.8087   0.7799   0.9185  
0.30       0.8553   0.8435   0.9308   0.8087   0.7799   0.9185  
0.35       0.8553   0.8435   0.9308   0.8087   0.7799   0.9185  
0.40       0.8553   0.8435   0.9308   0.8087   0.7799   0.9185  
0.45       0.8553   0.8435   0.9308   0.8087   0.7799   0.9185  
0.50       0.8553   0.8435   0.9308   0.8087   0.7799   0.9185  
0.55       0.8553   0.8435   0.9308   0.8087   0.7799   0.9185  
0.60       0.8553   0.8435   0.9308   0.8087   0.7799   0.9185  
0.65       0.8553   0.8435   0.9308   0.8087   0.7799   0.9185  
0.70       0.8553   0.8435   0.9308   0.8087   0.7799   0.9185  
0.75       0.8553   0.8435   0.9308   0.8087   0.7799   0.9185  
0.80       0.8553   0.8435   0.9308   0.8087   0.7799   0.9185  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8553, F1=0.8435, Normal Recall=0.9308, Normal Precision=0.8087, Attack Recall=0.7799, Attack Precision=0.9185

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
0.15       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529   <--
0.20       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.25       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.30       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.35       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.40       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.45       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.50       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.55       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.60       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.65       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.70       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.75       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.80       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9148, F1=0.6454, Normal Recall=0.9304, Normal Precision=0.9738, Attack Recall=0.7751, Attack Precision=0.5529

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
0.15       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367   <--
0.20       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.25       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.30       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.35       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.40       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.45       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.50       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.55       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.60       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.65       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.70       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.75       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.80       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9000, F1=0.7569, Normal Recall=0.9305, Normal Precision=0.9438, Attack Recall=0.7782, Attack Precision=0.7367

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
0.15       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275   <--
0.20       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.25       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.30       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.35       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.40       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.45       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.50       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.55       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.60       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.65       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.70       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.75       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.80       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8848, F1=0.8021, Normal Recall=0.9305, Normal Precision=0.9073, Attack Recall=0.7782, Attack Precision=0.8275

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
0.15       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811   <--
0.20       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.25       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.30       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.35       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.40       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.45       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.50       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.55       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.60       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.65       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.70       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.75       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.80       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8693, F1=0.8265, Normal Recall=0.9300, Normal Precision=0.8628, Attack Recall=0.7782, Attack Precision=0.8811

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
0.15       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177   <--
0.20       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.25       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.30       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.35       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.40       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.45       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.50       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.55       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.60       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.65       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.70       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.75       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.80       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8542, F1=0.8422, Normal Recall=0.9303, Normal Precision=0.8075, Attack Recall=0.7782, Attack Precision=0.9177

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
0.15       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529   <--
0.20       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.25       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.30       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.35       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.40       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.45       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.50       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.55       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.60       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.65       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.70       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.75       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
0.80       0.9148   0.6454   0.9304   0.9738   0.7751   0.5529  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9148, F1=0.6454, Normal Recall=0.9304, Normal Precision=0.9738, Attack Recall=0.7751, Attack Precision=0.5529

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
0.15       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367   <--
0.20       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.25       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.30       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.35       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.40       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.45       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.50       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.55       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.60       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.65       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.70       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.75       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
0.80       0.9000   0.7569   0.9305   0.9438   0.7782   0.7367  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9000, F1=0.7569, Normal Recall=0.9305, Normal Precision=0.9438, Attack Recall=0.7782, Attack Precision=0.7367

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
0.15       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275   <--
0.20       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.25       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.30       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.35       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.40       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.45       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.50       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.55       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.60       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.65       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.70       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.75       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
0.80       0.8848   0.8021   0.9305   0.9073   0.7782   0.8275  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8848, F1=0.8021, Normal Recall=0.9305, Normal Precision=0.9073, Attack Recall=0.7782, Attack Precision=0.8275

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
0.15       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811   <--
0.20       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.25       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.30       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.35       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.40       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.45       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.50       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.55       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.60       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.65       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.70       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.75       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
0.80       0.8693   0.8265   0.9300   0.8628   0.7782   0.8811  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8693, F1=0.8265, Normal Recall=0.9300, Normal Precision=0.8628, Attack Recall=0.7782, Attack Precision=0.8811

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
0.15       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177   <--
0.20       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.25       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.30       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.35       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.40       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.45       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.50       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.55       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.60       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.65       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.70       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.75       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
0.80       0.8542   0.8422   0.9303   0.8075   0.7782   0.9177  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8542, F1=0.8422, Normal Recall=0.9303, Normal Precision=0.8075, Attack Recall=0.7782, Attack Precision=0.9177

```

