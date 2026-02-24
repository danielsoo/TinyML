# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-12 05:33:01 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8808 | 0.8662 | 0.8520 | 0.8384 | 0.8239 | 0.8100 | 0.7962 | 0.7833 | 0.7672 | 0.7538 | 0.7399 |
| QAT+Prune only | 0.7234 | 0.7496 | 0.7762 | 0.8027 | 0.8303 | 0.8577 | 0.8864 | 0.9128 | 0.9405 | 0.9671 | 0.9947 |
| QAT+PTQ | 0.7205 | 0.7470 | 0.7738 | 0.8006 | 0.8286 | 0.8562 | 0.8852 | 0.9117 | 0.9401 | 0.9668 | 0.9947 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7205 | 0.7470 | 0.7738 | 0.8006 | 0.8286 | 0.8562 | 0.8852 | 0.9117 | 0.9401 | 0.9668 | 0.9947 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.5251 | 0.6667 | 0.7332 | 0.7707 | 0.7956 | 0.8133 | 0.8270 | 0.8357 | 0.8440 | 0.8505 |
| QAT+Prune only | 0.0000 | 0.4428 | 0.6400 | 0.7515 | 0.8243 | 0.8748 | 0.9131 | 0.9411 | 0.9640 | 0.9819 | 0.9973 |
| QAT+PTQ | 0.0000 | 0.4402 | 0.6376 | 0.7496 | 0.8228 | 0.8737 | 0.9123 | 0.9404 | 0.9637 | 0.9818 | 0.9974 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4402 | 0.6376 | 0.7496 | 0.8228 | 0.8737 | 0.9123 | 0.9404 | 0.9637 | 0.9818 | 0.9974 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8808 | 0.8803 | 0.8801 | 0.8807 | 0.8799 | 0.8800 | 0.8805 | 0.8844 | 0.8762 | 0.8784 | 0.0000 |
| QAT+Prune only | 0.7234 | 0.7223 | 0.7216 | 0.7204 | 0.7207 | 0.7207 | 0.7239 | 0.7217 | 0.7239 | 0.7183 | 0.0000 |
| QAT+PTQ | 0.7205 | 0.7194 | 0.7186 | 0.7174 | 0.7179 | 0.7176 | 0.7208 | 0.7181 | 0.7214 | 0.7153 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7205 | 0.7194 | 0.7186 | 0.7174 | 0.7179 | 0.7176 | 0.7208 | 0.7181 | 0.7214 | 0.7153 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8808 | 0.0000 | 0.0000 | 0.0000 | 0.8808 | 1.0000 |
| 90 | 10 | 299,940 | 0.8662 | 0.4071 | 0.7396 | 0.5251 | 0.8803 | 0.9682 |
| 80 | 20 | 291,350 | 0.8520 | 0.6067 | 0.7399 | 0.6667 | 0.8801 | 0.9312 |
| 70 | 30 | 194,230 | 0.8384 | 0.7266 | 0.7399 | 0.7332 | 0.8807 | 0.8877 |
| 60 | 40 | 145,675 | 0.8239 | 0.8042 | 0.7399 | 0.7707 | 0.8799 | 0.8354 |
| 50 | 50 | 116,540 | 0.8100 | 0.8604 | 0.7399 | 0.7956 | 0.8800 | 0.7719 |
| 40 | 60 | 97,115 | 0.7962 | 0.9028 | 0.7399 | 0.8133 | 0.8805 | 0.6930 |
| 30 | 70 | 83,240 | 0.7833 | 0.9372 | 0.7399 | 0.8270 | 0.8844 | 0.5931 |
| 20 | 80 | 72,835 | 0.7672 | 0.9599 | 0.7399 | 0.8357 | 0.8762 | 0.4572 |
| 10 | 90 | 64,740 | 0.7538 | 0.9821 | 0.7400 | 0.8440 | 0.8784 | 0.2729 |
| 0 | 100 | 58,270 | 0.7399 | 1.0000 | 0.7399 | 0.8505 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7234 | 0.0000 | 0.0000 | 0.0000 | 0.7234 | 1.0000 |
| 90 | 10 | 299,940 | 0.7496 | 0.2848 | 0.9950 | 0.4428 | 0.7223 | 0.9992 |
| 80 | 20 | 291,350 | 0.7762 | 0.4718 | 0.9947 | 0.6400 | 0.7216 | 0.9982 |
| 70 | 30 | 194,230 | 0.8027 | 0.6039 | 0.9947 | 0.7515 | 0.7204 | 0.9969 |
| 60 | 40 | 145,675 | 0.8303 | 0.7037 | 0.9947 | 0.8243 | 0.7207 | 0.9951 |
| 50 | 50 | 116,540 | 0.8577 | 0.7808 | 0.9947 | 0.8748 | 0.7207 | 0.9927 |
| 40 | 60 | 97,115 | 0.8864 | 0.8439 | 0.9947 | 0.9131 | 0.7239 | 0.9891 |
| 30 | 70 | 83,240 | 0.9128 | 0.8929 | 0.9947 | 0.9411 | 0.7217 | 0.9831 |
| 20 | 80 | 72,835 | 0.9405 | 0.9351 | 0.9947 | 0.9640 | 0.7239 | 0.9715 |
| 10 | 90 | 64,740 | 0.9671 | 0.9695 | 0.9947 | 0.9819 | 0.7183 | 0.9377 |
| 0 | 100 | 58,270 | 0.9947 | 1.0000 | 0.9947 | 0.9973 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7205 | 0.0000 | 0.0000 | 0.0000 | 0.7205 | 1.0000 |
| 90 | 10 | 299,940 | 0.7470 | 0.2826 | 0.9950 | 0.4402 | 0.7194 | 0.9992 |
| 80 | 20 | 291,350 | 0.7738 | 0.4692 | 0.9947 | 0.6376 | 0.7186 | 0.9982 |
| 70 | 30 | 194,230 | 0.8006 | 0.6014 | 0.9947 | 0.7496 | 0.7174 | 0.9969 |
| 60 | 40 | 145,675 | 0.8286 | 0.7015 | 0.9947 | 0.8228 | 0.7179 | 0.9951 |
| 50 | 50 | 116,540 | 0.8562 | 0.7789 | 0.9947 | 0.8737 | 0.7176 | 0.9927 |
| 40 | 60 | 97,115 | 0.8852 | 0.8424 | 0.9947 | 0.9123 | 0.7208 | 0.9892 |
| 30 | 70 | 83,240 | 0.9117 | 0.8917 | 0.9947 | 0.9404 | 0.7181 | 0.9832 |
| 20 | 80 | 72,835 | 0.9401 | 0.9346 | 0.9947 | 0.9637 | 0.7214 | 0.9717 |
| 10 | 90 | 64,740 | 0.9668 | 0.9692 | 0.9947 | 0.9818 | 0.7153 | 0.9380 |
| 0 | 100 | 58,270 | 0.9947 | 1.0000 | 0.9947 | 0.9974 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7205 | 0.0000 | 0.0000 | 0.0000 | 0.7205 | 1.0000 |
| 90 | 10 | 299,940 | 0.7470 | 0.2826 | 0.9950 | 0.4402 | 0.7194 | 0.9992 |
| 80 | 20 | 291,350 | 0.7738 | 0.4692 | 0.9947 | 0.6376 | 0.7186 | 0.9982 |
| 70 | 30 | 194,230 | 0.8006 | 0.6014 | 0.9947 | 0.7496 | 0.7174 | 0.9969 |
| 60 | 40 | 145,675 | 0.8286 | 0.7015 | 0.9947 | 0.8228 | 0.7179 | 0.9951 |
| 50 | 50 | 116,540 | 0.8562 | 0.7789 | 0.9947 | 0.8737 | 0.7176 | 0.9927 |
| 40 | 60 | 97,115 | 0.8852 | 0.8424 | 0.9947 | 0.9123 | 0.7208 | 0.9892 |
| 30 | 70 | 83,240 | 0.9117 | 0.8917 | 0.9947 | 0.9404 | 0.7181 | 0.9832 |
| 20 | 80 | 72,835 | 0.9401 | 0.9346 | 0.9947 | 0.9637 | 0.7214 | 0.9717 |
| 10 | 90 | 64,740 | 0.9668 | 0.9692 | 0.9947 | 0.9818 | 0.7153 | 0.9380 |
| 0 | 100 | 58,270 | 0.9947 | 1.0000 | 0.9947 | 0.9974 | 0.0000 | 0.0000 |


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
0.15       0.8660   0.5240   0.8803   0.9679   0.7375   0.4064   <--
0.20       0.8660   0.5240   0.8803   0.9679   0.7375   0.4064  
0.25       0.8660   0.5240   0.8803   0.9679   0.7375   0.4064  
0.30       0.8660   0.5240   0.8803   0.9679   0.7375   0.4064  
0.35       0.8660   0.5240   0.8803   0.9679   0.7375   0.4064  
0.40       0.8660   0.5240   0.8803   0.9679   0.7375   0.4064  
0.45       0.8660   0.5240   0.8803   0.9679   0.7375   0.4064  
0.50       0.8660   0.5240   0.8803   0.9679   0.7375   0.4064  
0.55       0.8660   0.5240   0.8803   0.9679   0.7375   0.4064  
0.60       0.8660   0.5240   0.8803   0.9679   0.7375   0.4064  
0.65       0.8660   0.5240   0.8803   0.9679   0.7375   0.4064  
0.70       0.8660   0.5240   0.8803   0.9679   0.7375   0.4064  
0.75       0.8660   0.5240   0.8803   0.9679   0.7375   0.4064  
0.80       0.8660   0.5240   0.8803   0.9679   0.7375   0.4064  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8660, F1=0.5240, Normal Recall=0.8803, Normal Precision=0.9679, Attack Recall=0.7375, Attack Precision=0.4064

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
0.15       0.8525   0.6673   0.8806   0.9312   0.7399   0.6077   <--
0.20       0.8525   0.6673   0.8806   0.9312   0.7399   0.6077  
0.25       0.8525   0.6673   0.8806   0.9312   0.7399   0.6077  
0.30       0.8525   0.6673   0.8806   0.9312   0.7399   0.6077  
0.35       0.8525   0.6673   0.8806   0.9312   0.7399   0.6077  
0.40       0.8525   0.6673   0.8806   0.9312   0.7399   0.6077  
0.45       0.8525   0.6673   0.8806   0.9312   0.7399   0.6077  
0.50       0.8525   0.6673   0.8806   0.9312   0.7399   0.6077  
0.55       0.8525   0.6673   0.8806   0.9312   0.7399   0.6077  
0.60       0.8525   0.6673   0.8806   0.9312   0.7399   0.6077  
0.65       0.8525   0.6673   0.8806   0.9312   0.7399   0.6077  
0.70       0.8525   0.6673   0.8806   0.9312   0.7399   0.6077  
0.75       0.8525   0.6673   0.8806   0.9312   0.7399   0.6077  
0.80       0.8525   0.6673   0.8806   0.9312   0.7399   0.6077  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8525, F1=0.6673, Normal Recall=0.8806, Normal Precision=0.9312, Attack Recall=0.7399, Attack Precision=0.6077

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
0.15       0.8387   0.7335   0.8810   0.8877   0.7399   0.7271   <--
0.20       0.8387   0.7335   0.8810   0.8877   0.7399   0.7271  
0.25       0.8387   0.7335   0.8810   0.8877   0.7399   0.7271  
0.30       0.8387   0.7335   0.8810   0.8877   0.7399   0.7271  
0.35       0.8387   0.7335   0.8810   0.8877   0.7399   0.7271  
0.40       0.8387   0.7335   0.8810   0.8877   0.7399   0.7271  
0.45       0.8387   0.7335   0.8810   0.8877   0.7399   0.7271  
0.50       0.8387   0.7335   0.8810   0.8877   0.7399   0.7271  
0.55       0.8387   0.7335   0.8810   0.8877   0.7399   0.7271  
0.60       0.8387   0.7335   0.8810   0.8877   0.7399   0.7271  
0.65       0.8387   0.7335   0.8810   0.8877   0.7399   0.7271  
0.70       0.8387   0.7335   0.8810   0.8877   0.7399   0.7271  
0.75       0.8387   0.7335   0.8810   0.8877   0.7399   0.7271  
0.80       0.8387   0.7335   0.8810   0.8877   0.7399   0.7271  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8387, F1=0.7335, Normal Recall=0.8810, Normal Precision=0.8877, Attack Recall=0.7399, Attack Precision=0.7271

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
0.15       0.8246   0.7714   0.8811   0.8356   0.7399   0.8057   <--
0.20       0.8246   0.7714   0.8811   0.8356   0.7399   0.8057  
0.25       0.8246   0.7714   0.8811   0.8356   0.7399   0.8057  
0.30       0.8246   0.7714   0.8811   0.8356   0.7399   0.8057  
0.35       0.8246   0.7714   0.8811   0.8356   0.7399   0.8057  
0.40       0.8246   0.7714   0.8811   0.8356   0.7399   0.8057  
0.45       0.8246   0.7714   0.8811   0.8356   0.7399   0.8057  
0.50       0.8246   0.7714   0.8811   0.8356   0.7399   0.8057  
0.55       0.8246   0.7714   0.8811   0.8356   0.7399   0.8057  
0.60       0.8246   0.7714   0.8811   0.8356   0.7399   0.8057  
0.65       0.8246   0.7714   0.8811   0.8356   0.7399   0.8057  
0.70       0.8246   0.7714   0.8811   0.8356   0.7399   0.8057  
0.75       0.8246   0.7714   0.8811   0.8356   0.7399   0.8057  
0.80       0.8246   0.7714   0.8811   0.8356   0.7399   0.8057  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8246, F1=0.7714, Normal Recall=0.8811, Normal Precision=0.8356, Attack Recall=0.7399, Attack Precision=0.8057

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
0.15       0.8104   0.7960   0.8809   0.7721   0.7399   0.8614   <--
0.20       0.8104   0.7960   0.8809   0.7721   0.7399   0.8614  
0.25       0.8104   0.7960   0.8809   0.7721   0.7399   0.8614  
0.30       0.8104   0.7960   0.8809   0.7721   0.7399   0.8614  
0.35       0.8104   0.7960   0.8809   0.7721   0.7399   0.8614  
0.40       0.8104   0.7960   0.8809   0.7721   0.7399   0.8614  
0.45       0.8104   0.7960   0.8809   0.7721   0.7399   0.8614  
0.50       0.8104   0.7960   0.8809   0.7721   0.7399   0.8614  
0.55       0.8104   0.7960   0.8809   0.7721   0.7399   0.8614  
0.60       0.8104   0.7960   0.8809   0.7721   0.7399   0.8614  
0.65       0.8104   0.7960   0.8809   0.7721   0.7399   0.8614  
0.70       0.8104   0.7960   0.8809   0.7721   0.7399   0.8614  
0.75       0.8104   0.7960   0.8809   0.7721   0.7399   0.8614  
0.80       0.8104   0.7960   0.8809   0.7721   0.7399   0.8614  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8104, F1=0.7960, Normal Recall=0.8809, Normal Precision=0.7721, Attack Recall=0.7399, Attack Precision=0.8614

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
0.15       0.7496   0.4428   0.7223   0.9992   0.9951   0.2848   <--
0.20       0.7496   0.4428   0.7223   0.9992   0.9951   0.2848  
0.25       0.7496   0.4428   0.7223   0.9992   0.9951   0.2848  
0.30       0.7496   0.4428   0.7223   0.9992   0.9951   0.2848  
0.35       0.7496   0.4428   0.7223   0.9992   0.9951   0.2848  
0.40       0.7496   0.4428   0.7223   0.9992   0.9951   0.2848  
0.45       0.7496   0.4428   0.7223   0.9992   0.9951   0.2848  
0.50       0.7496   0.4428   0.7223   0.9992   0.9951   0.2848  
0.55       0.7496   0.4428   0.7223   0.9992   0.9951   0.2848  
0.60       0.7496   0.4428   0.7223   0.9992   0.9951   0.2848  
0.65       0.7496   0.4428   0.7223   0.9992   0.9951   0.2848  
0.70       0.7496   0.4428   0.7223   0.9992   0.9951   0.2848  
0.75       0.7496   0.4428   0.7223   0.9992   0.9951   0.2848  
0.80       0.7496   0.4428   0.7223   0.9992   0.9951   0.2848  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7496, F1=0.4428, Normal Recall=0.7223, Normal Precision=0.9992, Attack Recall=0.9951, Attack Precision=0.2848

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
0.15       0.7772   0.6410   0.7228   0.9982   0.9947   0.4729   <--
0.20       0.7772   0.6410   0.7228   0.9982   0.9947   0.4729  
0.25       0.7772   0.6410   0.7228   0.9982   0.9947   0.4729  
0.30       0.7772   0.6410   0.7228   0.9982   0.9947   0.4729  
0.35       0.7772   0.6410   0.7228   0.9982   0.9947   0.4729  
0.40       0.7772   0.6410   0.7228   0.9982   0.9947   0.4729  
0.45       0.7772   0.6410   0.7228   0.9982   0.9947   0.4729  
0.50       0.7772   0.6410   0.7228   0.9982   0.9947   0.4729  
0.55       0.7772   0.6410   0.7228   0.9982   0.9947   0.4729  
0.60       0.7772   0.6410   0.7228   0.9982   0.9947   0.4729  
0.65       0.7772   0.6410   0.7228   0.9982   0.9947   0.4729  
0.70       0.7772   0.6410   0.7228   0.9982   0.9947   0.4729  
0.75       0.7772   0.6410   0.7228   0.9982   0.9947   0.4729  
0.80       0.7772   0.6410   0.7228   0.9982   0.9947   0.4729  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7772, F1=0.6410, Normal Recall=0.7228, Normal Precision=0.9982, Attack Recall=0.9947, Attack Precision=0.4729

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
0.15       0.8045   0.7533   0.7230   0.9969   0.9947   0.6062   <--
0.20       0.8045   0.7533   0.7230   0.9969   0.9947   0.6062  
0.25       0.8045   0.7533   0.7230   0.9969   0.9947   0.6062  
0.30       0.8045   0.7533   0.7230   0.9969   0.9947   0.6062  
0.35       0.8045   0.7533   0.7230   0.9969   0.9947   0.6062  
0.40       0.8045   0.7533   0.7230   0.9969   0.9947   0.6062  
0.45       0.8045   0.7533   0.7230   0.9969   0.9947   0.6062  
0.50       0.8045   0.7533   0.7230   0.9969   0.9947   0.6062  
0.55       0.8045   0.7533   0.7230   0.9969   0.9947   0.6062  
0.60       0.8045   0.7533   0.7230   0.9969   0.9947   0.6062  
0.65       0.8045   0.7533   0.7230   0.9969   0.9947   0.6062  
0.70       0.8045   0.7533   0.7230   0.9969   0.9947   0.6062  
0.75       0.8045   0.7533   0.7230   0.9969   0.9947   0.6062  
0.80       0.8045   0.7533   0.7230   0.9969   0.9947   0.6062  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8045, F1=0.7533, Normal Recall=0.7230, Normal Precision=0.9969, Attack Recall=0.9947, Attack Precision=0.6062

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
0.15       0.8312   0.8250   0.7222   0.9951   0.9947   0.7048   <--
0.20       0.8312   0.8250   0.7222   0.9951   0.9947   0.7048  
0.25       0.8312   0.8250   0.7222   0.9951   0.9947   0.7048  
0.30       0.8312   0.8250   0.7222   0.9951   0.9947   0.7048  
0.35       0.8312   0.8250   0.7222   0.9951   0.9947   0.7048  
0.40       0.8312   0.8250   0.7222   0.9951   0.9947   0.7048  
0.45       0.8312   0.8250   0.7222   0.9951   0.9947   0.7048  
0.50       0.8312   0.8250   0.7222   0.9951   0.9947   0.7048  
0.55       0.8312   0.8250   0.7222   0.9951   0.9947   0.7048  
0.60       0.8312   0.8250   0.7222   0.9951   0.9947   0.7048  
0.65       0.8312   0.8250   0.7222   0.9951   0.9947   0.7048  
0.70       0.8312   0.8250   0.7222   0.9951   0.9947   0.7048  
0.75       0.8312   0.8250   0.7222   0.9951   0.9947   0.7048  
0.80       0.8312   0.8250   0.7222   0.9951   0.9947   0.7048  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8312, F1=0.8250, Normal Recall=0.7222, Normal Precision=0.9951, Attack Recall=0.9947, Attack Precision=0.7048

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
0.15       0.8581   0.8751   0.7214   0.9927   0.9947   0.7812   <--
0.20       0.8581   0.8751   0.7214   0.9927   0.9947   0.7812  
0.25       0.8581   0.8751   0.7214   0.9927   0.9947   0.7812  
0.30       0.8581   0.8751   0.7214   0.9927   0.9947   0.7812  
0.35       0.8581   0.8751   0.7214   0.9927   0.9947   0.7812  
0.40       0.8581   0.8751   0.7214   0.9927   0.9947   0.7812  
0.45       0.8581   0.8751   0.7214   0.9927   0.9947   0.7812  
0.50       0.8581   0.8751   0.7214   0.9927   0.9947   0.7812  
0.55       0.8581   0.8751   0.7214   0.9927   0.9947   0.7812  
0.60       0.8581   0.8751   0.7214   0.9927   0.9947   0.7812  
0.65       0.8581   0.8751   0.7214   0.9927   0.9947   0.7812  
0.70       0.8581   0.8751   0.7214   0.9927   0.9947   0.7812  
0.75       0.8581   0.8751   0.7214   0.9927   0.9947   0.7812  
0.80       0.8581   0.8751   0.7214   0.9927   0.9947   0.7812  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8581, F1=0.8751, Normal Recall=0.7214, Normal Precision=0.9927, Attack Recall=0.9947, Attack Precision=0.7812

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
0.15       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827   <--
0.20       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.25       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.30       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.35       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.40       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.45       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.50       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.55       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.60       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.65       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.70       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.75       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.80       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7470, F1=0.4403, Normal Recall=0.7194, Normal Precision=0.9992, Attack Recall=0.9951, Attack Precision=0.2827

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
0.15       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703   <--
0.20       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.25       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.30       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.35       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.40       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.45       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.50       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.55       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.60       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.65       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.70       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.75       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.80       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7749, F1=0.6387, Normal Recall=0.7199, Normal Precision=0.9982, Attack Recall=0.9947, Attack Precision=0.4703

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
0.15       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038   <--
0.20       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.25       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.30       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.35       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.40       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.45       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.50       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.55       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.60       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.65       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.70       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.75       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.80       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8026, F1=0.7515, Normal Recall=0.7203, Normal Precision=0.9969, Attack Recall=0.9947, Attack Precision=0.6038

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
0.15       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027   <--
0.20       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.25       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.30       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.35       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.40       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.45       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.50       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.55       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.60       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.65       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.70       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.75       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.80       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8296, F1=0.8236, Normal Recall=0.7195, Normal Precision=0.9952, Attack Recall=0.9947, Attack Precision=0.7027

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
0.15       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795   <--
0.20       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.25       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.30       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.35       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.40       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.45       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.50       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.55       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.60       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.65       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.70       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.75       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.80       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8566, F1=0.8740, Normal Recall=0.7185, Normal Precision=0.9927, Attack Recall=0.9947, Attack Precision=0.7795

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
0.15       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827   <--
0.20       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.25       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.30       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.35       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.40       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.45       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.50       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.55       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.60       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.65       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.70       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.75       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
0.80       0.7470   0.4403   0.7194   0.9992   0.9951   0.2827  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7470, F1=0.4403, Normal Recall=0.7194, Normal Precision=0.9992, Attack Recall=0.9951, Attack Precision=0.2827

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
0.15       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703   <--
0.20       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.25       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.30       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.35       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.40       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.45       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.50       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.55       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.60       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.65       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.70       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.75       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
0.80       0.7749   0.6387   0.7199   0.9982   0.9947   0.4703  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7749, F1=0.6387, Normal Recall=0.7199, Normal Precision=0.9982, Attack Recall=0.9947, Attack Precision=0.4703

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
0.15       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038   <--
0.20       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.25       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.30       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.35       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.40       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.45       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.50       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.55       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.60       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.65       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.70       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.75       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
0.80       0.8026   0.7515   0.7203   0.9969   0.9947   0.6038  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8026, F1=0.7515, Normal Recall=0.7203, Normal Precision=0.9969, Attack Recall=0.9947, Attack Precision=0.6038

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
0.15       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027   <--
0.20       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.25       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.30       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.35       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.40       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.45       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.50       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.55       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.60       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.65       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.70       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.75       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
0.80       0.8296   0.8236   0.7195   0.9952   0.9947   0.7027  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8296, F1=0.8236, Normal Recall=0.7195, Normal Precision=0.9952, Attack Recall=0.9947, Attack Precision=0.7027

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
0.15       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795   <--
0.20       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.25       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.30       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.35       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.40       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.45       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.50       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.55       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.60       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.65       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.70       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.75       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
0.80       0.8566   0.8740   0.7185   0.9927   0.9947   0.7795  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8566, F1=0.8740, Normal Recall=0.7185, Normal Precision=0.9927, Attack Recall=0.9947, Attack Precision=0.7795

```

