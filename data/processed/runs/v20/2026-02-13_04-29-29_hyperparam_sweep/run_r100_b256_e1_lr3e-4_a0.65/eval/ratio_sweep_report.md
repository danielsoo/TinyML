# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e1_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-22 03:02:18 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4030 | 0.4627 | 0.5211 | 0.5807 | 0.6410 | 0.6970 | 0.7564 | 0.8143 | 0.8732 | 0.9319 | 0.9902 |
| QAT+Prune only | 0.4233 | 0.4800 | 0.5366 | 0.5935 | 0.6517 | 0.7083 | 0.7641 | 0.8213 | 0.8786 | 0.9348 | 0.9924 |
| QAT+PTQ | 0.4212 | 0.4780 | 0.5347 | 0.5918 | 0.6501 | 0.7072 | 0.7631 | 0.8206 | 0.8782 | 0.9346 | 0.9924 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.4212 | 0.4780 | 0.5347 | 0.5918 | 0.6501 | 0.7072 | 0.7631 | 0.8206 | 0.8782 | 0.9346 | 0.9924 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2694 | 0.4527 | 0.5863 | 0.6881 | 0.7657 | 0.8299 | 0.8819 | 0.9259 | 0.9632 | 0.9951 |
| QAT+Prune only | 0.0000 | 0.2763 | 0.4614 | 0.5943 | 0.6951 | 0.7728 | 0.8346 | 0.8860 | 0.9290 | 0.9648 | 0.9962 |
| QAT+PTQ | 0.0000 | 0.2755 | 0.4604 | 0.5933 | 0.6941 | 0.7721 | 0.8341 | 0.8856 | 0.9288 | 0.9647 | 0.9962 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2755 | 0.4604 | 0.5933 | 0.6941 | 0.7721 | 0.8341 | 0.8856 | 0.9288 | 0.9647 | 0.9962 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4030 | 0.4041 | 0.4038 | 0.4052 | 0.4082 | 0.4037 | 0.4058 | 0.4040 | 0.4052 | 0.4075 | 0.0000 |
| QAT+Prune only | 0.4233 | 0.4231 | 0.4227 | 0.4225 | 0.4246 | 0.4242 | 0.4216 | 0.4221 | 0.4234 | 0.4161 | 0.0000 |
| QAT+PTQ | 0.4212 | 0.4208 | 0.4203 | 0.4201 | 0.4219 | 0.4219 | 0.4192 | 0.4197 | 0.4216 | 0.4147 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.4212 | 0.4208 | 0.4203 | 0.4201 | 0.4219 | 0.4219 | 0.4192 | 0.4197 | 0.4216 | 0.4147 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4030 | 0.0000 | 0.0000 | 0.0000 | 0.4030 | 1.0000 |
| 90 | 10 | 299,940 | 0.4627 | 0.1559 | 0.9904 | 0.2694 | 0.4041 | 0.9974 |
| 80 | 20 | 291,350 | 0.5211 | 0.2934 | 0.9902 | 0.4527 | 0.4038 | 0.9940 |
| 70 | 30 | 194,230 | 0.5807 | 0.4164 | 0.9902 | 0.5863 | 0.4052 | 0.9897 |
| 60 | 40 | 145,675 | 0.6410 | 0.5273 | 0.9902 | 0.6881 | 0.4082 | 0.9842 |
| 50 | 50 | 116,540 | 0.6970 | 0.6241 | 0.9902 | 0.7657 | 0.4037 | 0.9763 |
| 40 | 60 | 97,115 | 0.7564 | 0.7143 | 0.9902 | 0.8299 | 0.4058 | 0.9650 |
| 30 | 70 | 83,240 | 0.8143 | 0.7949 | 0.9902 | 0.8819 | 0.4040 | 0.9463 |
| 20 | 80 | 72,835 | 0.8732 | 0.8694 | 0.9902 | 0.9259 | 0.4052 | 0.9116 |
| 10 | 90 | 64,740 | 0.9319 | 0.9377 | 0.9902 | 0.9632 | 0.4075 | 0.8218 |
| 0 | 100 | 58,270 | 0.9902 | 1.0000 | 0.9902 | 0.9951 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4233 | 0.0000 | 0.0000 | 0.0000 | 0.4233 | 1.0000 |
| 90 | 10 | 299,940 | 0.4800 | 0.1605 | 0.9927 | 0.2763 | 0.4231 | 0.9981 |
| 80 | 20 | 291,350 | 0.5366 | 0.3006 | 0.9924 | 0.4614 | 0.4227 | 0.9955 |
| 70 | 30 | 194,230 | 0.5935 | 0.4241 | 0.9924 | 0.5943 | 0.4225 | 0.9923 |
| 60 | 40 | 145,675 | 0.6517 | 0.5348 | 0.9924 | 0.6951 | 0.4246 | 0.9882 |
| 50 | 50 | 116,540 | 0.7083 | 0.6328 | 0.9924 | 0.7728 | 0.4242 | 0.9824 |
| 40 | 60 | 97,115 | 0.7641 | 0.7202 | 0.9924 | 0.8346 | 0.4216 | 0.9737 |
| 30 | 70 | 83,240 | 0.8213 | 0.8003 | 0.9924 | 0.8860 | 0.4221 | 0.9597 |
| 20 | 80 | 72,835 | 0.8786 | 0.8732 | 0.9924 | 0.9290 | 0.4234 | 0.9330 |
| 10 | 90 | 64,740 | 0.9348 | 0.9386 | 0.9924 | 0.9648 | 0.4161 | 0.8588 |
| 0 | 100 | 58,270 | 0.9924 | 1.0000 | 0.9924 | 0.9962 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4212 | 0.0000 | 0.0000 | 0.0000 | 0.4212 | 1.0000 |
| 90 | 10 | 299,940 | 0.4780 | 0.1600 | 0.9927 | 0.2755 | 0.4208 | 0.9981 |
| 80 | 20 | 291,350 | 0.5347 | 0.2997 | 0.9924 | 0.4604 | 0.4203 | 0.9955 |
| 70 | 30 | 194,230 | 0.5918 | 0.4231 | 0.9924 | 0.5933 | 0.4201 | 0.9923 |
| 60 | 40 | 145,675 | 0.6501 | 0.5337 | 0.9924 | 0.6941 | 0.4219 | 0.9881 |
| 50 | 50 | 116,540 | 0.7072 | 0.6319 | 0.9924 | 0.7721 | 0.4219 | 0.9823 |
| 40 | 60 | 97,115 | 0.7631 | 0.7193 | 0.9924 | 0.8341 | 0.4192 | 0.9735 |
| 30 | 70 | 83,240 | 0.8206 | 0.7996 | 0.9924 | 0.8856 | 0.4197 | 0.9594 |
| 20 | 80 | 72,835 | 0.8782 | 0.8728 | 0.9924 | 0.9288 | 0.4216 | 0.9327 |
| 10 | 90 | 64,740 | 0.9346 | 0.9385 | 0.9924 | 0.9647 | 0.4147 | 0.8584 |
| 0 | 100 | 58,270 | 0.9924 | 1.0000 | 0.9924 | 0.9962 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.4212 | 0.0000 | 0.0000 | 0.0000 | 0.4212 | 1.0000 |
| 90 | 10 | 299,940 | 0.4780 | 0.1600 | 0.9927 | 0.2755 | 0.4208 | 0.9981 |
| 80 | 20 | 291,350 | 0.5347 | 0.2997 | 0.9924 | 0.4604 | 0.4203 | 0.9955 |
| 70 | 30 | 194,230 | 0.5918 | 0.4231 | 0.9924 | 0.5933 | 0.4201 | 0.9923 |
| 60 | 40 | 145,675 | 0.6501 | 0.5337 | 0.9924 | 0.6941 | 0.4219 | 0.9881 |
| 50 | 50 | 116,540 | 0.7072 | 0.6319 | 0.9924 | 0.7721 | 0.4219 | 0.9823 |
| 40 | 60 | 97,115 | 0.7631 | 0.7193 | 0.9924 | 0.8341 | 0.4192 | 0.9735 |
| 30 | 70 | 83,240 | 0.8206 | 0.7996 | 0.9924 | 0.8856 | 0.4197 | 0.9594 |
| 20 | 80 | 72,835 | 0.8782 | 0.8728 | 0.9924 | 0.9288 | 0.4216 | 0.9327 |
| 10 | 90 | 64,740 | 0.9346 | 0.9385 | 0.9924 | 0.9647 | 0.4147 | 0.8584 |
| 0 | 100 | 58,270 | 0.9924 | 1.0000 | 0.9924 | 0.9962 | 0.0000 | 0.0000 |


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
0.15       0.4628   0.2695   0.4041   0.9975   0.9909   0.1560   <--
0.20       0.4628   0.2695   0.4041   0.9975   0.9909   0.1560  
0.25       0.4628   0.2695   0.4041   0.9975   0.9909   0.1560  
0.30       0.4628   0.2695   0.4041   0.9975   0.9909   0.1560  
0.35       0.4628   0.2695   0.4041   0.9975   0.9909   0.1560  
0.40       0.4628   0.2695   0.4041   0.9975   0.9909   0.1560  
0.45       0.4628   0.2695   0.4041   0.9975   0.9909   0.1560  
0.50       0.4628   0.2695   0.4041   0.9975   0.9909   0.1560  
0.55       0.4628   0.2695   0.4041   0.9975   0.9909   0.1560  
0.60       0.4628   0.2695   0.4041   0.9975   0.9909   0.1560  
0.65       0.4628   0.2695   0.4041   0.9975   0.9909   0.1560  
0.70       0.4628   0.2695   0.4041   0.9975   0.9909   0.1560  
0.75       0.4628   0.2695   0.4041   0.9975   0.9909   0.1560  
0.80       0.4628   0.2695   0.4041   0.9975   0.9909   0.1560  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4628, F1=0.2695, Normal Recall=0.4041, Normal Precision=0.9975, Attack Recall=0.9909, Attack Precision=0.1560

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
0.15       0.5210   0.4526   0.4037   0.9940   0.9902   0.2934   <--
0.20       0.5210   0.4526   0.4037   0.9940   0.9902   0.2934  
0.25       0.5210   0.4526   0.4037   0.9940   0.9902   0.2934  
0.30       0.5210   0.4526   0.4037   0.9940   0.9902   0.2934  
0.35       0.5210   0.4526   0.4037   0.9940   0.9902   0.2934  
0.40       0.5210   0.4526   0.4037   0.9940   0.9902   0.2934  
0.45       0.5210   0.4526   0.4037   0.9940   0.9902   0.2934  
0.50       0.5210   0.4526   0.4037   0.9940   0.9902   0.2934  
0.55       0.5210   0.4526   0.4037   0.9940   0.9902   0.2934  
0.60       0.5210   0.4526   0.4037   0.9940   0.9902   0.2934  
0.65       0.5210   0.4526   0.4037   0.9940   0.9902   0.2934  
0.70       0.5210   0.4526   0.4037   0.9940   0.9902   0.2934  
0.75       0.5210   0.4526   0.4037   0.9940   0.9902   0.2934  
0.80       0.5210   0.4526   0.4037   0.9940   0.9902   0.2934  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5210, F1=0.4526, Normal Recall=0.4037, Normal Precision=0.9940, Attack Recall=0.9902, Attack Precision=0.2934

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
0.15       0.5794   0.5855   0.4034   0.9897   0.9902   0.4156   <--
0.20       0.5794   0.5855   0.4034   0.9897   0.9902   0.4156  
0.25       0.5794   0.5855   0.4034   0.9897   0.9902   0.4156  
0.30       0.5794   0.5855   0.4034   0.9897   0.9902   0.4156  
0.35       0.5794   0.5855   0.4034   0.9897   0.9902   0.4156  
0.40       0.5794   0.5855   0.4034   0.9897   0.9902   0.4156  
0.45       0.5794   0.5855   0.4034   0.9897   0.9902   0.4156  
0.50       0.5794   0.5855   0.4034   0.9897   0.9902   0.4156  
0.55       0.5794   0.5855   0.4034   0.9897   0.9902   0.4156  
0.60       0.5794   0.5855   0.4034   0.9897   0.9902   0.4156  
0.65       0.5794   0.5855   0.4034   0.9897   0.9902   0.4156  
0.70       0.5794   0.5855   0.4034   0.9897   0.9902   0.4156  
0.75       0.5794   0.5855   0.4034   0.9897   0.9902   0.4156  
0.80       0.5794   0.5855   0.4034   0.9897   0.9902   0.4156  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5794, F1=0.5855, Normal Recall=0.4034, Normal Precision=0.9897, Attack Recall=0.9902, Attack Precision=0.4156

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
0.15       0.6379   0.6863   0.4031   0.9840   0.9902   0.5252   <--
0.20       0.6379   0.6863   0.4031   0.9840   0.9902   0.5252  
0.25       0.6379   0.6863   0.4031   0.9840   0.9902   0.5252  
0.30       0.6379   0.6863   0.4031   0.9840   0.9902   0.5252  
0.35       0.6379   0.6863   0.4031   0.9840   0.9902   0.5252  
0.40       0.6379   0.6863   0.4031   0.9840   0.9902   0.5252  
0.45       0.6379   0.6863   0.4031   0.9840   0.9902   0.5252  
0.50       0.6379   0.6863   0.4031   0.9840   0.9902   0.5252  
0.55       0.6379   0.6863   0.4031   0.9840   0.9902   0.5252  
0.60       0.6379   0.6863   0.4031   0.9840   0.9902   0.5252  
0.65       0.6379   0.6863   0.4031   0.9840   0.9902   0.5252  
0.70       0.6379   0.6863   0.4031   0.9840   0.9902   0.5252  
0.75       0.6379   0.6863   0.4031   0.9840   0.9902   0.5252  
0.80       0.6379   0.6863   0.4031   0.9840   0.9902   0.5252  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6379, F1=0.6863, Normal Recall=0.4031, Normal Precision=0.9840, Attack Recall=0.9902, Attack Precision=0.5252

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
0.15       0.6962   0.7652   0.4023   0.9762   0.9902   0.6236   <--
0.20       0.6962   0.7652   0.4023   0.9762   0.9902   0.6236  
0.25       0.6962   0.7652   0.4023   0.9762   0.9902   0.6236  
0.30       0.6962   0.7652   0.4023   0.9762   0.9902   0.6236  
0.35       0.6962   0.7652   0.4023   0.9762   0.9902   0.6236  
0.40       0.6962   0.7652   0.4023   0.9762   0.9902   0.6236  
0.45       0.6962   0.7652   0.4023   0.9762   0.9902   0.6236  
0.50       0.6962   0.7652   0.4023   0.9762   0.9902   0.6236  
0.55       0.6962   0.7652   0.4023   0.9762   0.9902   0.6236  
0.60       0.6962   0.7652   0.4023   0.9762   0.9902   0.6236  
0.65       0.6962   0.7652   0.4023   0.9762   0.9902   0.6236  
0.70       0.6962   0.7652   0.4023   0.9762   0.9902   0.6236  
0.75       0.6962   0.7652   0.4023   0.9762   0.9902   0.6236  
0.80       0.6962   0.7652   0.4023   0.9762   0.9902   0.6236  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6962, F1=0.7652, Normal Recall=0.4023, Normal Precision=0.9762, Attack Recall=0.9902, Attack Precision=0.6236

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
0.15       0.4800   0.2763   0.4231   0.9981   0.9926   0.1605   <--
0.20       0.4800   0.2763   0.4231   0.9981   0.9926   0.1605  
0.25       0.4800   0.2763   0.4231   0.9981   0.9926   0.1605  
0.30       0.4800   0.2763   0.4231   0.9981   0.9926   0.1605  
0.35       0.4800   0.2763   0.4231   0.9981   0.9926   0.1605  
0.40       0.4800   0.2763   0.4231   0.9981   0.9926   0.1605  
0.45       0.4800   0.2763   0.4231   0.9981   0.9926   0.1605  
0.50       0.4800   0.2763   0.4231   0.9981   0.9926   0.1605  
0.55       0.4800   0.2763   0.4231   0.9981   0.9926   0.1605  
0.60       0.4800   0.2763   0.4231   0.9981   0.9926   0.1605  
0.65       0.4800   0.2763   0.4231   0.9981   0.9926   0.1605  
0.70       0.4800   0.2763   0.4231   0.9981   0.9926   0.1605  
0.75       0.4800   0.2763   0.4231   0.9981   0.9926   0.1605  
0.80       0.4800   0.2763   0.4231   0.9981   0.9926   0.1605  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4800, F1=0.2763, Normal Recall=0.4231, Normal Precision=0.9981, Attack Recall=0.9926, Attack Precision=0.1605

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
0.15       0.5371   0.4617   0.4233   0.9955   0.9924   0.3008   <--
0.20       0.5371   0.4617   0.4233   0.9955   0.9924   0.3008  
0.25       0.5371   0.4617   0.4233   0.9955   0.9924   0.3008  
0.30       0.5371   0.4617   0.4233   0.9955   0.9924   0.3008  
0.35       0.5371   0.4617   0.4233   0.9955   0.9924   0.3008  
0.40       0.5371   0.4617   0.4233   0.9955   0.9924   0.3008  
0.45       0.5371   0.4617   0.4233   0.9955   0.9924   0.3008  
0.50       0.5371   0.4617   0.4233   0.9955   0.9924   0.3008  
0.55       0.5371   0.4617   0.4233   0.9955   0.9924   0.3008  
0.60       0.5371   0.4617   0.4233   0.9955   0.9924   0.3008  
0.65       0.5371   0.4617   0.4233   0.9955   0.9924   0.3008  
0.70       0.5371   0.4617   0.4233   0.9955   0.9924   0.3008  
0.75       0.5371   0.4617   0.4233   0.9955   0.9924   0.3008  
0.80       0.5371   0.4617   0.4233   0.9955   0.9924   0.3008  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5371, F1=0.4617, Normal Recall=0.4233, Normal Precision=0.9955, Attack Recall=0.9924, Attack Precision=0.3008

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
0.15       0.5934   0.5942   0.4224   0.9923   0.9924   0.4241   <--
0.20       0.5934   0.5942   0.4224   0.9923   0.9924   0.4241  
0.25       0.5934   0.5942   0.4224   0.9923   0.9924   0.4241  
0.30       0.5934   0.5942   0.4224   0.9923   0.9924   0.4241  
0.35       0.5934   0.5942   0.4224   0.9923   0.9924   0.4241  
0.40       0.5934   0.5942   0.4224   0.9923   0.9924   0.4241  
0.45       0.5934   0.5942   0.4224   0.9923   0.9924   0.4241  
0.50       0.5934   0.5942   0.4224   0.9923   0.9924   0.4241  
0.55       0.5934   0.5942   0.4224   0.9923   0.9924   0.4241  
0.60       0.5934   0.5942   0.4224   0.9923   0.9924   0.4241  
0.65       0.5934   0.5942   0.4224   0.9923   0.9924   0.4241  
0.70       0.5934   0.5942   0.4224   0.9923   0.9924   0.4241  
0.75       0.5934   0.5942   0.4224   0.9923   0.9924   0.4241  
0.80       0.5934   0.5942   0.4224   0.9923   0.9924   0.4241  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5934, F1=0.5942, Normal Recall=0.4224, Normal Precision=0.9923, Attack Recall=0.9924, Attack Precision=0.4241

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
0.15       0.6510   0.6946   0.4234   0.9882   0.9924   0.5343   <--
0.20       0.6510   0.6946   0.4234   0.9882   0.9924   0.5343  
0.25       0.6510   0.6946   0.4234   0.9882   0.9924   0.5343  
0.30       0.6510   0.6946   0.4234   0.9882   0.9924   0.5343  
0.35       0.6510   0.6946   0.4234   0.9882   0.9924   0.5343  
0.40       0.6510   0.6946   0.4234   0.9882   0.9924   0.5343  
0.45       0.6510   0.6946   0.4234   0.9882   0.9924   0.5343  
0.50       0.6510   0.6946   0.4234   0.9882   0.9924   0.5343  
0.55       0.6510   0.6946   0.4234   0.9882   0.9924   0.5343  
0.60       0.6510   0.6946   0.4234   0.9882   0.9924   0.5343  
0.65       0.6510   0.6946   0.4234   0.9882   0.9924   0.5343  
0.70       0.6510   0.6946   0.4234   0.9882   0.9924   0.5343  
0.75       0.6510   0.6946   0.4234   0.9882   0.9924   0.5343  
0.80       0.6510   0.6946   0.4234   0.9882   0.9924   0.5343  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6510, F1=0.6946, Normal Recall=0.4234, Normal Precision=0.9882, Attack Recall=0.9924, Attack Precision=0.5343

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
0.15       0.7077   0.7725   0.4230   0.9823   0.9924   0.6323   <--
0.20       0.7077   0.7725   0.4230   0.9823   0.9924   0.6323  
0.25       0.7077   0.7725   0.4230   0.9823   0.9924   0.6323  
0.30       0.7077   0.7725   0.4230   0.9823   0.9924   0.6323  
0.35       0.7077   0.7725   0.4230   0.9823   0.9924   0.6323  
0.40       0.7077   0.7725   0.4230   0.9823   0.9924   0.6323  
0.45       0.7077   0.7725   0.4230   0.9823   0.9924   0.6323  
0.50       0.7077   0.7725   0.4230   0.9823   0.9924   0.6323  
0.55       0.7077   0.7725   0.4230   0.9823   0.9924   0.6323  
0.60       0.7077   0.7725   0.4230   0.9823   0.9924   0.6323  
0.65       0.7077   0.7725   0.4230   0.9823   0.9924   0.6323  
0.70       0.7077   0.7725   0.4230   0.9823   0.9924   0.6323  
0.75       0.7077   0.7725   0.4230   0.9823   0.9924   0.6323  
0.80       0.7077   0.7725   0.4230   0.9823   0.9924   0.6323  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7077, F1=0.7725, Normal Recall=0.4230, Normal Precision=0.9823, Attack Recall=0.9924, Attack Precision=0.6323

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
0.15       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600   <--
0.20       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.25       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.30       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.35       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.40       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.45       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.50       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.55       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.60       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.65       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.70       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.75       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.80       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4780, F1=0.2755, Normal Recall=0.4208, Normal Precision=0.9980, Attack Recall=0.9926, Attack Precision=0.1600

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
0.15       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000   <--
0.20       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.25       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.30       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.35       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.40       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.45       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.50       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.55       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.60       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.65       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.70       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.75       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.80       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5353, F1=0.4607, Normal Recall=0.4210, Normal Precision=0.9955, Attack Recall=0.9924, Attack Precision=0.3000

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
0.15       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231   <--
0.20       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.25       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.30       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.35       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.40       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.45       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.50       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.55       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.60       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.65       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.70       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.75       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.80       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5918, F1=0.5933, Normal Recall=0.4202, Normal Precision=0.9923, Attack Recall=0.9924, Attack Precision=0.4231

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
0.15       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334   <--
0.20       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.25       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.30       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.35       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.40       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.45       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.50       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.55       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.60       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.65       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.70       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.75       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.80       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6497, F1=0.6939, Normal Recall=0.4213, Normal Precision=0.9881, Attack Recall=0.9924, Attack Precision=0.5334

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
0.15       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315   <--
0.20       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.25       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.30       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.35       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.40       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.45       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.50       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.55       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.60       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.65       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.70       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.75       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.80       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7066, F1=0.7718, Normal Recall=0.4208, Normal Precision=0.9823, Attack Recall=0.9924, Attack Precision=0.6315

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
0.15       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600   <--
0.20       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.25       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.30       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.35       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.40       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.45       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.50       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.55       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.60       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.65       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.70       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.75       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
0.80       0.4780   0.2755   0.4208   0.9980   0.9926   0.1600  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4780, F1=0.2755, Normal Recall=0.4208, Normal Precision=0.9980, Attack Recall=0.9926, Attack Precision=0.1600

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
0.15       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000   <--
0.20       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.25       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.30       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.35       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.40       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.45       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.50       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.55       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.60       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.65       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.70       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.75       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
0.80       0.5353   0.4607   0.4210   0.9955   0.9924   0.3000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5353, F1=0.4607, Normal Recall=0.4210, Normal Precision=0.9955, Attack Recall=0.9924, Attack Precision=0.3000

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
0.15       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231   <--
0.20       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.25       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.30       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.35       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.40       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.45       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.50       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.55       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.60       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.65       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.70       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.75       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
0.80       0.5918   0.5933   0.4202   0.9923   0.9924   0.4231  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5918, F1=0.5933, Normal Recall=0.4202, Normal Precision=0.9923, Attack Recall=0.9924, Attack Precision=0.4231

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
0.15       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334   <--
0.20       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.25       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.30       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.35       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.40       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.45       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.50       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.55       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.60       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.65       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.70       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.75       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
0.80       0.6497   0.6939   0.4213   0.9881   0.9924   0.5334  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6497, F1=0.6939, Normal Recall=0.4213, Normal Precision=0.9881, Attack Recall=0.9924, Attack Precision=0.5334

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
0.15       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315   <--
0.20       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.25       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.30       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.35       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.40       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.45       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.50       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.55       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.60       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.65       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.70       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.75       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
0.80       0.7066   0.7718   0.4208   0.9823   0.9924   0.6315  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7066, F1=0.7718, Normal Recall=0.4208, Normal Precision=0.9823, Attack Recall=0.9924, Attack Precision=0.6315

```

