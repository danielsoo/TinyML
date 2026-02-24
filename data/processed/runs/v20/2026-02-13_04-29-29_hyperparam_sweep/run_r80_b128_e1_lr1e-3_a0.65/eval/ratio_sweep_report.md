# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e1_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-17 13:47:42 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 80 |
| **Local epochs** | 1 |
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7357 | 0.7328 | 0.7295 | 0.7261 | 0.7239 | 0.7201 | 0.7167 | 0.7137 | 0.7111 | 0.7073 | 0.7048 |
| QAT+Prune only | 0.7558 | 0.7814 | 0.8052 | 0.8290 | 0.8534 | 0.8765 | 0.9017 | 0.9247 | 0.9495 | 0.9730 | 0.9973 |
| QAT+PTQ | 0.7559 | 0.7815 | 0.8053 | 0.8291 | 0.8537 | 0.8765 | 0.9018 | 0.9247 | 0.9495 | 0.9729 | 0.9972 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7559 | 0.7815 | 0.8053 | 0.8291 | 0.8537 | 0.8765 | 0.9018 | 0.9247 | 0.9495 | 0.9729 | 0.9972 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3449 | 0.5103 | 0.6069 | 0.6713 | 0.7157 | 0.7491 | 0.7751 | 0.7961 | 0.8125 | 0.8268 |
| QAT+Prune only | 0.0000 | 0.4772 | 0.6719 | 0.7778 | 0.8448 | 0.8898 | 0.9241 | 0.9489 | 0.9693 | 0.9852 | 0.9986 |
| QAT+PTQ | 0.0000 | 0.4772 | 0.6719 | 0.7778 | 0.8450 | 0.8898 | 0.9241 | 0.9489 | 0.9693 | 0.9851 | 0.9986 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4772 | 0.6719 | 0.7778 | 0.8450 | 0.8898 | 0.9241 | 0.9489 | 0.9693 | 0.9851 | 0.9986 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7357 | 0.7361 | 0.7357 | 0.7352 | 0.7367 | 0.7353 | 0.7346 | 0.7344 | 0.7365 | 0.7300 | 0.0000 |
| QAT+Prune only | 0.7558 | 0.7574 | 0.7572 | 0.7569 | 0.7575 | 0.7558 | 0.7584 | 0.7556 | 0.7586 | 0.7546 | 0.0000 |
| QAT+PTQ | 0.7559 | 0.7575 | 0.7573 | 0.7571 | 0.7580 | 0.7559 | 0.7586 | 0.7558 | 0.7587 | 0.7549 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7559 | 0.7575 | 0.7573 | 0.7571 | 0.7580 | 0.7559 | 0.7586 | 0.7558 | 0.7587 | 0.7549 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7357 | 0.0000 | 0.0000 | 0.0000 | 0.7357 | 1.0000 |
| 90 | 10 | 299,940 | 0.7328 | 0.2285 | 0.7034 | 0.3449 | 0.7361 | 0.9571 |
| 80 | 20 | 291,350 | 0.7295 | 0.4000 | 0.7048 | 0.5103 | 0.7357 | 0.9088 |
| 70 | 30 | 194,230 | 0.7261 | 0.5329 | 0.7048 | 0.6069 | 0.7352 | 0.8532 |
| 60 | 40 | 145,675 | 0.7239 | 0.6408 | 0.7048 | 0.6713 | 0.7367 | 0.7892 |
| 50 | 50 | 116,540 | 0.7201 | 0.7270 | 0.7048 | 0.7157 | 0.7353 | 0.7135 |
| 40 | 60 | 97,115 | 0.7167 | 0.7993 | 0.7048 | 0.7491 | 0.7346 | 0.6239 |
| 30 | 70 | 83,240 | 0.7137 | 0.8609 | 0.7048 | 0.7751 | 0.7344 | 0.5160 |
| 20 | 80 | 72,835 | 0.7111 | 0.9145 | 0.7048 | 0.7961 | 0.7365 | 0.3841 |
| 10 | 90 | 64,740 | 0.7073 | 0.9592 | 0.7048 | 0.8125 | 0.7300 | 0.2155 |
| 0 | 100 | 58,270 | 0.7048 | 1.0000 | 0.7048 | 0.8268 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7558 | 0.0000 | 0.0000 | 0.0000 | 0.7558 | 1.0000 |
| 90 | 10 | 299,940 | 0.7814 | 0.3136 | 0.9976 | 0.4772 | 0.7574 | 0.9996 |
| 80 | 20 | 291,350 | 0.8052 | 0.5066 | 0.9973 | 0.6719 | 0.7572 | 0.9991 |
| 70 | 30 | 194,230 | 0.8290 | 0.6374 | 0.9973 | 0.7778 | 0.7569 | 0.9984 |
| 60 | 40 | 145,675 | 0.8534 | 0.7327 | 0.9973 | 0.8448 | 0.7575 | 0.9976 |
| 50 | 50 | 116,540 | 0.8765 | 0.8033 | 0.9973 | 0.8898 | 0.7558 | 0.9964 |
| 40 | 60 | 97,115 | 0.9017 | 0.8610 | 0.9973 | 0.9241 | 0.7584 | 0.9946 |
| 30 | 70 | 83,240 | 0.9247 | 0.9049 | 0.9973 | 0.9489 | 0.7556 | 0.9916 |
| 20 | 80 | 72,835 | 0.9495 | 0.9429 | 0.9973 | 0.9693 | 0.7586 | 0.9857 |
| 10 | 90 | 64,740 | 0.9730 | 0.9734 | 0.9973 | 0.9852 | 0.7546 | 0.9683 |
| 0 | 100 | 58,270 | 0.9973 | 1.0000 | 0.9973 | 0.9986 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7559 | 0.0000 | 0.0000 | 0.0000 | 0.7559 | 1.0000 |
| 90 | 10 | 299,940 | 0.7815 | 0.3137 | 0.9974 | 0.4772 | 0.7575 | 0.9996 |
| 80 | 20 | 291,350 | 0.8053 | 0.5067 | 0.9972 | 0.6719 | 0.7573 | 0.9991 |
| 70 | 30 | 194,230 | 0.8291 | 0.6376 | 0.9972 | 0.7778 | 0.7571 | 0.9984 |
| 60 | 40 | 145,675 | 0.8537 | 0.7331 | 0.9972 | 0.8450 | 0.7580 | 0.9975 |
| 50 | 50 | 116,540 | 0.8765 | 0.8033 | 0.9972 | 0.8898 | 0.7559 | 0.9963 |
| 40 | 60 | 97,115 | 0.9018 | 0.8611 | 0.9972 | 0.9241 | 0.7586 | 0.9944 |
| 30 | 70 | 83,240 | 0.9247 | 0.9050 | 0.9972 | 0.9489 | 0.7558 | 0.9913 |
| 20 | 80 | 72,835 | 0.9495 | 0.9430 | 0.9972 | 0.9693 | 0.7587 | 0.9853 |
| 10 | 90 | 64,740 | 0.9729 | 0.9734 | 0.9972 | 0.9851 | 0.7549 | 0.9673 |
| 0 | 100 | 58,270 | 0.9972 | 1.0000 | 0.9972 | 0.9986 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7559 | 0.0000 | 0.0000 | 0.0000 | 0.7559 | 1.0000 |
| 90 | 10 | 299,940 | 0.7815 | 0.3137 | 0.9974 | 0.4772 | 0.7575 | 0.9996 |
| 80 | 20 | 291,350 | 0.8053 | 0.5067 | 0.9972 | 0.6719 | 0.7573 | 0.9991 |
| 70 | 30 | 194,230 | 0.8291 | 0.6376 | 0.9972 | 0.7778 | 0.7571 | 0.9984 |
| 60 | 40 | 145,675 | 0.8537 | 0.7331 | 0.9972 | 0.8450 | 0.7580 | 0.9975 |
| 50 | 50 | 116,540 | 0.8765 | 0.8033 | 0.9972 | 0.8898 | 0.7559 | 0.9963 |
| 40 | 60 | 97,115 | 0.9018 | 0.8611 | 0.9972 | 0.9241 | 0.7586 | 0.9944 |
| 30 | 70 | 83,240 | 0.9247 | 0.9050 | 0.9972 | 0.9489 | 0.7558 | 0.9913 |
| 20 | 80 | 72,835 | 0.9495 | 0.9430 | 0.9972 | 0.9693 | 0.7587 | 0.9853 |
| 10 | 90 | 64,740 | 0.9729 | 0.9734 | 0.9972 | 0.9851 | 0.7549 | 0.9673 |
| 0 | 100 | 58,270 | 0.9972 | 1.0000 | 0.9972 | 0.9986 | 0.0000 | 0.0000 |


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
0.15       0.7327   0.3445   0.7361   0.9570   0.7023   0.2282   <--
0.20       0.7327   0.3445   0.7361   0.9570   0.7023   0.2282  
0.25       0.7327   0.3445   0.7361   0.9570   0.7023   0.2282  
0.30       0.7327   0.3445   0.7361   0.9570   0.7023   0.2282  
0.35       0.7327   0.3445   0.7361   0.9570   0.7023   0.2282  
0.40       0.7327   0.3445   0.7361   0.9570   0.7023   0.2282  
0.45       0.7327   0.3445   0.7361   0.9570   0.7023   0.2282  
0.50       0.7327   0.3445   0.7361   0.9570   0.7023   0.2282  
0.55       0.7327   0.3445   0.7361   0.9570   0.7023   0.2282  
0.60       0.7327   0.3445   0.7361   0.9570   0.7023   0.2282  
0.65       0.7327   0.3445   0.7361   0.9570   0.7023   0.2282  
0.70       0.7327   0.3445   0.7361   0.9570   0.7023   0.2282  
0.75       0.7327   0.3445   0.7361   0.9570   0.7023   0.2282  
0.80       0.7327   0.3445   0.7361   0.9570   0.7023   0.2282  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7327, F1=0.3445, Normal Recall=0.7361, Normal Precision=0.9570, Attack Recall=0.7023, Attack Precision=0.2282

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
0.15       0.7301   0.5109   0.7364   0.9089   0.7048   0.4007   <--
0.20       0.7301   0.5109   0.7364   0.9089   0.7048   0.4007  
0.25       0.7301   0.5109   0.7364   0.9089   0.7048   0.4007  
0.30       0.7301   0.5109   0.7364   0.9089   0.7048   0.4007  
0.35       0.7301   0.5109   0.7364   0.9089   0.7048   0.4007  
0.40       0.7301   0.5109   0.7364   0.9089   0.7048   0.4007  
0.45       0.7301   0.5109   0.7364   0.9089   0.7048   0.4007  
0.50       0.7301   0.5109   0.7364   0.9089   0.7048   0.4007  
0.55       0.7301   0.5109   0.7364   0.9089   0.7048   0.4007  
0.60       0.7301   0.5109   0.7364   0.9089   0.7048   0.4007  
0.65       0.7301   0.5109   0.7364   0.9089   0.7048   0.4007  
0.70       0.7301   0.5109   0.7364   0.9089   0.7048   0.4007  
0.75       0.7301   0.5109   0.7364   0.9089   0.7048   0.4007  
0.80       0.7301   0.5109   0.7364   0.9089   0.7048   0.4007  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7301, F1=0.5109, Normal Recall=0.7364, Normal Precision=0.9089, Attack Recall=0.7048, Attack Precision=0.4007

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
0.15       0.7272   0.6078   0.7368   0.8534   0.7048   0.5344   <--
0.20       0.7272   0.6078   0.7368   0.8534   0.7048   0.5344  
0.25       0.7272   0.6078   0.7368   0.8534   0.7048   0.5344  
0.30       0.7272   0.6078   0.7368   0.8534   0.7048   0.5344  
0.35       0.7272   0.6078   0.7368   0.8534   0.7048   0.5344  
0.40       0.7272   0.6078   0.7368   0.8534   0.7048   0.5344  
0.45       0.7272   0.6078   0.7368   0.8534   0.7048   0.5344  
0.50       0.7272   0.6078   0.7368   0.8534   0.7048   0.5344  
0.55       0.7272   0.6078   0.7368   0.8534   0.7048   0.5344  
0.60       0.7272   0.6078   0.7368   0.8534   0.7048   0.5344  
0.65       0.7272   0.6078   0.7368   0.8534   0.7048   0.5344  
0.70       0.7272   0.6078   0.7368   0.8534   0.7048   0.5344  
0.75       0.7272   0.6078   0.7368   0.8534   0.7048   0.5344  
0.80       0.7272   0.6078   0.7368   0.8534   0.7048   0.5344  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7272, F1=0.6078, Normal Recall=0.7368, Normal Precision=0.8534, Attack Recall=0.7048, Attack Precision=0.5344

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
0.15       0.7237   0.6711   0.7363   0.7891   0.7048   0.6405   <--
0.20       0.7237   0.6711   0.7363   0.7891   0.7048   0.6405  
0.25       0.7237   0.6711   0.7363   0.7891   0.7048   0.6405  
0.30       0.7237   0.6711   0.7363   0.7891   0.7048   0.6405  
0.35       0.7237   0.6711   0.7363   0.7891   0.7048   0.6405  
0.40       0.7237   0.6711   0.7363   0.7891   0.7048   0.6405  
0.45       0.7237   0.6711   0.7363   0.7891   0.7048   0.6405  
0.50       0.7237   0.6711   0.7363   0.7891   0.7048   0.6405  
0.55       0.7237   0.6711   0.7363   0.7891   0.7048   0.6405  
0.60       0.7237   0.6711   0.7363   0.7891   0.7048   0.6405  
0.65       0.7237   0.6711   0.7363   0.7891   0.7048   0.6405  
0.70       0.7237   0.6711   0.7363   0.7891   0.7048   0.6405  
0.75       0.7237   0.6711   0.7363   0.7891   0.7048   0.6405  
0.80       0.7237   0.6711   0.7363   0.7891   0.7048   0.6405  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7237, F1=0.6711, Normal Recall=0.7363, Normal Precision=0.7891, Attack Recall=0.7048, Attack Precision=0.6405

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
0.15       0.7210   0.7164   0.7372   0.7141   0.7048   0.7284   <--
0.20       0.7210   0.7164   0.7372   0.7141   0.7048   0.7284  
0.25       0.7210   0.7164   0.7372   0.7141   0.7048   0.7284  
0.30       0.7210   0.7164   0.7372   0.7141   0.7048   0.7284  
0.35       0.7210   0.7164   0.7372   0.7141   0.7048   0.7284  
0.40       0.7210   0.7164   0.7372   0.7141   0.7048   0.7284  
0.45       0.7210   0.7164   0.7372   0.7141   0.7048   0.7284  
0.50       0.7210   0.7164   0.7372   0.7141   0.7048   0.7284  
0.55       0.7210   0.7164   0.7372   0.7141   0.7048   0.7284  
0.60       0.7210   0.7164   0.7372   0.7141   0.7048   0.7284  
0.65       0.7210   0.7164   0.7372   0.7141   0.7048   0.7284  
0.70       0.7210   0.7164   0.7372   0.7141   0.7048   0.7284  
0.75       0.7210   0.7164   0.7372   0.7141   0.7048   0.7284  
0.80       0.7210   0.7164   0.7372   0.7141   0.7048   0.7284  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7210, F1=0.7164, Normal Recall=0.7372, Normal Precision=0.7141, Attack Recall=0.7048, Attack Precision=0.7284

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
0.15       0.7814   0.4772   0.7574   0.9996   0.9974   0.3136   <--
0.20       0.7814   0.4772   0.7574   0.9996   0.9974   0.3136  
0.25       0.7814   0.4772   0.7574   0.9996   0.9974   0.3136  
0.30       0.7814   0.4772   0.7574   0.9996   0.9974   0.3136  
0.35       0.7814   0.4772   0.7574   0.9996   0.9974   0.3136  
0.40       0.7814   0.4772   0.7574   0.9996   0.9974   0.3136  
0.45       0.7814   0.4772   0.7574   0.9996   0.9974   0.3136  
0.50       0.7814   0.4772   0.7574   0.9996   0.9974   0.3136  
0.55       0.7814   0.4772   0.7574   0.9996   0.9974   0.3136  
0.60       0.7814   0.4772   0.7574   0.9996   0.9974   0.3136  
0.65       0.7814   0.4772   0.7574   0.9996   0.9974   0.3136  
0.70       0.7814   0.4772   0.7574   0.9996   0.9974   0.3136  
0.75       0.7814   0.4772   0.7574   0.9996   0.9974   0.3136  
0.80       0.7814   0.4772   0.7574   0.9996   0.9974   0.3136  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7814, F1=0.4772, Normal Recall=0.7574, Normal Precision=0.9996, Attack Recall=0.9974, Attack Precision=0.3136

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
0.15       0.8058   0.6726   0.7580   0.9991   0.9973   0.5074   <--
0.20       0.8058   0.6726   0.7580   0.9991   0.9973   0.5074  
0.25       0.8058   0.6726   0.7580   0.9991   0.9973   0.5074  
0.30       0.8058   0.6726   0.7580   0.9991   0.9973   0.5074  
0.35       0.8058   0.6726   0.7580   0.9991   0.9973   0.5074  
0.40       0.8058   0.6726   0.7580   0.9991   0.9973   0.5074  
0.45       0.8058   0.6726   0.7580   0.9991   0.9973   0.5074  
0.50       0.8058   0.6726   0.7580   0.9991   0.9973   0.5074  
0.55       0.8058   0.6726   0.7580   0.9991   0.9973   0.5074  
0.60       0.8058   0.6726   0.7580   0.9991   0.9973   0.5074  
0.65       0.8058   0.6726   0.7580   0.9991   0.9973   0.5074  
0.70       0.8058   0.6726   0.7580   0.9991   0.9973   0.5074  
0.75       0.8058   0.6726   0.7580   0.9991   0.9973   0.5074  
0.80       0.8058   0.6726   0.7580   0.9991   0.9973   0.5074  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8058, F1=0.6726, Normal Recall=0.7580, Normal Precision=0.9991, Attack Recall=0.9973, Attack Precision=0.5074

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
0.15       0.8290   0.7777   0.7568   0.9984   0.9973   0.6374   <--
0.20       0.8290   0.7777   0.7568   0.9984   0.9973   0.6374  
0.25       0.8290   0.7777   0.7568   0.9984   0.9973   0.6374  
0.30       0.8290   0.7777   0.7568   0.9984   0.9973   0.6374  
0.35       0.8290   0.7777   0.7568   0.9984   0.9973   0.6374  
0.40       0.8290   0.7777   0.7568   0.9984   0.9973   0.6374  
0.45       0.8290   0.7777   0.7568   0.9984   0.9973   0.6374  
0.50       0.8290   0.7777   0.7568   0.9984   0.9973   0.6374  
0.55       0.8290   0.7777   0.7568   0.9984   0.9973   0.6374  
0.60       0.8290   0.7777   0.7568   0.9984   0.9973   0.6374  
0.65       0.8290   0.7777   0.7568   0.9984   0.9973   0.6374  
0.70       0.8290   0.7777   0.7568   0.9984   0.9973   0.6374  
0.75       0.8290   0.7777   0.7568   0.9984   0.9973   0.6374  
0.80       0.8290   0.7777   0.7568   0.9984   0.9973   0.6374  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8290, F1=0.7777, Normal Recall=0.7568, Normal Precision=0.9984, Attack Recall=0.9973, Attack Precision=0.6374

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
0.15       0.8525   0.8440   0.7560   0.9976   0.9973   0.7315   <--
0.20       0.8525   0.8440   0.7560   0.9976   0.9973   0.7315  
0.25       0.8525   0.8440   0.7560   0.9976   0.9973   0.7315  
0.30       0.8525   0.8440   0.7560   0.9976   0.9973   0.7315  
0.35       0.8525   0.8440   0.7560   0.9976   0.9973   0.7315  
0.40       0.8525   0.8440   0.7560   0.9976   0.9973   0.7315  
0.45       0.8525   0.8440   0.7560   0.9976   0.9973   0.7315  
0.50       0.8525   0.8440   0.7560   0.9976   0.9973   0.7315  
0.55       0.8525   0.8440   0.7560   0.9976   0.9973   0.7315  
0.60       0.8525   0.8440   0.7560   0.9976   0.9973   0.7315  
0.65       0.8525   0.8440   0.7560   0.9976   0.9973   0.7315  
0.70       0.8525   0.8440   0.7560   0.9976   0.9973   0.7315  
0.75       0.8525   0.8440   0.7560   0.9976   0.9973   0.7315  
0.80       0.8525   0.8440   0.7560   0.9976   0.9973   0.7315  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8525, F1=0.8440, Normal Recall=0.7560, Normal Precision=0.9976, Attack Recall=0.9973, Attack Precision=0.7315

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
0.15       0.8758   0.8893   0.7544   0.9964   0.9973   0.8024   <--
0.20       0.8758   0.8893   0.7544   0.9964   0.9973   0.8024  
0.25       0.8758   0.8893   0.7544   0.9964   0.9973   0.8024  
0.30       0.8758   0.8893   0.7544   0.9964   0.9973   0.8024  
0.35       0.8758   0.8893   0.7544   0.9964   0.9973   0.8024  
0.40       0.8758   0.8893   0.7544   0.9964   0.9973   0.8024  
0.45       0.8758   0.8893   0.7544   0.9964   0.9973   0.8024  
0.50       0.8758   0.8893   0.7544   0.9964   0.9973   0.8024  
0.55       0.8758   0.8893   0.7544   0.9964   0.9973   0.8024  
0.60       0.8758   0.8893   0.7544   0.9964   0.9973   0.8024  
0.65       0.8758   0.8893   0.7544   0.9964   0.9973   0.8024  
0.70       0.8758   0.8893   0.7544   0.9964   0.9973   0.8024  
0.75       0.8758   0.8893   0.7544   0.9964   0.9973   0.8024  
0.80       0.8758   0.8893   0.7544   0.9964   0.9973   0.8024  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8758, F1=0.8893, Normal Recall=0.7544, Normal Precision=0.9964, Attack Recall=0.9973, Attack Precision=0.8024

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
0.15       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137   <--
0.20       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.25       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.30       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.35       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.40       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.45       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.50       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.55       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.60       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.65       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.70       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.75       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.80       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7815, F1=0.4772, Normal Recall=0.7575, Normal Precision=0.9996, Attack Recall=0.9974, Attack Precision=0.3137

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
0.15       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076   <--
0.20       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.25       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.30       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.35       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.40       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.45       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.50       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.55       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.60       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.65       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.70       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.75       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.80       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8060, F1=0.6727, Normal Recall=0.7581, Normal Precision=0.9991, Attack Recall=0.9972, Attack Precision=0.5076

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
0.15       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376   <--
0.20       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.25       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.30       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.35       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.40       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.45       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.50       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.55       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.60       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.65       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.70       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.75       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.80       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8291, F1=0.7778, Normal Recall=0.7571, Normal Precision=0.9984, Attack Recall=0.9972, Attack Precision=0.6376

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
0.15       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315   <--
0.20       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.25       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.30       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.35       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.40       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.45       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.50       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.55       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.60       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.65       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.70       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.75       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.80       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8525, F1=0.8439, Normal Recall=0.7560, Normal Precision=0.9975, Attack Recall=0.9972, Attack Precision=0.7315

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
0.15       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023   <--
0.20       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.25       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.30       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.35       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.40       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.45       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.50       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.55       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.60       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.65       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.70       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.75       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.80       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8757, F1=0.8892, Normal Recall=0.7542, Normal Precision=0.9963, Attack Recall=0.9972, Attack Precision=0.8023

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
0.15       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137   <--
0.20       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.25       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.30       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.35       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.40       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.45       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.50       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.55       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.60       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.65       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.70       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.75       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
0.80       0.7815   0.4772   0.7575   0.9996   0.9974   0.3137  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7815, F1=0.4772, Normal Recall=0.7575, Normal Precision=0.9996, Attack Recall=0.9974, Attack Precision=0.3137

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
0.15       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076   <--
0.20       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.25       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.30       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.35       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.40       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.45       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.50       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.55       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.60       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.65       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.70       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.75       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
0.80       0.8060   0.6727   0.7581   0.9991   0.9972   0.5076  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8060, F1=0.6727, Normal Recall=0.7581, Normal Precision=0.9991, Attack Recall=0.9972, Attack Precision=0.5076

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
0.15       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376   <--
0.20       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.25       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.30       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.35       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.40       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.45       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.50       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.55       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.60       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.65       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.70       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.75       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
0.80       0.8291   0.7778   0.7571   0.9984   0.9972   0.6376  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8291, F1=0.7778, Normal Recall=0.7571, Normal Precision=0.9984, Attack Recall=0.9972, Attack Precision=0.6376

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
0.15       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315   <--
0.20       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.25       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.30       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.35       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.40       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.45       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.50       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.55       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.60       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.65       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.70       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.75       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
0.80       0.8525   0.8439   0.7560   0.9975   0.9972   0.7315  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8525, F1=0.8439, Normal Recall=0.7560, Normal Precision=0.9975, Attack Recall=0.9972, Attack Precision=0.7315

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
0.15       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023   <--
0.20       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.25       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.30       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.35       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.40       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.45       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.50       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.55       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.60       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.65       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.70       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.75       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
0.80       0.8757   0.8892   0.7542   0.9963   0.9972   0.8023  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8757, F1=0.8892, Normal Recall=0.7542, Normal Precision=0.9963, Attack Recall=0.9972, Attack Precision=0.8023

```

