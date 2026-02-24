# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e2_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-18 04:21:01 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1730 | 0.2237 | 0.2740 | 0.3231 | 0.3731 | 0.4238 | 0.4727 | 0.5231 | 0.5726 | 0.6233 | 0.6731 |
| QAT+Prune only | 0.7134 | 0.7413 | 0.7693 | 0.7989 | 0.8265 | 0.8532 | 0.8839 | 0.9120 | 0.9395 | 0.9685 | 0.9973 |
| QAT+PTQ | 0.7112 | 0.7393 | 0.7675 | 0.7974 | 0.8251 | 0.8520 | 0.8830 | 0.9113 | 0.9390 | 0.9682 | 0.9973 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7112 | 0.7393 | 0.7675 | 0.7974 | 0.8251 | 0.8520 | 0.8830 | 0.9113 | 0.9390 | 0.9682 | 0.9973 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1477 | 0.2705 | 0.3737 | 0.4621 | 0.5388 | 0.6050 | 0.6640 | 0.7159 | 0.7628 | 0.8046 |
| QAT+Prune only | 0.0000 | 0.4354 | 0.6336 | 0.7485 | 0.8213 | 0.8717 | 0.9116 | 0.9407 | 0.9635 | 0.9827 | 0.9986 |
| QAT+PTQ | 0.0000 | 0.4335 | 0.6318 | 0.7471 | 0.8202 | 0.8708 | 0.9110 | 0.9403 | 0.9632 | 0.9826 | 0.9986 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4335 | 0.6318 | 0.7471 | 0.8202 | 0.8708 | 0.9110 | 0.9403 | 0.9632 | 0.9826 | 0.9986 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1730 | 0.1738 | 0.1742 | 0.1731 | 0.1731 | 0.1745 | 0.1722 | 0.1732 | 0.1707 | 0.1750 | 0.0000 |
| QAT+Prune only | 0.7134 | 0.7129 | 0.7123 | 0.7140 | 0.7126 | 0.7090 | 0.7138 | 0.7129 | 0.7083 | 0.7093 | 0.0000 |
| QAT+PTQ | 0.7112 | 0.7106 | 0.7101 | 0.7118 | 0.7103 | 0.7068 | 0.7117 | 0.7108 | 0.7060 | 0.7070 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7112 | 0.7106 | 0.7101 | 0.7118 | 0.7103 | 0.7068 | 0.7117 | 0.7108 | 0.7060 | 0.7070 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1730 | 0.0000 | 0.0000 | 0.0000 | 0.1730 | 1.0000 |
| 90 | 10 | 299,940 | 0.2237 | 0.0829 | 0.6725 | 0.1477 | 0.1738 | 0.8269 |
| 80 | 20 | 291,350 | 0.2740 | 0.1693 | 0.6731 | 0.2705 | 0.1742 | 0.6806 |
| 70 | 30 | 194,230 | 0.3231 | 0.2586 | 0.6731 | 0.3737 | 0.1731 | 0.5527 |
| 60 | 40 | 145,675 | 0.3731 | 0.3518 | 0.6731 | 0.4621 | 0.1731 | 0.4427 |
| 50 | 50 | 116,540 | 0.4238 | 0.4492 | 0.6731 | 0.5388 | 0.1745 | 0.3481 |
| 40 | 60 | 97,115 | 0.4727 | 0.5495 | 0.6731 | 0.6050 | 0.1722 | 0.2599 |
| 30 | 70 | 83,240 | 0.5231 | 0.6551 | 0.6731 | 0.6640 | 0.1732 | 0.1851 |
| 20 | 80 | 72,835 | 0.5726 | 0.7645 | 0.6731 | 0.7159 | 0.1707 | 0.1155 |
| 10 | 90 | 64,740 | 0.6233 | 0.8801 | 0.6731 | 0.7628 | 0.1750 | 0.0561 |
| 0 | 100 | 58,270 | 0.6731 | 1.0000 | 0.6731 | 0.8046 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7134 | 0.0000 | 0.0000 | 0.0000 | 0.7134 | 1.0000 |
| 90 | 10 | 299,940 | 0.7413 | 0.2785 | 0.9975 | 0.4354 | 0.7129 | 0.9996 |
| 80 | 20 | 291,350 | 0.7693 | 0.4643 | 0.9973 | 0.6336 | 0.7123 | 0.9990 |
| 70 | 30 | 194,230 | 0.7989 | 0.5991 | 0.9973 | 0.7485 | 0.7140 | 0.9984 |
| 60 | 40 | 145,675 | 0.8265 | 0.6982 | 0.9973 | 0.8213 | 0.7126 | 0.9975 |
| 50 | 50 | 116,540 | 0.8532 | 0.7741 | 0.9973 | 0.8717 | 0.7090 | 0.9962 |
| 40 | 60 | 97,115 | 0.8839 | 0.8394 | 0.9973 | 0.9116 | 0.7138 | 0.9943 |
| 30 | 70 | 83,240 | 0.9120 | 0.8902 | 0.9973 | 0.9407 | 0.7129 | 0.9911 |
| 20 | 80 | 72,835 | 0.9395 | 0.9319 | 0.9973 | 0.9635 | 0.7083 | 0.9848 |
| 10 | 90 | 64,740 | 0.9685 | 0.9686 | 0.9973 | 0.9827 | 0.7093 | 0.9665 |
| 0 | 100 | 58,270 | 0.9973 | 1.0000 | 0.9973 | 0.9986 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7112 | 0.0000 | 0.0000 | 0.0000 | 0.7112 | 1.0000 |
| 90 | 10 | 299,940 | 0.7393 | 0.2769 | 0.9975 | 0.4335 | 0.7106 | 0.9996 |
| 80 | 20 | 291,350 | 0.7675 | 0.4623 | 0.9973 | 0.6318 | 0.7101 | 0.9990 |
| 70 | 30 | 194,230 | 0.7974 | 0.5972 | 0.9973 | 0.7471 | 0.7118 | 0.9983 |
| 60 | 40 | 145,675 | 0.8251 | 0.6965 | 0.9973 | 0.8202 | 0.7103 | 0.9974 |
| 50 | 50 | 116,540 | 0.8520 | 0.7728 | 0.9973 | 0.8708 | 0.7068 | 0.9961 |
| 40 | 60 | 97,115 | 0.8830 | 0.8384 | 0.9973 | 0.9110 | 0.7117 | 0.9942 |
| 30 | 70 | 83,240 | 0.9113 | 0.8895 | 0.9973 | 0.9403 | 0.7108 | 0.9911 |
| 20 | 80 | 72,835 | 0.9390 | 0.9314 | 0.9973 | 0.9632 | 0.7060 | 0.9847 |
| 10 | 90 | 64,740 | 0.9682 | 0.9684 | 0.9973 | 0.9826 | 0.7070 | 0.9662 |
| 0 | 100 | 58,270 | 0.9973 | 1.0000 | 0.9973 | 0.9986 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7112 | 0.0000 | 0.0000 | 0.0000 | 0.7112 | 1.0000 |
| 90 | 10 | 299,940 | 0.7393 | 0.2769 | 0.9975 | 0.4335 | 0.7106 | 0.9996 |
| 80 | 20 | 291,350 | 0.7675 | 0.4623 | 0.9973 | 0.6318 | 0.7101 | 0.9990 |
| 70 | 30 | 194,230 | 0.7974 | 0.5972 | 0.9973 | 0.7471 | 0.7118 | 0.9983 |
| 60 | 40 | 145,675 | 0.8251 | 0.6965 | 0.9973 | 0.8202 | 0.7103 | 0.9974 |
| 50 | 50 | 116,540 | 0.8520 | 0.7728 | 0.9973 | 0.8708 | 0.7068 | 0.9961 |
| 40 | 60 | 97,115 | 0.8830 | 0.8384 | 0.9973 | 0.9110 | 0.7117 | 0.9942 |
| 30 | 70 | 83,240 | 0.9113 | 0.8895 | 0.9973 | 0.9403 | 0.7108 | 0.9911 |
| 20 | 80 | 72,835 | 0.9390 | 0.9314 | 0.9973 | 0.9632 | 0.7060 | 0.9847 |
| 10 | 90 | 64,740 | 0.9682 | 0.9684 | 0.9973 | 0.9826 | 0.7070 | 0.9662 |
| 0 | 100 | 58,270 | 0.9973 | 1.0000 | 0.9973 | 0.9986 | 0.0000 | 0.0000 |


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
0.15       0.2234   0.1473   0.1738   0.8260   0.6705   0.0827   <--
0.20       0.2234   0.1473   0.1738   0.8260   0.6705   0.0827  
0.25       0.2234   0.1473   0.1738   0.8260   0.6705   0.0827  
0.30       0.2234   0.1473   0.1738   0.8260   0.6705   0.0827  
0.35       0.2234   0.1473   0.1738   0.8260   0.6705   0.0827  
0.40       0.2234   0.1473   0.1738   0.8260   0.6705   0.0827  
0.45       0.2234   0.1473   0.1738   0.8260   0.6705   0.0827  
0.50       0.2234   0.1473   0.1738   0.8260   0.6705   0.0827  
0.55       0.2234   0.1473   0.1738   0.8260   0.6705   0.0827  
0.60       0.2234   0.1473   0.1738   0.8260   0.6705   0.0827  
0.65       0.2234   0.1473   0.1738   0.8260   0.6705   0.0827  
0.70       0.2234   0.1473   0.1738   0.8260   0.6705   0.0827  
0.75       0.2234   0.1473   0.1738   0.8260   0.6705   0.0827  
0.80       0.2234   0.1473   0.1738   0.8260   0.6705   0.0827  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2234, F1=0.1473, Normal Recall=0.1738, Normal Precision=0.8260, Attack Recall=0.6705, Attack Precision=0.0827

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
0.15       0.2730   0.2703   0.1730   0.6791   0.6731   0.1691   <--
0.20       0.2730   0.2703   0.1730   0.6791   0.6731   0.1691  
0.25       0.2730   0.2703   0.1730   0.6791   0.6731   0.1691  
0.30       0.2730   0.2703   0.1730   0.6791   0.6731   0.1691  
0.35       0.2730   0.2703   0.1730   0.6791   0.6731   0.1691  
0.40       0.2730   0.2703   0.1730   0.6791   0.6731   0.1691  
0.45       0.2730   0.2703   0.1730   0.6791   0.6731   0.1691  
0.50       0.2730   0.2703   0.1730   0.6791   0.6731   0.1691  
0.55       0.2730   0.2703   0.1730   0.6791   0.6731   0.1691  
0.60       0.2730   0.2703   0.1730   0.6791   0.6731   0.1691  
0.65       0.2730   0.2703   0.1730   0.6791   0.6731   0.1691  
0.70       0.2730   0.2703   0.1730   0.6791   0.6731   0.1691  
0.75       0.2730   0.2703   0.1730   0.6791   0.6731   0.1691  
0.80       0.2730   0.2703   0.1730   0.6791   0.6731   0.1691  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2730, F1=0.2703, Normal Recall=0.1730, Normal Precision=0.6791, Attack Recall=0.6731, Attack Precision=0.1691

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
0.15       0.3231   0.3737   0.1731   0.5527   0.6731   0.2586   <--
0.20       0.3231   0.3737   0.1731   0.5527   0.6731   0.2586  
0.25       0.3231   0.3737   0.1731   0.5527   0.6731   0.2586  
0.30       0.3231   0.3737   0.1731   0.5527   0.6731   0.2586  
0.35       0.3231   0.3737   0.1731   0.5527   0.6731   0.2586  
0.40       0.3231   0.3737   0.1731   0.5527   0.6731   0.2586  
0.45       0.3231   0.3737   0.1731   0.5527   0.6731   0.2586  
0.50       0.3231   0.3737   0.1731   0.5527   0.6731   0.2586  
0.55       0.3231   0.3737   0.1731   0.5527   0.6731   0.2586  
0.60       0.3231   0.3737   0.1731   0.5527   0.6731   0.2586  
0.65       0.3231   0.3737   0.1731   0.5527   0.6731   0.2586  
0.70       0.3231   0.3737   0.1731   0.5527   0.6731   0.2586  
0.75       0.3231   0.3737   0.1731   0.5527   0.6731   0.2586  
0.80       0.3231   0.3737   0.1731   0.5527   0.6731   0.2586  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3231, F1=0.3737, Normal Recall=0.1731, Normal Precision=0.5527, Attack Recall=0.6731, Attack Precision=0.2586

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
0.15       0.3730   0.4620   0.1729   0.4425   0.6731   0.3517   <--
0.20       0.3730   0.4620   0.1729   0.4425   0.6731   0.3517  
0.25       0.3730   0.4620   0.1729   0.4425   0.6731   0.3517  
0.30       0.3730   0.4620   0.1729   0.4425   0.6731   0.3517  
0.35       0.3730   0.4620   0.1729   0.4425   0.6731   0.3517  
0.40       0.3730   0.4620   0.1729   0.4425   0.6731   0.3517  
0.45       0.3730   0.4620   0.1729   0.4425   0.6731   0.3517  
0.50       0.3730   0.4620   0.1729   0.4425   0.6731   0.3517  
0.55       0.3730   0.4620   0.1729   0.4425   0.6731   0.3517  
0.60       0.3730   0.4620   0.1729   0.4425   0.6731   0.3517  
0.65       0.3730   0.4620   0.1729   0.4425   0.6731   0.3517  
0.70       0.3730   0.4620   0.1729   0.4425   0.6731   0.3517  
0.75       0.3730   0.4620   0.1729   0.4425   0.6731   0.3517  
0.80       0.3730   0.4620   0.1729   0.4425   0.6731   0.3517  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3730, F1=0.4620, Normal Recall=0.1729, Normal Precision=0.4425, Attack Recall=0.6731, Attack Precision=0.3517

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
0.15       0.4238   0.5388   0.1744   0.3479   0.6731   0.4491   <--
0.20       0.4238   0.5388   0.1744   0.3479   0.6731   0.4491  
0.25       0.4238   0.5388   0.1744   0.3479   0.6731   0.4491  
0.30       0.4238   0.5388   0.1744   0.3479   0.6731   0.4491  
0.35       0.4238   0.5388   0.1744   0.3479   0.6731   0.4491  
0.40       0.4238   0.5388   0.1744   0.3479   0.6731   0.4491  
0.45       0.4238   0.5388   0.1744   0.3479   0.6731   0.4491  
0.50       0.4238   0.5388   0.1744   0.3479   0.6731   0.4491  
0.55       0.4238   0.5388   0.1744   0.3479   0.6731   0.4491  
0.60       0.4238   0.5388   0.1744   0.3479   0.6731   0.4491  
0.65       0.4238   0.5388   0.1744   0.3479   0.6731   0.4491  
0.70       0.4238   0.5388   0.1744   0.3479   0.6731   0.4491  
0.75       0.4238   0.5388   0.1744   0.3479   0.6731   0.4491  
0.80       0.4238   0.5388   0.1744   0.3479   0.6731   0.4491  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4238, F1=0.5388, Normal Recall=0.1744, Normal Precision=0.3479, Attack Recall=0.6731, Attack Precision=0.4491

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
0.15       0.7413   0.4354   0.7129   0.9996   0.9974   0.2785   <--
0.20       0.7413   0.4354   0.7129   0.9996   0.9974   0.2785  
0.25       0.7413   0.4354   0.7129   0.9996   0.9974   0.2785  
0.30       0.7413   0.4354   0.7129   0.9996   0.9974   0.2785  
0.35       0.7413   0.4354   0.7129   0.9996   0.9974   0.2785  
0.40       0.7413   0.4354   0.7129   0.9996   0.9974   0.2785  
0.45       0.7413   0.4354   0.7129   0.9996   0.9974   0.2785  
0.50       0.7413   0.4354   0.7129   0.9996   0.9974   0.2785  
0.55       0.7413   0.4354   0.7129   0.9996   0.9974   0.2785  
0.60       0.7413   0.4354   0.7129   0.9996   0.9974   0.2785  
0.65       0.7413   0.4354   0.7129   0.9996   0.9974   0.2785  
0.70       0.7413   0.4354   0.7129   0.9996   0.9974   0.2785  
0.75       0.7413   0.4354   0.7129   0.9996   0.9974   0.2785  
0.80       0.7413   0.4354   0.7129   0.9996   0.9974   0.2785  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7413, F1=0.4354, Normal Recall=0.7129, Normal Precision=0.9996, Attack Recall=0.9974, Attack Precision=0.2785

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
0.15       0.7702   0.6345   0.7134   0.9990   0.9973   0.4652   <--
0.20       0.7702   0.6345   0.7134   0.9990   0.9973   0.4652  
0.25       0.7702   0.6345   0.7134   0.9990   0.9973   0.4652  
0.30       0.7702   0.6345   0.7134   0.9990   0.9973   0.4652  
0.35       0.7702   0.6345   0.7134   0.9990   0.9973   0.4652  
0.40       0.7702   0.6345   0.7134   0.9990   0.9973   0.4652  
0.45       0.7702   0.6345   0.7134   0.9990   0.9973   0.4652  
0.50       0.7702   0.6345   0.7134   0.9990   0.9973   0.4652  
0.55       0.7702   0.6345   0.7134   0.9990   0.9973   0.4652  
0.60       0.7702   0.6345   0.7134   0.9990   0.9973   0.4652  
0.65       0.7702   0.6345   0.7134   0.9990   0.9973   0.4652  
0.70       0.7702   0.6345   0.7134   0.9990   0.9973   0.4652  
0.75       0.7702   0.6345   0.7134   0.9990   0.9973   0.4652  
0.80       0.7702   0.6345   0.7134   0.9990   0.9973   0.4652  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7702, F1=0.6345, Normal Recall=0.7134, Normal Precision=0.9990, Attack Recall=0.9973, Attack Precision=0.4652

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
0.15       0.7989   0.7484   0.7139   0.9984   0.9973   0.5990   <--
0.20       0.7989   0.7484   0.7139   0.9984   0.9973   0.5990  
0.25       0.7989   0.7484   0.7139   0.9984   0.9973   0.5990  
0.30       0.7989   0.7484   0.7139   0.9984   0.9973   0.5990  
0.35       0.7989   0.7484   0.7139   0.9984   0.9973   0.5990  
0.40       0.7989   0.7484   0.7139   0.9984   0.9973   0.5990  
0.45       0.7989   0.7484   0.7139   0.9984   0.9973   0.5990  
0.50       0.7989   0.7484   0.7139   0.9984   0.9973   0.5990  
0.55       0.7989   0.7484   0.7139   0.9984   0.9973   0.5990  
0.60       0.7989   0.7484   0.7139   0.9984   0.9973   0.5990  
0.65       0.7989   0.7484   0.7139   0.9984   0.9973   0.5990  
0.70       0.7989   0.7484   0.7139   0.9984   0.9973   0.5990  
0.75       0.7989   0.7484   0.7139   0.9984   0.9973   0.5990  
0.80       0.7989   0.7484   0.7139   0.9984   0.9973   0.5990  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7989, F1=0.7484, Normal Recall=0.7139, Normal Precision=0.9984, Attack Recall=0.9973, Attack Precision=0.5990

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
0.15       0.8269   0.8217   0.7133   0.9975   0.9973   0.6987   <--
0.20       0.8269   0.8217   0.7133   0.9975   0.9973   0.6987  
0.25       0.8269   0.8217   0.7133   0.9975   0.9973   0.6987  
0.30       0.8269   0.8217   0.7133   0.9975   0.9973   0.6987  
0.35       0.8269   0.8217   0.7133   0.9975   0.9973   0.6987  
0.40       0.8269   0.8217   0.7133   0.9975   0.9973   0.6987  
0.45       0.8269   0.8217   0.7133   0.9975   0.9973   0.6987  
0.50       0.8269   0.8217   0.7133   0.9975   0.9973   0.6987  
0.55       0.8269   0.8217   0.7133   0.9975   0.9973   0.6987  
0.60       0.8269   0.8217   0.7133   0.9975   0.9973   0.6987  
0.65       0.8269   0.8217   0.7133   0.9975   0.9973   0.6987  
0.70       0.8269   0.8217   0.7133   0.9975   0.9973   0.6987  
0.75       0.8269   0.8217   0.7133   0.9975   0.9973   0.6987  
0.80       0.8269   0.8217   0.7133   0.9975   0.9973   0.6987  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8269, F1=0.8217, Normal Recall=0.7133, Normal Precision=0.9975, Attack Recall=0.9973, Attack Precision=0.6987

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
0.15       0.8544   0.8726   0.7115   0.9962   0.9973   0.7756   <--
0.20       0.8544   0.8726   0.7115   0.9962   0.9973   0.7756  
0.25       0.8544   0.8726   0.7115   0.9962   0.9973   0.7756  
0.30       0.8544   0.8726   0.7115   0.9962   0.9973   0.7756  
0.35       0.8544   0.8726   0.7115   0.9962   0.9973   0.7756  
0.40       0.8544   0.8726   0.7115   0.9962   0.9973   0.7756  
0.45       0.8544   0.8726   0.7115   0.9962   0.9973   0.7756  
0.50       0.8544   0.8726   0.7115   0.9962   0.9973   0.7756  
0.55       0.8544   0.8726   0.7115   0.9962   0.9973   0.7756  
0.60       0.8544   0.8726   0.7115   0.9962   0.9973   0.7756  
0.65       0.8544   0.8726   0.7115   0.9962   0.9973   0.7756  
0.70       0.8544   0.8726   0.7115   0.9962   0.9973   0.7756  
0.75       0.8544   0.8726   0.7115   0.9962   0.9973   0.7756  
0.80       0.8544   0.8726   0.7115   0.9962   0.9973   0.7756  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8544, F1=0.8726, Normal Recall=0.7115, Normal Precision=0.9962, Attack Recall=0.9973, Attack Precision=0.7756

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
0.15       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769   <--
0.20       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.25       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.30       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.35       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.40       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.45       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.50       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.55       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.60       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.65       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.70       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.75       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.80       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7393, F1=0.4335, Normal Recall=0.7106, Normal Precision=0.9996, Attack Recall=0.9974, Attack Precision=0.2769

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
0.15       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632   <--
0.20       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.25       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.30       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.35       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.40       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.45       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.50       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.55       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.60       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.65       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.70       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.75       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.80       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7683, F1=0.6326, Normal Recall=0.7111, Normal Precision=0.9990, Attack Recall=0.9973, Attack Precision=0.4632

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
0.15       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972   <--
0.20       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.25       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.30       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.35       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.40       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.45       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.50       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.55       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.60       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.65       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.70       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.75       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.80       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7974, F1=0.7470, Normal Recall=0.7117, Normal Precision=0.9983, Attack Recall=0.9973, Attack Precision=0.5972

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
0.15       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972   <--
0.20       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.25       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.30       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.35       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.40       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.45       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.50       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.55       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.60       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.65       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.70       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.75       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.80       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8256, F1=0.8207, Normal Recall=0.7112, Normal Precision=0.9974, Attack Recall=0.9973, Attack Precision=0.6972

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
0.15       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744   <--
0.20       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.25       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.30       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.35       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.40       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.45       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.50       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.55       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.60       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.65       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.70       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.75       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.80       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8533, F1=0.8718, Normal Recall=0.7094, Normal Precision=0.9961, Attack Recall=0.9973, Attack Precision=0.7744

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
0.15       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769   <--
0.20       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.25       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.30       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.35       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.40       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.45       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.50       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.55       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.60       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.65       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.70       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.75       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
0.80       0.7393   0.4335   0.7106   0.9996   0.9974   0.2769  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7393, F1=0.4335, Normal Recall=0.7106, Normal Precision=0.9996, Attack Recall=0.9974, Attack Precision=0.2769

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
0.15       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632   <--
0.20       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.25       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.30       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.35       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.40       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.45       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.50       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.55       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.60       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.65       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.70       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.75       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
0.80       0.7683   0.6326   0.7111   0.9990   0.9973   0.4632  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7683, F1=0.6326, Normal Recall=0.7111, Normal Precision=0.9990, Attack Recall=0.9973, Attack Precision=0.4632

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
0.15       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972   <--
0.20       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.25       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.30       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.35       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.40       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.45       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.50       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.55       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.60       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.65       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.70       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.75       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
0.80       0.7974   0.7470   0.7117   0.9983   0.9973   0.5972  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7974, F1=0.7470, Normal Recall=0.7117, Normal Precision=0.9983, Attack Recall=0.9973, Attack Precision=0.5972

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
0.15       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972   <--
0.20       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.25       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.30       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.35       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.40       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.45       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.50       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.55       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.60       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.65       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.70       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.75       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
0.80       0.8256   0.8207   0.7112   0.9974   0.9973   0.6972  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8256, F1=0.8207, Normal Recall=0.7112, Normal Precision=0.9974, Attack Recall=0.9973, Attack Precision=0.6972

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
0.15       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744   <--
0.20       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.25       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.30       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.35       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.40       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.45       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.50       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.55       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.60       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.65       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.70       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.75       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
0.80       0.8533   0.8718   0.7094   0.9961   0.9973   0.7744  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8533, F1=0.8718, Normal Recall=0.7094, Normal Precision=0.9961, Attack Recall=0.9973, Attack Precision=0.7744

```

