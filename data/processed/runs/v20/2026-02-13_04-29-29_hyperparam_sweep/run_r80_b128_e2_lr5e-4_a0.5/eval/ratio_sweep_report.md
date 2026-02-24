# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e2_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-17 21:49:12 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7901 | 0.7905 | 0.7906 | 0.7908 | 0.7914 | 0.7925 | 0.7925 | 0.7939 | 0.7942 | 0.7935 | 0.7948 |
| QAT+Prune only | 0.9788 | 0.9125 | 0.8462 | 0.7799 | 0.7132 | 0.6465 | 0.5804 | 0.5133 | 0.4468 | 0.3806 | 0.3140 |
| QAT+PTQ | 0.9789 | 0.9120 | 0.8452 | 0.7784 | 0.7110 | 0.6438 | 0.5772 | 0.5094 | 0.4425 | 0.3756 | 0.3085 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9789 | 0.9120 | 0.8452 | 0.7784 | 0.7110 | 0.6438 | 0.5772 | 0.5094 | 0.4425 | 0.3756 | 0.3085 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.4312 | 0.6029 | 0.6950 | 0.7530 | 0.7930 | 0.8213 | 0.8437 | 0.8607 | 0.8738 | 0.8857 |
| QAT+Prune only | 0.0000 | 0.4159 | 0.4496 | 0.4612 | 0.4669 | 0.4704 | 0.4732 | 0.4746 | 0.4759 | 0.4771 | 0.4779 |
| QAT+PTQ | 0.0000 | 0.4102 | 0.4436 | 0.4551 | 0.4607 | 0.4641 | 0.4669 | 0.4682 | 0.4696 | 0.4708 | 0.4716 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4102 | 0.4436 | 0.4551 | 0.4607 | 0.4641 | 0.4669 | 0.4682 | 0.4696 | 0.4708 | 0.4716 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7901 | 0.7900 | 0.7895 | 0.7890 | 0.7891 | 0.7901 | 0.7891 | 0.7916 | 0.7919 | 0.7813 | 0.0000 |
| QAT+Prune only | 0.9788 | 0.9793 | 0.9793 | 0.9796 | 0.9793 | 0.9790 | 0.9801 | 0.9781 | 0.9781 | 0.9795 | 0.0000 |
| QAT+PTQ | 0.9789 | 0.9794 | 0.9794 | 0.9797 | 0.9794 | 0.9790 | 0.9802 | 0.9782 | 0.9783 | 0.9796 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9789 | 0.9794 | 0.9794 | 0.9797 | 0.9794 | 0.9790 | 0.9802 | 0.9782 | 0.9783 | 0.9796 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7901 | 0.0000 | 0.0000 | 0.0000 | 0.7901 | 1.0000 |
| 90 | 10 | 299,940 | 0.7905 | 0.2959 | 0.7944 | 0.4312 | 0.7900 | 0.9719 |
| 80 | 20 | 291,350 | 0.7906 | 0.4856 | 0.7948 | 0.6029 | 0.7895 | 0.9390 |
| 70 | 30 | 194,230 | 0.7908 | 0.6175 | 0.7948 | 0.6950 | 0.7890 | 0.8997 |
| 60 | 40 | 145,675 | 0.7914 | 0.7153 | 0.7948 | 0.7530 | 0.7891 | 0.8523 |
| 50 | 50 | 116,540 | 0.7925 | 0.7911 | 0.7948 | 0.7930 | 0.7901 | 0.7939 |
| 40 | 60 | 97,115 | 0.7925 | 0.8497 | 0.7948 | 0.8213 | 0.7891 | 0.7194 |
| 30 | 70 | 83,240 | 0.7939 | 0.8990 | 0.7948 | 0.8437 | 0.7916 | 0.6232 |
| 20 | 80 | 72,835 | 0.7942 | 0.9386 | 0.7948 | 0.8607 | 0.7919 | 0.4911 |
| 10 | 90 | 64,740 | 0.7935 | 0.9703 | 0.7948 | 0.8738 | 0.7813 | 0.2973 |
| 0 | 100 | 58,270 | 0.7948 | 1.0000 | 0.7948 | 0.8857 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9788 | 0.0000 | 0.0000 | 0.0000 | 0.9788 | 1.0000 |
| 90 | 10 | 299,940 | 0.9125 | 0.6259 | 0.3114 | 0.4159 | 0.9793 | 0.9275 |
| 80 | 20 | 291,350 | 0.8462 | 0.7911 | 0.3140 | 0.4496 | 0.9793 | 0.8510 |
| 70 | 30 | 194,230 | 0.7799 | 0.8685 | 0.3140 | 0.4612 | 0.9796 | 0.7692 |
| 60 | 40 | 145,675 | 0.7132 | 0.9100 | 0.3140 | 0.4669 | 0.9793 | 0.6817 |
| 50 | 50 | 116,540 | 0.6465 | 0.9373 | 0.3140 | 0.4704 | 0.9790 | 0.5880 |
| 40 | 60 | 97,115 | 0.5804 | 0.9594 | 0.3140 | 0.4732 | 0.9801 | 0.4878 |
| 30 | 70 | 83,240 | 0.5133 | 0.9710 | 0.3140 | 0.4746 | 0.9781 | 0.3793 |
| 20 | 80 | 72,835 | 0.4468 | 0.9829 | 0.3140 | 0.4759 | 0.9781 | 0.2628 |
| 10 | 90 | 64,740 | 0.3806 | 0.9928 | 0.3140 | 0.4771 | 0.9795 | 0.1369 |
| 0 | 100 | 58,270 | 0.3140 | 1.0000 | 0.3140 | 0.4779 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9789 | 0.0000 | 0.0000 | 0.0000 | 0.9789 | 1.0000 |
| 90 | 10 | 299,940 | 0.9120 | 0.6226 | 0.3059 | 0.4102 | 0.9794 | 0.9270 |
| 80 | 20 | 291,350 | 0.8452 | 0.7888 | 0.3085 | 0.4436 | 0.9794 | 0.8500 |
| 70 | 30 | 194,230 | 0.7784 | 0.8670 | 0.3085 | 0.4551 | 0.9797 | 0.7678 |
| 60 | 40 | 145,675 | 0.7110 | 0.9089 | 0.3085 | 0.4607 | 0.9794 | 0.6800 |
| 50 | 50 | 116,540 | 0.6438 | 0.9364 | 0.3085 | 0.4641 | 0.9790 | 0.5861 |
| 40 | 60 | 97,115 | 0.5772 | 0.9589 | 0.3085 | 0.4669 | 0.9802 | 0.4859 |
| 30 | 70 | 83,240 | 0.5094 | 0.9706 | 0.3085 | 0.4682 | 0.9782 | 0.3774 |
| 20 | 80 | 72,835 | 0.4425 | 0.9827 | 0.3085 | 0.4696 | 0.9783 | 0.2613 |
| 10 | 90 | 64,740 | 0.3756 | 0.9927 | 0.3085 | 0.4708 | 0.9796 | 0.1360 |
| 0 | 100 | 58,270 | 0.3085 | 1.0000 | 0.3085 | 0.4716 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9789 | 0.0000 | 0.0000 | 0.0000 | 0.9789 | 1.0000 |
| 90 | 10 | 299,940 | 0.9120 | 0.6226 | 0.3059 | 0.4102 | 0.9794 | 0.9270 |
| 80 | 20 | 291,350 | 0.8452 | 0.7888 | 0.3085 | 0.4436 | 0.9794 | 0.8500 |
| 70 | 30 | 194,230 | 0.7784 | 0.8670 | 0.3085 | 0.4551 | 0.9797 | 0.7678 |
| 60 | 40 | 145,675 | 0.7110 | 0.9089 | 0.3085 | 0.4607 | 0.9794 | 0.6800 |
| 50 | 50 | 116,540 | 0.6438 | 0.9364 | 0.3085 | 0.4641 | 0.9790 | 0.5861 |
| 40 | 60 | 97,115 | 0.5772 | 0.9589 | 0.3085 | 0.4669 | 0.9802 | 0.4859 |
| 30 | 70 | 83,240 | 0.5094 | 0.9706 | 0.3085 | 0.4682 | 0.9782 | 0.3774 |
| 20 | 80 | 72,835 | 0.4425 | 0.9827 | 0.3085 | 0.4696 | 0.9783 | 0.2613 |
| 10 | 90 | 64,740 | 0.3756 | 0.9927 | 0.3085 | 0.4708 | 0.9796 | 0.1360 |
| 0 | 100 | 58,270 | 0.3085 | 1.0000 | 0.3085 | 0.4716 | 0.0000 | 0.0000 |


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
0.15       0.7903   0.4306   0.7900   0.9717   0.7928   0.2955   <--
0.20       0.7903   0.4306   0.7900   0.9717   0.7928   0.2955  
0.25       0.7903   0.4306   0.7900   0.9717   0.7928   0.2955  
0.30       0.7903   0.4306   0.7900   0.9717   0.7928   0.2955  
0.35       0.7903   0.4306   0.7900   0.9717   0.7928   0.2955  
0.40       0.7903   0.4306   0.7900   0.9717   0.7928   0.2955  
0.45       0.7903   0.4306   0.7900   0.9717   0.7928   0.2955  
0.50       0.7903   0.4306   0.7900   0.9717   0.7928   0.2955  
0.55       0.7903   0.4306   0.7900   0.9717   0.7928   0.2955  
0.60       0.7903   0.4306   0.7900   0.9717   0.7928   0.2955  
0.65       0.7903   0.4306   0.7900   0.9717   0.7928   0.2955  
0.70       0.7903   0.4306   0.7900   0.9717   0.7928   0.2955  
0.75       0.7903   0.4306   0.7900   0.9717   0.7928   0.2955  
0.80       0.7903   0.4306   0.7900   0.9717   0.7928   0.2955  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7903, F1=0.4306, Normal Recall=0.7900, Normal Precision=0.9717, Attack Recall=0.7928, Attack Precision=0.2955

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
0.15       0.7908   0.6031   0.7898   0.9390   0.7948   0.4859   <--
0.20       0.7908   0.6031   0.7898   0.9390   0.7948   0.4859  
0.25       0.7908   0.6031   0.7898   0.9390   0.7948   0.4859  
0.30       0.7908   0.6031   0.7898   0.9390   0.7948   0.4859  
0.35       0.7908   0.6031   0.7898   0.9390   0.7948   0.4859  
0.40       0.7908   0.6031   0.7898   0.9390   0.7948   0.4859  
0.45       0.7908   0.6031   0.7898   0.9390   0.7948   0.4859  
0.50       0.7908   0.6031   0.7898   0.9390   0.7948   0.4859  
0.55       0.7908   0.6031   0.7898   0.9390   0.7948   0.4859  
0.60       0.7908   0.6031   0.7898   0.9390   0.7948   0.4859  
0.65       0.7908   0.6031   0.7898   0.9390   0.7948   0.4859  
0.70       0.7908   0.6031   0.7898   0.9390   0.7948   0.4859  
0.75       0.7908   0.6031   0.7898   0.9390   0.7948   0.4859  
0.80       0.7908   0.6031   0.7898   0.9390   0.7948   0.4859  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7908, F1=0.6031, Normal Recall=0.7898, Normal Precision=0.9390, Attack Recall=0.7948, Attack Precision=0.4859

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
0.15       0.7917   0.6960   0.7904   0.8999   0.7948   0.6191   <--
0.20       0.7917   0.6960   0.7904   0.8999   0.7948   0.6191  
0.25       0.7917   0.6960   0.7904   0.8999   0.7948   0.6191  
0.30       0.7917   0.6960   0.7904   0.8999   0.7948   0.6191  
0.35       0.7917   0.6960   0.7904   0.8999   0.7948   0.6191  
0.40       0.7917   0.6960   0.7904   0.8999   0.7948   0.6191  
0.45       0.7917   0.6960   0.7904   0.8999   0.7948   0.6191  
0.50       0.7917   0.6960   0.7904   0.8999   0.7948   0.6191  
0.55       0.7917   0.6960   0.7904   0.8999   0.7948   0.6191  
0.60       0.7917   0.6960   0.7904   0.8999   0.7948   0.6191  
0.65       0.7917   0.6960   0.7904   0.8999   0.7948   0.6191  
0.70       0.7917   0.6960   0.7904   0.8999   0.7948   0.6191  
0.75       0.7917   0.6960   0.7904   0.8999   0.7948   0.6191  
0.80       0.7917   0.6960   0.7904   0.8999   0.7948   0.6191  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7917, F1=0.6960, Normal Recall=0.7904, Normal Precision=0.8999, Attack Recall=0.7948, Attack Precision=0.6191

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
0.15       0.7923   0.7538   0.7906   0.8525   0.7948   0.7167   <--
0.20       0.7923   0.7538   0.7906   0.8525   0.7948   0.7167  
0.25       0.7923   0.7538   0.7906   0.8525   0.7948   0.7167  
0.30       0.7923   0.7538   0.7906   0.8525   0.7948   0.7167  
0.35       0.7923   0.7538   0.7906   0.8525   0.7948   0.7167  
0.40       0.7923   0.7538   0.7906   0.8525   0.7948   0.7167  
0.45       0.7923   0.7538   0.7906   0.8525   0.7948   0.7167  
0.50       0.7923   0.7538   0.7906   0.8525   0.7948   0.7167  
0.55       0.7923   0.7538   0.7906   0.8525   0.7948   0.7167  
0.60       0.7923   0.7538   0.7906   0.8525   0.7948   0.7167  
0.65       0.7923   0.7538   0.7906   0.8525   0.7948   0.7167  
0.70       0.7923   0.7538   0.7906   0.8525   0.7948   0.7167  
0.75       0.7923   0.7538   0.7906   0.8525   0.7948   0.7167  
0.80       0.7923   0.7538   0.7906   0.8525   0.7948   0.7167  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7923, F1=0.7538, Normal Recall=0.7906, Normal Precision=0.8525, Attack Recall=0.7948, Attack Precision=0.7167

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
0.15       0.7933   0.7936   0.7918   0.7942   0.7948   0.7924   <--
0.20       0.7933   0.7936   0.7918   0.7942   0.7948   0.7924  
0.25       0.7933   0.7936   0.7918   0.7942   0.7948   0.7924  
0.30       0.7933   0.7936   0.7918   0.7942   0.7948   0.7924  
0.35       0.7933   0.7936   0.7918   0.7942   0.7948   0.7924  
0.40       0.7933   0.7936   0.7918   0.7942   0.7948   0.7924  
0.45       0.7933   0.7936   0.7918   0.7942   0.7948   0.7924  
0.50       0.7933   0.7936   0.7918   0.7942   0.7948   0.7924  
0.55       0.7933   0.7936   0.7918   0.7942   0.7948   0.7924  
0.60       0.7933   0.7936   0.7918   0.7942   0.7948   0.7924  
0.65       0.7933   0.7936   0.7918   0.7942   0.7948   0.7924  
0.70       0.7933   0.7936   0.7918   0.7942   0.7948   0.7924  
0.75       0.7933   0.7936   0.7918   0.7942   0.7948   0.7924  
0.80       0.7933   0.7936   0.7918   0.7942   0.7948   0.7924  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7933, F1=0.7936, Normal Recall=0.7918, Normal Precision=0.7942, Attack Recall=0.7948, Attack Precision=0.7924

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
0.15       0.9127   0.4173   0.9793   0.9277   0.3127   0.6269   <--
0.20       0.9127   0.4173   0.9793   0.9277   0.3127   0.6269  
0.25       0.9127   0.4173   0.9793   0.9277   0.3127   0.6269  
0.30       0.9127   0.4173   0.9793   0.9277   0.3127   0.6269  
0.35       0.9127   0.4173   0.9793   0.9277   0.3127   0.6269  
0.40       0.9127   0.4173   0.9793   0.9277   0.3127   0.6269  
0.45       0.9127   0.4173   0.9793   0.9277   0.3127   0.6269  
0.50       0.9127   0.4173   0.9793   0.9277   0.3127   0.6269  
0.55       0.9127   0.4173   0.9793   0.9277   0.3127   0.6269  
0.60       0.9127   0.4173   0.9793   0.9277   0.3127   0.6269  
0.65       0.9127   0.4173   0.9793   0.9277   0.3127   0.6269  
0.70       0.9127   0.4173   0.9793   0.9277   0.3127   0.6269  
0.75       0.9127   0.4173   0.9793   0.9277   0.3127   0.6269  
0.80       0.9127   0.4173   0.9793   0.9277   0.3127   0.6269  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9127, F1=0.4173, Normal Recall=0.9793, Normal Precision=0.9277, Attack Recall=0.3127, Attack Precision=0.6269

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
0.15       0.8462   0.4496   0.9793   0.8510   0.3140   0.7911   <--
0.20       0.8462   0.4496   0.9793   0.8510   0.3140   0.7911  
0.25       0.8462   0.4496   0.9793   0.8510   0.3140   0.7911  
0.30       0.8462   0.4496   0.9793   0.8510   0.3140   0.7911  
0.35       0.8462   0.4496   0.9793   0.8510   0.3140   0.7911  
0.40       0.8462   0.4496   0.9793   0.8510   0.3140   0.7911  
0.45       0.8462   0.4496   0.9793   0.8510   0.3140   0.7911  
0.50       0.8462   0.4496   0.9793   0.8510   0.3140   0.7911  
0.55       0.8462   0.4496   0.9793   0.8510   0.3140   0.7911  
0.60       0.8462   0.4496   0.9793   0.8510   0.3140   0.7911  
0.65       0.8462   0.4496   0.9793   0.8510   0.3140   0.7911  
0.70       0.8462   0.4496   0.9793   0.8510   0.3140   0.7911  
0.75       0.8462   0.4496   0.9793   0.8510   0.3140   0.7911  
0.80       0.8462   0.4496   0.9793   0.8510   0.3140   0.7911  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8462, F1=0.4496, Normal Recall=0.9793, Normal Precision=0.8510, Attack Recall=0.3140, Attack Precision=0.7911

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
0.15       0.7795   0.4608   0.9791   0.7691   0.3140   0.8654   <--
0.20       0.7795   0.4608   0.9791   0.7691   0.3140   0.8654  
0.25       0.7795   0.4608   0.9791   0.7691   0.3140   0.8654  
0.30       0.7795   0.4608   0.9791   0.7691   0.3140   0.8654  
0.35       0.7795   0.4608   0.9791   0.7691   0.3140   0.8654  
0.40       0.7795   0.4608   0.9791   0.7691   0.3140   0.8654  
0.45       0.7795   0.4608   0.9791   0.7691   0.3140   0.8654  
0.50       0.7795   0.4608   0.9791   0.7691   0.3140   0.8654  
0.55       0.7795   0.4608   0.9791   0.7691   0.3140   0.8654  
0.60       0.7795   0.4608   0.9791   0.7691   0.3140   0.8654  
0.65       0.7795   0.4608   0.9791   0.7691   0.3140   0.8654  
0.70       0.7795   0.4608   0.9791   0.7691   0.3140   0.8654  
0.75       0.7795   0.4608   0.9791   0.7691   0.3140   0.8654  
0.80       0.7795   0.4608   0.9791   0.7691   0.3140   0.8654  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7795, F1=0.4608, Normal Recall=0.9791, Normal Precision=0.7691, Attack Recall=0.3140, Attack Precision=0.8654

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
0.15       0.7131   0.4668   0.9791   0.6816   0.3140   0.9093   <--
0.20       0.7131   0.4668   0.9791   0.6816   0.3140   0.9093  
0.25       0.7131   0.4668   0.9791   0.6816   0.3140   0.9093  
0.30       0.7131   0.4668   0.9791   0.6816   0.3140   0.9093  
0.35       0.7131   0.4668   0.9791   0.6816   0.3140   0.9093  
0.40       0.7131   0.4668   0.9791   0.6816   0.3140   0.9093  
0.45       0.7131   0.4668   0.9791   0.6816   0.3140   0.9093  
0.50       0.7131   0.4668   0.9791   0.6816   0.3140   0.9093  
0.55       0.7131   0.4668   0.9791   0.6816   0.3140   0.9093  
0.60       0.7131   0.4668   0.9791   0.6816   0.3140   0.9093  
0.65       0.7131   0.4668   0.9791   0.6816   0.3140   0.9093  
0.70       0.7131   0.4668   0.9791   0.6816   0.3140   0.9093  
0.75       0.7131   0.4668   0.9791   0.6816   0.3140   0.9093  
0.80       0.7131   0.4668   0.9791   0.6816   0.3140   0.9093  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7131, F1=0.4668, Normal Recall=0.9791, Normal Precision=0.6816, Attack Recall=0.3140, Attack Precision=0.9093

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
0.15       0.6467   0.4706   0.9794   0.5881   0.3140   0.9385   <--
0.20       0.6467   0.4706   0.9794   0.5881   0.3140   0.9385  
0.25       0.6467   0.4706   0.9794   0.5881   0.3140   0.9385  
0.30       0.6467   0.4706   0.9794   0.5881   0.3140   0.9385  
0.35       0.6467   0.4706   0.9794   0.5881   0.3140   0.9385  
0.40       0.6467   0.4706   0.9794   0.5881   0.3140   0.9385  
0.45       0.6467   0.4706   0.9794   0.5881   0.3140   0.9385  
0.50       0.6467   0.4706   0.9794   0.5881   0.3140   0.9385  
0.55       0.6467   0.4706   0.9794   0.5881   0.3140   0.9385  
0.60       0.6467   0.4706   0.9794   0.5881   0.3140   0.9385  
0.65       0.6467   0.4706   0.9794   0.5881   0.3140   0.9385  
0.70       0.6467   0.4706   0.9794   0.5881   0.3140   0.9385  
0.75       0.6467   0.4706   0.9794   0.5881   0.3140   0.9385  
0.80       0.6467   0.4706   0.9794   0.5881   0.3140   0.9385  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6467, F1=0.4706, Normal Recall=0.9794, Normal Precision=0.5881, Attack Recall=0.3140, Attack Precision=0.9385

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
0.15       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242   <--
0.20       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.25       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.30       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.35       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.40       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.45       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.50       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.55       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.60       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.65       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.70       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.75       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.80       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9122, F1=0.4124, Normal Recall=0.9794, Normal Precision=0.9272, Attack Recall=0.3079, Attack Precision=0.6242

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
0.15       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889   <--
0.20       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.25       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.30       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.35       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.40       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.45       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.50       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.55       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.60       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.65       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.70       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.75       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.80       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8452, F1=0.4436, Normal Recall=0.9794, Normal Precision=0.8500, Attack Recall=0.3085, Attack Precision=0.7889

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
0.15       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637   <--
0.20       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.25       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.30       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.35       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.40       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.45       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.50       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.55       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.60       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.65       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.70       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.75       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.80       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7780, F1=0.4547, Normal Recall=0.9791, Normal Precision=0.7677, Attack Recall=0.3085, Attack Precision=0.8637

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
0.15       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083   <--
0.20       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.25       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.30       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.35       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.40       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.45       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.50       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.55       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.60       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.65       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.70       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.75       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.80       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7110, F1=0.4606, Normal Recall=0.9792, Normal Precision=0.6799, Attack Recall=0.3085, Attack Precision=0.9083

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
0.15       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377   <--
0.20       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.25       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.30       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.35       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.40       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.45       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.50       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.55       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.60       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.65       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.70       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.75       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.80       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6440, F1=0.4643, Normal Recall=0.9795, Normal Precision=0.5862, Attack Recall=0.3085, Attack Precision=0.9377

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
0.15       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242   <--
0.20       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.25       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.30       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.35       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.40       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.45       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.50       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.55       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.60       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.65       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.70       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.75       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
0.80       0.9122   0.4124   0.9794   0.9272   0.3079   0.6242  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9122, F1=0.4124, Normal Recall=0.9794, Normal Precision=0.9272, Attack Recall=0.3079, Attack Precision=0.6242

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
0.15       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889   <--
0.20       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.25       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.30       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.35       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.40       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.45       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.50       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.55       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.60       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.65       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.70       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.75       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
0.80       0.8452   0.4436   0.9794   0.8500   0.3085   0.7889  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8452, F1=0.4436, Normal Recall=0.9794, Normal Precision=0.8500, Attack Recall=0.3085, Attack Precision=0.7889

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
0.15       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637   <--
0.20       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.25       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.30       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.35       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.40       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.45       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.50       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.55       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.60       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.65       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.70       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.75       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
0.80       0.7780   0.4547   0.9791   0.7677   0.3085   0.8637  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7780, F1=0.4547, Normal Recall=0.9791, Normal Precision=0.7677, Attack Recall=0.3085, Attack Precision=0.8637

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
0.15       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083   <--
0.20       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.25       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.30       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.35       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.40       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.45       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.50       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.55       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.60       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.65       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.70       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.75       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
0.80       0.7110   0.4606   0.9792   0.6799   0.3085   0.9083  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7110, F1=0.4606, Normal Recall=0.9792, Normal Precision=0.6799, Attack Recall=0.3085, Attack Precision=0.9083

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
0.15       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377   <--
0.20       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.25       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.30       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.35       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.40       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.45       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.50       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.55       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.60       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.65       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.70       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.75       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
0.80       0.6440   0.4643   0.9795   0.5862   0.3085   0.9377  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6440, F1=0.4643, Normal Recall=0.9795, Normal Precision=0.5862, Attack Recall=0.3085, Attack Precision=0.9377

```

