# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e2_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-22 19:46:46 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9051 | 0.8957 | 0.8859 | 0.8770 | 0.8667 | 0.8569 | 0.8474 | 0.8373 | 0.8273 | 0.8176 | 0.8084 |
| QAT+Prune only | 0.1841 | 0.2633 | 0.3450 | 0.4269 | 0.5100 | 0.5907 | 0.6720 | 0.7547 | 0.8363 | 0.9179 | 0.9997 |
| QAT+PTQ | 0.1836 | 0.2629 | 0.3445 | 0.4266 | 0.5097 | 0.5904 | 0.6715 | 0.7547 | 0.8360 | 0.9177 | 0.9997 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.1836 | 0.2629 | 0.3445 | 0.4266 | 0.5097 | 0.5904 | 0.6715 | 0.7547 | 0.8360 | 0.9177 | 0.9997 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6079 | 0.7391 | 0.7978 | 0.8291 | 0.8496 | 0.8641 | 0.8743 | 0.8822 | 0.8886 | 0.8941 |
| QAT+Prune only | 0.0000 | 0.2134 | 0.3791 | 0.5114 | 0.6201 | 0.7095 | 0.7853 | 0.8509 | 0.9072 | 0.9564 | 0.9999 |
| QAT+PTQ | 0.0000 | 0.2133 | 0.3789 | 0.5113 | 0.6200 | 0.7093 | 0.7850 | 0.8509 | 0.9070 | 0.9563 | 0.9999 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2133 | 0.3789 | 0.5113 | 0.6200 | 0.7093 | 0.7850 | 0.8509 | 0.9070 | 0.9563 | 0.9999 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9051 | 0.9054 | 0.9052 | 0.9064 | 0.9056 | 0.9053 | 0.9059 | 0.9046 | 0.9029 | 0.8996 | 0.0000 |
| QAT+Prune only | 0.1841 | 0.1815 | 0.1813 | 0.1814 | 0.1836 | 0.1816 | 0.1804 | 0.1831 | 0.1828 | 0.1820 | 0.0000 |
| QAT+PTQ | 0.1836 | 0.1810 | 0.1807 | 0.1810 | 0.1831 | 0.1810 | 0.1791 | 0.1829 | 0.1811 | 0.1796 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.1836 | 0.1810 | 0.1807 | 0.1810 | 0.1831 | 0.1810 | 0.1791 | 0.1829 | 0.1811 | 0.1796 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9051 | 0.0000 | 0.0000 | 0.0000 | 0.9051 | 1.0000 |
| 90 | 10 | 299,940 | 0.8957 | 0.4871 | 0.8083 | 0.6079 | 0.9054 | 0.9770 |
| 80 | 20 | 291,350 | 0.8859 | 0.6808 | 0.8084 | 0.7391 | 0.9052 | 0.9498 |
| 70 | 30 | 194,230 | 0.8770 | 0.7873 | 0.8084 | 0.7978 | 0.9064 | 0.9169 |
| 60 | 40 | 145,675 | 0.8667 | 0.8509 | 0.8084 | 0.8291 | 0.9056 | 0.8764 |
| 50 | 50 | 116,540 | 0.8569 | 0.8952 | 0.8084 | 0.8496 | 0.9053 | 0.8254 |
| 40 | 60 | 97,115 | 0.8474 | 0.9280 | 0.8085 | 0.8641 | 0.9059 | 0.7592 |
| 30 | 70 | 83,240 | 0.8373 | 0.9518 | 0.8084 | 0.8743 | 0.9046 | 0.6693 |
| 20 | 80 | 72,835 | 0.8273 | 0.9709 | 0.8085 | 0.8822 | 0.9029 | 0.5410 |
| 10 | 90 | 64,740 | 0.8176 | 0.9864 | 0.8084 | 0.8886 | 0.8996 | 0.3429 |
| 0 | 100 | 58,270 | 0.8084 | 1.0000 | 0.8084 | 0.8941 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1841 | 0.0000 | 0.0000 | 0.0000 | 0.1841 | 1.0000 |
| 90 | 10 | 299,940 | 0.2633 | 0.1195 | 0.9996 | 0.2134 | 0.1815 | 0.9997 |
| 80 | 20 | 291,350 | 0.3450 | 0.2339 | 0.9997 | 0.3791 | 0.1813 | 0.9996 |
| 70 | 30 | 194,230 | 0.4269 | 0.3436 | 0.9997 | 0.5114 | 0.1814 | 0.9994 |
| 60 | 40 | 145,675 | 0.5100 | 0.4494 | 0.9997 | 0.6201 | 0.1836 | 0.9990 |
| 50 | 50 | 116,540 | 0.5907 | 0.5499 | 0.9997 | 0.7095 | 0.1816 | 0.9985 |
| 40 | 60 | 97,115 | 0.6720 | 0.6466 | 0.9997 | 0.7853 | 0.1804 | 0.9977 |
| 30 | 70 | 83,240 | 0.7547 | 0.7406 | 0.9997 | 0.8509 | 0.1831 | 0.9965 |
| 20 | 80 | 72,835 | 0.8363 | 0.8303 | 0.9997 | 0.9072 | 0.1828 | 0.9940 |
| 10 | 90 | 64,740 | 0.9179 | 0.9167 | 0.9997 | 0.9564 | 0.1820 | 0.9866 |
| 0 | 100 | 58,270 | 0.9997 | 1.0000 | 0.9997 | 0.9999 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1836 | 0.0000 | 0.0000 | 0.0000 | 0.1836 | 1.0000 |
| 90 | 10 | 299,940 | 0.2629 | 0.1194 | 0.9996 | 0.2133 | 0.1810 | 0.9997 |
| 80 | 20 | 291,350 | 0.3445 | 0.2337 | 0.9997 | 0.3789 | 0.1807 | 0.9996 |
| 70 | 30 | 194,230 | 0.4266 | 0.3435 | 0.9997 | 0.5113 | 0.1810 | 0.9994 |
| 60 | 40 | 145,675 | 0.5097 | 0.4493 | 0.9997 | 0.6200 | 0.1831 | 0.9990 |
| 50 | 50 | 116,540 | 0.5904 | 0.5497 | 0.9997 | 0.7093 | 0.1810 | 0.9985 |
| 40 | 60 | 97,115 | 0.6715 | 0.6463 | 0.9997 | 0.7850 | 0.1791 | 0.9977 |
| 30 | 70 | 83,240 | 0.7547 | 0.7406 | 0.9997 | 0.8509 | 0.1829 | 0.9965 |
| 20 | 80 | 72,835 | 0.8360 | 0.8300 | 0.9997 | 0.9070 | 0.1811 | 0.9940 |
| 10 | 90 | 64,740 | 0.9177 | 0.9164 | 0.9997 | 0.9563 | 0.1796 | 0.9864 |
| 0 | 100 | 58,270 | 0.9997 | 1.0000 | 0.9997 | 0.9999 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.1836 | 0.0000 | 0.0000 | 0.0000 | 0.1836 | 1.0000 |
| 90 | 10 | 299,940 | 0.2629 | 0.1194 | 0.9996 | 0.2133 | 0.1810 | 0.9997 |
| 80 | 20 | 291,350 | 0.3445 | 0.2337 | 0.9997 | 0.3789 | 0.1807 | 0.9996 |
| 70 | 30 | 194,230 | 0.4266 | 0.3435 | 0.9997 | 0.5113 | 0.1810 | 0.9994 |
| 60 | 40 | 145,675 | 0.5097 | 0.4493 | 0.9997 | 0.6200 | 0.1831 | 0.9990 |
| 50 | 50 | 116,540 | 0.5904 | 0.5497 | 0.9997 | 0.7093 | 0.1810 | 0.9985 |
| 40 | 60 | 97,115 | 0.6715 | 0.6463 | 0.9997 | 0.7850 | 0.1791 | 0.9977 |
| 30 | 70 | 83,240 | 0.7547 | 0.7406 | 0.9997 | 0.8509 | 0.1829 | 0.9965 |
| 20 | 80 | 72,835 | 0.8360 | 0.8300 | 0.9997 | 0.9070 | 0.1811 | 0.9940 |
| 10 | 90 | 64,740 | 0.9177 | 0.9164 | 0.9997 | 0.9563 | 0.1796 | 0.9864 |
| 0 | 100 | 58,270 | 0.9997 | 1.0000 | 0.9997 | 0.9999 | 0.0000 | 0.0000 |


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
0.15       0.8961   0.6097   0.9054   0.9774   0.8118   0.4882   <--
0.20       0.8961   0.6097   0.9054   0.9774   0.8118   0.4882  
0.25       0.8961   0.6097   0.9054   0.9774   0.8118   0.4882  
0.30       0.8961   0.6097   0.9054   0.9774   0.8118   0.4882  
0.35       0.8961   0.6097   0.9054   0.9774   0.8118   0.4882  
0.40       0.8961   0.6097   0.9054   0.9774   0.8118   0.4882  
0.45       0.8961   0.6097   0.9054   0.9774   0.8118   0.4882  
0.50       0.8961   0.6097   0.9054   0.9774   0.8118   0.4882  
0.55       0.8961   0.6097   0.9054   0.9774   0.8118   0.4882  
0.60       0.8961   0.6097   0.9054   0.9774   0.8118   0.4882  
0.65       0.8961   0.6097   0.9054   0.9774   0.8118   0.4882  
0.70       0.8961   0.6097   0.9054   0.9774   0.8118   0.4882  
0.75       0.8961   0.6097   0.9054   0.9774   0.8118   0.4882  
0.80       0.8961   0.6097   0.9054   0.9774   0.8118   0.4882  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8961, F1=0.6097, Normal Recall=0.9054, Normal Precision=0.9774, Attack Recall=0.8118, Attack Precision=0.4882

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
0.15       0.8865   0.7403   0.9061   0.9498   0.8084   0.6827   <--
0.20       0.8865   0.7403   0.9061   0.9498   0.8084   0.6827  
0.25       0.8865   0.7403   0.9061   0.9498   0.8084   0.6827  
0.30       0.8865   0.7403   0.9061   0.9498   0.8084   0.6827  
0.35       0.8865   0.7403   0.9061   0.9498   0.8084   0.6827  
0.40       0.8865   0.7403   0.9061   0.9498   0.8084   0.6827  
0.45       0.8865   0.7403   0.9061   0.9498   0.8084   0.6827  
0.50       0.8865   0.7403   0.9061   0.9498   0.8084   0.6827  
0.55       0.8865   0.7403   0.9061   0.9498   0.8084   0.6827  
0.60       0.8865   0.7403   0.9061   0.9498   0.8084   0.6827  
0.65       0.8865   0.7403   0.9061   0.9498   0.8084   0.6827  
0.70       0.8865   0.7403   0.9061   0.9498   0.8084   0.6827  
0.75       0.8865   0.7403   0.9061   0.9498   0.8084   0.6827  
0.80       0.8865   0.7403   0.9061   0.9498   0.8084   0.6827  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8865, F1=0.7403, Normal Recall=0.9061, Normal Precision=0.9498, Attack Recall=0.8084, Attack Precision=0.6827

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
0.15       0.8765   0.7971   0.9057   0.9169   0.8085   0.7860   <--
0.20       0.8765   0.7971   0.9057   0.9169   0.8085   0.7860  
0.25       0.8765   0.7971   0.9057   0.9169   0.8085   0.7860  
0.30       0.8765   0.7971   0.9057   0.9169   0.8085   0.7860  
0.35       0.8765   0.7971   0.9057   0.9169   0.8085   0.7860  
0.40       0.8765   0.7971   0.9057   0.9169   0.8085   0.7860  
0.45       0.8765   0.7971   0.9057   0.9169   0.8085   0.7860  
0.50       0.8765   0.7971   0.9057   0.9169   0.8085   0.7860  
0.55       0.8765   0.7971   0.9057   0.9169   0.8085   0.7860  
0.60       0.8765   0.7971   0.9057   0.9169   0.8085   0.7860  
0.65       0.8765   0.7971   0.9057   0.9169   0.8085   0.7860  
0.70       0.8765   0.7971   0.9057   0.9169   0.8085   0.7860  
0.75       0.8765   0.7971   0.9057   0.9169   0.8085   0.7860  
0.80       0.8765   0.7971   0.9057   0.9169   0.8085   0.7860  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8765, F1=0.7971, Normal Recall=0.9057, Normal Precision=0.9169, Attack Recall=0.8085, Attack Precision=0.7860

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
0.15       0.8666   0.8290   0.9054   0.8764   0.8084   0.8507   <--
0.20       0.8666   0.8290   0.9054   0.8764   0.8084   0.8507  
0.25       0.8666   0.8290   0.9054   0.8764   0.8084   0.8507  
0.30       0.8666   0.8290   0.9054   0.8764   0.8084   0.8507  
0.35       0.8666   0.8290   0.9054   0.8764   0.8084   0.8507  
0.40       0.8666   0.8290   0.9054   0.8764   0.8084   0.8507  
0.45       0.8666   0.8290   0.9054   0.8764   0.8084   0.8507  
0.50       0.8666   0.8290   0.9054   0.8764   0.8084   0.8507  
0.55       0.8666   0.8290   0.9054   0.8764   0.8084   0.8507  
0.60       0.8666   0.8290   0.9054   0.8764   0.8084   0.8507  
0.65       0.8666   0.8290   0.9054   0.8764   0.8084   0.8507  
0.70       0.8666   0.8290   0.9054   0.8764   0.8084   0.8507  
0.75       0.8666   0.8290   0.9054   0.8764   0.8084   0.8507  
0.80       0.8666   0.8290   0.9054   0.8764   0.8084   0.8507  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8666, F1=0.8290, Normal Recall=0.9054, Normal Precision=0.8764, Attack Recall=0.8084, Attack Precision=0.8507

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
0.15       0.8564   0.8492   0.9044   0.8252   0.8084   0.8943   <--
0.20       0.8564   0.8492   0.9044   0.8252   0.8084   0.8943  
0.25       0.8564   0.8492   0.9044   0.8252   0.8084   0.8943  
0.30       0.8564   0.8492   0.9044   0.8252   0.8084   0.8943  
0.35       0.8564   0.8492   0.9044   0.8252   0.8084   0.8943  
0.40       0.8564   0.8492   0.9044   0.8252   0.8084   0.8943  
0.45       0.8564   0.8492   0.9044   0.8252   0.8084   0.8943  
0.50       0.8564   0.8492   0.9044   0.8252   0.8084   0.8943  
0.55       0.8564   0.8492   0.9044   0.8252   0.8084   0.8943  
0.60       0.8564   0.8492   0.9044   0.8252   0.8084   0.8943  
0.65       0.8564   0.8492   0.9044   0.8252   0.8084   0.8943  
0.70       0.8564   0.8492   0.9044   0.8252   0.8084   0.8943  
0.75       0.8564   0.8492   0.9044   0.8252   0.8084   0.8943  
0.80       0.8564   0.8492   0.9044   0.8252   0.8084   0.8943  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8564, F1=0.8492, Normal Recall=0.9044, Normal Precision=0.8252, Attack Recall=0.8084, Attack Precision=0.8943

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
0.15       0.2634   0.2135   0.1815   0.9999   0.9998   0.1195   <--
0.20       0.2634   0.2135   0.1815   0.9999   0.9998   0.1195  
0.25       0.2634   0.2135   0.1815   0.9999   0.9998   0.1195  
0.30       0.2634   0.2135   0.1815   0.9999   0.9998   0.1195  
0.35       0.2634   0.2135   0.1815   0.9999   0.9998   0.1195  
0.40       0.2634   0.2135   0.1815   0.9999   0.9998   0.1195  
0.45       0.2634   0.2135   0.1815   0.9999   0.9998   0.1195  
0.50       0.2634   0.2135   0.1815   0.9999   0.9998   0.1195  
0.55       0.2634   0.2135   0.1815   0.9999   0.9998   0.1195  
0.60       0.2634   0.2135   0.1815   0.9999   0.9998   0.1195  
0.65       0.2634   0.2135   0.1815   0.9999   0.9998   0.1195  
0.70       0.2634   0.2135   0.1815   0.9999   0.9998   0.1195  
0.75       0.2634   0.2135   0.1815   0.9999   0.9998   0.1195  
0.80       0.2634   0.2135   0.1815   0.9999   0.9998   0.1195  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2634, F1=0.2135, Normal Recall=0.1815, Normal Precision=0.9999, Attack Recall=0.9998, Attack Precision=0.1195

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
0.15       0.3452   0.3792   0.1816   0.9996   0.9997   0.2340   <--
0.20       0.3452   0.3792   0.1816   0.9996   0.9997   0.2340  
0.25       0.3452   0.3792   0.1816   0.9996   0.9997   0.2340  
0.30       0.3452   0.3792   0.1816   0.9996   0.9997   0.2340  
0.35       0.3452   0.3792   0.1816   0.9996   0.9997   0.2340  
0.40       0.3452   0.3792   0.1816   0.9996   0.9997   0.2340  
0.45       0.3452   0.3792   0.1816   0.9996   0.9997   0.2340  
0.50       0.3452   0.3792   0.1816   0.9996   0.9997   0.2340  
0.55       0.3452   0.3792   0.1816   0.9996   0.9997   0.2340  
0.60       0.3452   0.3792   0.1816   0.9996   0.9997   0.2340  
0.65       0.3452   0.3792   0.1816   0.9996   0.9997   0.2340  
0.70       0.3452   0.3792   0.1816   0.9996   0.9997   0.2340  
0.75       0.3452   0.3792   0.1816   0.9996   0.9997   0.2340  
0.80       0.3452   0.3792   0.1816   0.9996   0.9997   0.2340  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3452, F1=0.3792, Normal Recall=0.1816, Normal Precision=0.9996, Attack Recall=0.9997, Attack Precision=0.2340

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
0.15       0.4279   0.5118   0.1829   0.9994   0.9997   0.3440   <--
0.20       0.4279   0.5118   0.1829   0.9994   0.9997   0.3440  
0.25       0.4279   0.5118   0.1829   0.9994   0.9997   0.3440  
0.30       0.4279   0.5118   0.1829   0.9994   0.9997   0.3440  
0.35       0.4279   0.5118   0.1829   0.9994   0.9997   0.3440  
0.40       0.4279   0.5118   0.1829   0.9994   0.9997   0.3440  
0.45       0.4279   0.5118   0.1829   0.9994   0.9997   0.3440  
0.50       0.4279   0.5118   0.1829   0.9994   0.9997   0.3440  
0.55       0.4279   0.5118   0.1829   0.9994   0.9997   0.3440  
0.60       0.4279   0.5118   0.1829   0.9994   0.9997   0.3440  
0.65       0.4279   0.5118   0.1829   0.9994   0.9997   0.3440  
0.70       0.4279   0.5118   0.1829   0.9994   0.9997   0.3440  
0.75       0.4279   0.5118   0.1829   0.9994   0.9997   0.3440  
0.80       0.4279   0.5118   0.1829   0.9994   0.9997   0.3440  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4279, F1=0.5118, Normal Recall=0.1829, Normal Precision=0.9994, Attack Recall=0.9997, Attack Precision=0.3440

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
0.15       0.5100   0.6201   0.1836   0.9990   0.9997   0.4494   <--
0.20       0.5100   0.6201   0.1836   0.9990   0.9997   0.4494  
0.25       0.5100   0.6201   0.1836   0.9990   0.9997   0.4494  
0.30       0.5100   0.6201   0.1836   0.9990   0.9997   0.4494  
0.35       0.5100   0.6201   0.1836   0.9990   0.9997   0.4494  
0.40       0.5100   0.6201   0.1836   0.9990   0.9997   0.4494  
0.45       0.5100   0.6201   0.1836   0.9990   0.9997   0.4494  
0.50       0.5100   0.6201   0.1836   0.9990   0.9997   0.4494  
0.55       0.5100   0.6201   0.1836   0.9990   0.9997   0.4494  
0.60       0.5100   0.6201   0.1836   0.9990   0.9997   0.4494  
0.65       0.5100   0.6201   0.1836   0.9990   0.9997   0.4494  
0.70       0.5100   0.6201   0.1836   0.9990   0.9997   0.4494  
0.75       0.5100   0.6201   0.1836   0.9990   0.9997   0.4494  
0.80       0.5100   0.6201   0.1836   0.9990   0.9997   0.4494  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5100, F1=0.6201, Normal Recall=0.1836, Normal Precision=0.9990, Attack Recall=0.9997, Attack Precision=0.4494

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
0.15       0.5920   0.7102   0.1843   0.9985   0.9997   0.5507   <--
0.20       0.5920   0.7102   0.1843   0.9985   0.9997   0.5507  
0.25       0.5920   0.7102   0.1843   0.9985   0.9997   0.5507  
0.30       0.5920   0.7102   0.1843   0.9985   0.9997   0.5507  
0.35       0.5920   0.7102   0.1843   0.9985   0.9997   0.5507  
0.40       0.5920   0.7102   0.1843   0.9985   0.9997   0.5507  
0.45       0.5920   0.7102   0.1843   0.9985   0.9997   0.5507  
0.50       0.5920   0.7102   0.1843   0.9985   0.9997   0.5507  
0.55       0.5920   0.7102   0.1843   0.9985   0.9997   0.5507  
0.60       0.5920   0.7102   0.1843   0.9985   0.9997   0.5507  
0.65       0.5920   0.7102   0.1843   0.9985   0.9997   0.5507  
0.70       0.5920   0.7102   0.1843   0.9985   0.9997   0.5507  
0.75       0.5920   0.7102   0.1843   0.9985   0.9997   0.5507  
0.80       0.5920   0.7102   0.1843   0.9985   0.9997   0.5507  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5920, F1=0.7102, Normal Recall=0.1843, Normal Precision=0.9985, Attack Recall=0.9997, Attack Precision=0.5507

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
0.15       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194   <--
0.20       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.25       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.30       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.35       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.40       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.45       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.50       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.55       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.60       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.65       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.70       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.75       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.80       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2629, F1=0.2134, Normal Recall=0.1810, Normal Precision=0.9999, Attack Recall=0.9998, Attack Precision=0.1194

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
0.15       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339   <--
0.20       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.25       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.30       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.35       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.40       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.45       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.50       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.55       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.60       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.65       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.70       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.75       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.80       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3449, F1=0.3790, Normal Recall=0.1812, Normal Precision=0.9996, Attack Recall=0.9997, Attack Precision=0.2339

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
0.15       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438   <--
0.20       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.25       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.30       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.35       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.40       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.45       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.50       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.55       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.60       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.65       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.70       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.75       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.80       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4276, F1=0.5117, Normal Recall=0.1824, Normal Precision=0.9994, Attack Recall=0.9997, Attack Precision=0.3438

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
0.15       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493   <--
0.20       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.25       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.30       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.35       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.40       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.45       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.50       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.55       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.60       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.65       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.70       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.75       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.80       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5098, F1=0.6200, Normal Recall=0.1831, Normal Precision=0.9990, Attack Recall=0.9997, Attack Precision=0.4493

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
0.15       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505   <--
0.20       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.25       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.30       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.35       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.40       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.45       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.50       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.55       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.60       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.65       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.70       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.75       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.80       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5917, F1=0.7100, Normal Recall=0.1836, Normal Precision=0.9985, Attack Recall=0.9997, Attack Precision=0.5505

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
0.15       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194   <--
0.20       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.25       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.30       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.35       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.40       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.45       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.50       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.55       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.60       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.65       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.70       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.75       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
0.80       0.2629   0.2134   0.1810   0.9999   0.9998   0.1194  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2629, F1=0.2134, Normal Recall=0.1810, Normal Precision=0.9999, Attack Recall=0.9998, Attack Precision=0.1194

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
0.15       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339   <--
0.20       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.25       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.30       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.35       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.40       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.45       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.50       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.55       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.60       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.65       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.70       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.75       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
0.80       0.3449   0.3790   0.1812   0.9996   0.9997   0.2339  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3449, F1=0.3790, Normal Recall=0.1812, Normal Precision=0.9996, Attack Recall=0.9997, Attack Precision=0.2339

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
0.15       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438   <--
0.20       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.25       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.30       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.35       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.40       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.45       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.50       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.55       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.60       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.65       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.70       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.75       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
0.80       0.4276   0.5117   0.1824   0.9994   0.9997   0.3438  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4276, F1=0.5117, Normal Recall=0.1824, Normal Precision=0.9994, Attack Recall=0.9997, Attack Precision=0.3438

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
0.15       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493   <--
0.20       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.25       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.30       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.35       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.40       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.45       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.50       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.55       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.60       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.65       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.70       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.75       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
0.80       0.5098   0.6200   0.1831   0.9990   0.9997   0.4493  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5098, F1=0.6200, Normal Recall=0.1831, Normal Precision=0.9990, Attack Recall=0.9997, Attack Precision=0.4493

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
0.15       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505   <--
0.20       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.25       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.30       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.35       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.40       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.45       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.50       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.55       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.60       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.65       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.70       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.75       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
0.80       0.5917   0.7100   0.1836   0.9985   0.9997   0.5505  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5917, F1=0.7100, Normal Recall=0.1836, Normal Precision=0.9985, Attack Recall=0.9997, Attack Precision=0.5505

```

