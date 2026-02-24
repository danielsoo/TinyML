# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e2_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-14 04:39:05 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 60 |
| **Local epochs** | 2 |
| **Batch size** | 64 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1931 | 0.2718 | 0.3510 | 0.4297 | 0.5096 | 0.5877 | 0.6675 | 0.7452 | 0.8249 | 0.9044 | 0.9829 |
| QAT+Prune only | 0.7783 | 0.8001 | 0.8211 | 0.8437 | 0.8645 | 0.8847 | 0.9073 | 0.9285 | 0.9504 | 0.9708 | 0.9932 |
| QAT+PTQ | 0.7775 | 0.7994 | 0.8204 | 0.8430 | 0.8639 | 0.8839 | 0.9066 | 0.9280 | 0.9498 | 0.9704 | 0.9927 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7775 | 0.7994 | 0.8204 | 0.8430 | 0.8639 | 0.8839 | 0.9066 | 0.9280 | 0.9498 | 0.9704 | 0.9927 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2125 | 0.3772 | 0.5084 | 0.6159 | 0.7045 | 0.7801 | 0.8437 | 0.8998 | 0.9487 | 0.9914 |
| QAT+Prune only | 0.0000 | 0.4983 | 0.6895 | 0.7923 | 0.8543 | 0.8960 | 0.9278 | 0.9511 | 0.9697 | 0.9839 | 0.9966 |
| QAT+PTQ | 0.0000 | 0.4973 | 0.6886 | 0.7914 | 0.8537 | 0.8953 | 0.9273 | 0.9507 | 0.9693 | 0.9837 | 0.9963 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4973 | 0.6886 | 0.7914 | 0.8537 | 0.8953 | 0.9273 | 0.9507 | 0.9693 | 0.9837 | 0.9963 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1931 | 0.1929 | 0.1930 | 0.1927 | 0.1941 | 0.1925 | 0.1945 | 0.1905 | 0.1928 | 0.1976 | 0.0000 |
| QAT+Prune only | 0.7783 | 0.7786 | 0.7780 | 0.7797 | 0.7787 | 0.7763 | 0.7784 | 0.7775 | 0.7792 | 0.7695 | 0.0000 |
| QAT+PTQ | 0.7775 | 0.7780 | 0.7774 | 0.7789 | 0.7781 | 0.7752 | 0.7776 | 0.7770 | 0.7782 | 0.7694 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7775 | 0.7780 | 0.7774 | 0.7789 | 0.7781 | 0.7752 | 0.7776 | 0.7770 | 0.7782 | 0.7694 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1931 | 0.0000 | 0.0000 | 0.0000 | 0.1931 | 1.0000 |
| 90 | 10 | 299,940 | 0.2718 | 0.1191 | 0.9825 | 0.2125 | 0.1929 | 0.9900 |
| 80 | 20 | 291,350 | 0.3510 | 0.2334 | 0.9829 | 0.3772 | 0.1930 | 0.9783 |
| 70 | 30 | 194,230 | 0.4297 | 0.3429 | 0.9829 | 0.5084 | 0.1927 | 0.9633 |
| 60 | 40 | 145,675 | 0.5096 | 0.4485 | 0.9829 | 0.6159 | 0.1941 | 0.9445 |
| 50 | 50 | 116,540 | 0.5877 | 0.5490 | 0.9829 | 0.7045 | 0.1925 | 0.9184 |
| 40 | 60 | 97,115 | 0.6675 | 0.6467 | 0.9829 | 0.7801 | 0.1945 | 0.8834 |
| 30 | 70 | 83,240 | 0.7452 | 0.7391 | 0.9829 | 0.8437 | 0.1905 | 0.8267 |
| 20 | 80 | 72,835 | 0.8249 | 0.8297 | 0.9829 | 0.8998 | 0.1928 | 0.7380 |
| 10 | 90 | 64,740 | 0.9044 | 0.9168 | 0.9829 | 0.9487 | 0.1976 | 0.5620 |
| 0 | 100 | 58,270 | 0.9829 | 1.0000 | 0.9829 | 0.9914 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7783 | 0.0000 | 0.0000 | 0.0000 | 0.7783 | 1.0000 |
| 90 | 10 | 299,940 | 0.8001 | 0.3326 | 0.9928 | 0.4983 | 0.7786 | 0.9990 |
| 80 | 20 | 291,350 | 0.8211 | 0.5280 | 0.9932 | 0.6895 | 0.7780 | 0.9978 |
| 70 | 30 | 194,230 | 0.8437 | 0.6590 | 0.9932 | 0.7923 | 0.7797 | 0.9963 |
| 60 | 40 | 145,675 | 0.8645 | 0.7495 | 0.9932 | 0.8543 | 0.7787 | 0.9942 |
| 50 | 50 | 116,540 | 0.8847 | 0.8161 | 0.9932 | 0.8960 | 0.7763 | 0.9913 |
| 40 | 60 | 97,115 | 0.9073 | 0.8705 | 0.9932 | 0.9278 | 0.7784 | 0.9870 |
| 30 | 70 | 83,240 | 0.9285 | 0.9124 | 0.9932 | 0.9511 | 0.7775 | 0.9799 |
| 20 | 80 | 72,835 | 0.9504 | 0.9474 | 0.9932 | 0.9697 | 0.7792 | 0.9661 |
| 10 | 90 | 64,740 | 0.9708 | 0.9749 | 0.9932 | 0.9839 | 0.7695 | 0.9260 |
| 0 | 100 | 58,270 | 0.9932 | 1.0000 | 0.9932 | 0.9966 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7775 | 0.0000 | 0.0000 | 0.0000 | 0.7775 | 1.0000 |
| 90 | 10 | 299,940 | 0.7994 | 0.3318 | 0.9923 | 0.4973 | 0.7780 | 0.9989 |
| 80 | 20 | 291,350 | 0.8204 | 0.5271 | 0.9927 | 0.6886 | 0.7774 | 0.9976 |
| 70 | 30 | 194,230 | 0.8430 | 0.6580 | 0.9927 | 0.7914 | 0.7789 | 0.9960 |
| 60 | 40 | 145,675 | 0.8639 | 0.7489 | 0.9927 | 0.8537 | 0.7781 | 0.9938 |
| 50 | 50 | 116,540 | 0.8839 | 0.8154 | 0.9927 | 0.8953 | 0.7752 | 0.9906 |
| 40 | 60 | 97,115 | 0.9066 | 0.8701 | 0.9927 | 0.9273 | 0.7776 | 0.9861 |
| 30 | 70 | 83,240 | 0.9280 | 0.9122 | 0.9927 | 0.9507 | 0.7770 | 0.9785 |
| 20 | 80 | 72,835 | 0.9498 | 0.9471 | 0.9927 | 0.9693 | 0.7782 | 0.9637 |
| 10 | 90 | 64,740 | 0.9704 | 0.9748 | 0.9927 | 0.9837 | 0.7694 | 0.9212 |
| 0 | 100 | 58,270 | 0.9927 | 1.0000 | 0.9927 | 0.9963 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7775 | 0.0000 | 0.0000 | 0.0000 | 0.7775 | 1.0000 |
| 90 | 10 | 299,940 | 0.7994 | 0.3318 | 0.9923 | 0.4973 | 0.7780 | 0.9989 |
| 80 | 20 | 291,350 | 0.8204 | 0.5271 | 0.9927 | 0.6886 | 0.7774 | 0.9976 |
| 70 | 30 | 194,230 | 0.8430 | 0.6580 | 0.9927 | 0.7914 | 0.7789 | 0.9960 |
| 60 | 40 | 145,675 | 0.8639 | 0.7489 | 0.9927 | 0.8537 | 0.7781 | 0.9938 |
| 50 | 50 | 116,540 | 0.8839 | 0.8154 | 0.9927 | 0.8953 | 0.7752 | 0.9906 |
| 40 | 60 | 97,115 | 0.9066 | 0.8701 | 0.9927 | 0.9273 | 0.7776 | 0.9861 |
| 30 | 70 | 83,240 | 0.9280 | 0.9122 | 0.9927 | 0.9507 | 0.7770 | 0.9785 |
| 20 | 80 | 72,835 | 0.9498 | 0.9471 | 0.9927 | 0.9693 | 0.7782 | 0.9637 |
| 10 | 90 | 64,740 | 0.9704 | 0.9748 | 0.9927 | 0.9837 | 0.7694 | 0.9212 |
| 0 | 100 | 58,270 | 0.9927 | 1.0000 | 0.9927 | 0.9963 | 0.0000 | 0.0000 |


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
0.15       0.2718   0.2124   0.1929   0.9897   0.9819   0.1191   <--
0.20       0.2718   0.2124   0.1929   0.9897   0.9819   0.1191  
0.25       0.2718   0.2124   0.1929   0.9897   0.9819   0.1191  
0.30       0.2718   0.2124   0.1929   0.9897   0.9819   0.1191  
0.35       0.2718   0.2124   0.1929   0.9897   0.9819   0.1191  
0.40       0.2718   0.2124   0.1929   0.9897   0.9819   0.1191  
0.45       0.2718   0.2124   0.1929   0.9897   0.9819   0.1191  
0.50       0.2718   0.2124   0.1929   0.9897   0.9819   0.1191  
0.55       0.2718   0.2124   0.1929   0.9897   0.9819   0.1191  
0.60       0.2718   0.2124   0.1929   0.9897   0.9819   0.1191  
0.65       0.2718   0.2124   0.1929   0.9897   0.9819   0.1191  
0.70       0.2718   0.2124   0.1929   0.9897   0.9819   0.1191  
0.75       0.2718   0.2124   0.1929   0.9897   0.9819   0.1191  
0.80       0.2718   0.2124   0.1929   0.9897   0.9819   0.1191  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2718, F1=0.2124, Normal Recall=0.1929, Normal Precision=0.9897, Attack Recall=0.9819, Attack Precision=0.1191

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
0.15       0.3510   0.3772   0.1930   0.9783   0.9829   0.2334   <--
0.20       0.3510   0.3772   0.1930   0.9783   0.9829   0.2334  
0.25       0.3510   0.3772   0.1930   0.9783   0.9829   0.2334  
0.30       0.3510   0.3772   0.1930   0.9783   0.9829   0.2334  
0.35       0.3510   0.3772   0.1930   0.9783   0.9829   0.2334  
0.40       0.3510   0.3772   0.1930   0.9783   0.9829   0.2334  
0.45       0.3510   0.3772   0.1930   0.9783   0.9829   0.2334  
0.50       0.3510   0.3772   0.1930   0.9783   0.9829   0.2334  
0.55       0.3510   0.3772   0.1930   0.9783   0.9829   0.2334  
0.60       0.3510   0.3772   0.1930   0.9783   0.9829   0.2334  
0.65       0.3510   0.3772   0.1930   0.9783   0.9829   0.2334  
0.70       0.3510   0.3772   0.1930   0.9783   0.9829   0.2334  
0.75       0.3510   0.3772   0.1930   0.9783   0.9829   0.2334  
0.80       0.3510   0.3772   0.1930   0.9783   0.9829   0.2334  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3510, F1=0.3772, Normal Recall=0.1930, Normal Precision=0.9783, Attack Recall=0.9829, Attack Precision=0.2334

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
0.15       0.4302   0.5086   0.1933   0.9635   0.9829   0.3430   <--
0.20       0.4302   0.5086   0.1933   0.9635   0.9829   0.3430  
0.25       0.4302   0.5086   0.1933   0.9635   0.9829   0.3430  
0.30       0.4302   0.5086   0.1933   0.9635   0.9829   0.3430  
0.35       0.4302   0.5086   0.1933   0.9635   0.9829   0.3430  
0.40       0.4302   0.5086   0.1933   0.9635   0.9829   0.3430  
0.45       0.4302   0.5086   0.1933   0.9635   0.9829   0.3430  
0.50       0.4302   0.5086   0.1933   0.9635   0.9829   0.3430  
0.55       0.4302   0.5086   0.1933   0.9635   0.9829   0.3430  
0.60       0.4302   0.5086   0.1933   0.9635   0.9829   0.3430  
0.65       0.4302   0.5086   0.1933   0.9635   0.9829   0.3430  
0.70       0.4302   0.5086   0.1933   0.9635   0.9829   0.3430  
0.75       0.4302   0.5086   0.1933   0.9635   0.9829   0.3430  
0.80       0.4302   0.5086   0.1933   0.9635   0.9829   0.3430  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4302, F1=0.5086, Normal Recall=0.1933, Normal Precision=0.9635, Attack Recall=0.9829, Attack Precision=0.3430

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
0.15       0.5088   0.6155   0.1927   0.9441   0.9829   0.4480   <--
0.20       0.5088   0.6155   0.1927   0.9441   0.9829   0.4480  
0.25       0.5088   0.6155   0.1927   0.9441   0.9829   0.4480  
0.30       0.5088   0.6155   0.1927   0.9441   0.9829   0.4480  
0.35       0.5088   0.6155   0.1927   0.9441   0.9829   0.4480  
0.40       0.5088   0.6155   0.1927   0.9441   0.9829   0.4480  
0.45       0.5088   0.6155   0.1927   0.9441   0.9829   0.4480  
0.50       0.5088   0.6155   0.1927   0.9441   0.9829   0.4480  
0.55       0.5088   0.6155   0.1927   0.9441   0.9829   0.4480  
0.60       0.5088   0.6155   0.1927   0.9441   0.9829   0.4480  
0.65       0.5088   0.6155   0.1927   0.9441   0.9829   0.4480  
0.70       0.5088   0.6155   0.1927   0.9441   0.9829   0.4480  
0.75       0.5088   0.6155   0.1927   0.9441   0.9829   0.4480  
0.80       0.5088   0.6155   0.1927   0.9441   0.9829   0.4480  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5088, F1=0.6155, Normal Recall=0.1927, Normal Precision=0.9441, Attack Recall=0.9829, Attack Precision=0.4480

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
0.15       0.5878   0.7045   0.1927   0.9185   0.9829   0.5490   <--
0.20       0.5878   0.7045   0.1927   0.9185   0.9829   0.5490  
0.25       0.5878   0.7045   0.1927   0.9185   0.9829   0.5490  
0.30       0.5878   0.7045   0.1927   0.9185   0.9829   0.5490  
0.35       0.5878   0.7045   0.1927   0.9185   0.9829   0.5490  
0.40       0.5878   0.7045   0.1927   0.9185   0.9829   0.5490  
0.45       0.5878   0.7045   0.1927   0.9185   0.9829   0.5490  
0.50       0.5878   0.7045   0.1927   0.9185   0.9829   0.5490  
0.55       0.5878   0.7045   0.1927   0.9185   0.9829   0.5490  
0.60       0.5878   0.7045   0.1927   0.9185   0.9829   0.5490  
0.65       0.5878   0.7045   0.1927   0.9185   0.9829   0.5490  
0.70       0.5878   0.7045   0.1927   0.9185   0.9829   0.5490  
0.75       0.5878   0.7045   0.1927   0.9185   0.9829   0.5490  
0.80       0.5878   0.7045   0.1927   0.9185   0.9829   0.5490  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5878, F1=0.7045, Normal Recall=0.1927, Normal Precision=0.9185, Attack Recall=0.9829, Attack Precision=0.5490

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
0.15       0.8001   0.4983   0.7786   0.9990   0.9929   0.3326   <--
0.20       0.8001   0.4983   0.7786   0.9990   0.9929   0.3326  
0.25       0.8001   0.4983   0.7786   0.9990   0.9929   0.3326  
0.30       0.8001   0.4983   0.7786   0.9990   0.9929   0.3326  
0.35       0.8001   0.4983   0.7786   0.9990   0.9929   0.3326  
0.40       0.8001   0.4983   0.7786   0.9990   0.9929   0.3326  
0.45       0.8001   0.4983   0.7786   0.9990   0.9929   0.3326  
0.50       0.8001   0.4983   0.7786   0.9990   0.9929   0.3326  
0.55       0.8001   0.4983   0.7786   0.9990   0.9929   0.3326  
0.60       0.8001   0.4983   0.7786   0.9990   0.9929   0.3326  
0.65       0.8001   0.4983   0.7786   0.9990   0.9929   0.3326  
0.70       0.8001   0.4983   0.7786   0.9990   0.9929   0.3326  
0.75       0.8001   0.4983   0.7786   0.9990   0.9929   0.3326  
0.80       0.8001   0.4983   0.7786   0.9990   0.9929   0.3326  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8001, F1=0.4983, Normal Recall=0.7786, Normal Precision=0.9990, Attack Recall=0.9929, Attack Precision=0.3326

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
0.15       0.8220   0.6906   0.7792   0.9978   0.9932   0.5293   <--
0.20       0.8220   0.6906   0.7792   0.9978   0.9932   0.5293  
0.25       0.8220   0.6906   0.7792   0.9978   0.9932   0.5293  
0.30       0.8220   0.6906   0.7792   0.9978   0.9932   0.5293  
0.35       0.8220   0.6906   0.7792   0.9978   0.9932   0.5293  
0.40       0.8220   0.6906   0.7792   0.9978   0.9932   0.5293  
0.45       0.8220   0.6906   0.7792   0.9978   0.9932   0.5293  
0.50       0.8220   0.6906   0.7792   0.9978   0.9932   0.5293  
0.55       0.8220   0.6906   0.7792   0.9978   0.9932   0.5293  
0.60       0.8220   0.6906   0.7792   0.9978   0.9932   0.5293  
0.65       0.8220   0.6906   0.7792   0.9978   0.9932   0.5293  
0.70       0.8220   0.6906   0.7792   0.9978   0.9932   0.5293  
0.75       0.8220   0.6906   0.7792   0.9978   0.9932   0.5293  
0.80       0.8220   0.6906   0.7792   0.9978   0.9932   0.5293  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8220, F1=0.6906, Normal Recall=0.7792, Normal Precision=0.9978, Attack Recall=0.9932, Attack Precision=0.5293

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
0.15       0.8432   0.7917   0.7789   0.9963   0.9932   0.6581   <--
0.20       0.8432   0.7917   0.7789   0.9963   0.9932   0.6581  
0.25       0.8432   0.7917   0.7789   0.9963   0.9932   0.6581  
0.30       0.8432   0.7917   0.7789   0.9963   0.9932   0.6581  
0.35       0.8432   0.7917   0.7789   0.9963   0.9932   0.6581  
0.40       0.8432   0.7917   0.7789   0.9963   0.9932   0.6581  
0.45       0.8432   0.7917   0.7789   0.9963   0.9932   0.6581  
0.50       0.8432   0.7917   0.7789   0.9963   0.9932   0.6581  
0.55       0.8432   0.7917   0.7789   0.9963   0.9932   0.6581  
0.60       0.8432   0.7917   0.7789   0.9963   0.9932   0.6581  
0.65       0.8432   0.7917   0.7789   0.9963   0.9932   0.6581  
0.70       0.8432   0.7917   0.7789   0.9963   0.9932   0.6581  
0.75       0.8432   0.7917   0.7789   0.9963   0.9932   0.6581  
0.80       0.8432   0.7917   0.7789   0.9963   0.9932   0.6581  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8432, F1=0.7917, Normal Recall=0.7789, Normal Precision=0.9963, Attack Recall=0.9932, Attack Precision=0.6581

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
0.15       0.8644   0.8542   0.7785   0.9942   0.9932   0.7493   <--
0.20       0.8644   0.8542   0.7785   0.9942   0.9932   0.7493  
0.25       0.8644   0.8542   0.7785   0.9942   0.9932   0.7493  
0.30       0.8644   0.8542   0.7785   0.9942   0.9932   0.7493  
0.35       0.8644   0.8542   0.7785   0.9942   0.9932   0.7493  
0.40       0.8644   0.8542   0.7785   0.9942   0.9932   0.7493  
0.45       0.8644   0.8542   0.7785   0.9942   0.9932   0.7493  
0.50       0.8644   0.8542   0.7785   0.9942   0.9932   0.7493  
0.55       0.8644   0.8542   0.7785   0.9942   0.9932   0.7493  
0.60       0.8644   0.8542   0.7785   0.9942   0.9932   0.7493  
0.65       0.8644   0.8542   0.7785   0.9942   0.9932   0.7493  
0.70       0.8644   0.8542   0.7785   0.9942   0.9932   0.7493  
0.75       0.8644   0.8542   0.7785   0.9942   0.9932   0.7493  
0.80       0.8644   0.8542   0.7785   0.9942   0.9932   0.7493  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8644, F1=0.8542, Normal Recall=0.7785, Normal Precision=0.9942, Attack Recall=0.9932, Attack Precision=0.7493

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
0.15       0.8848   0.8961   0.7765   0.9913   0.9932   0.8163   <--
0.20       0.8848   0.8961   0.7765   0.9913   0.9932   0.8163  
0.25       0.8848   0.8961   0.7765   0.9913   0.9932   0.8163  
0.30       0.8848   0.8961   0.7765   0.9913   0.9932   0.8163  
0.35       0.8848   0.8961   0.7765   0.9913   0.9932   0.8163  
0.40       0.8848   0.8961   0.7765   0.9913   0.9932   0.8163  
0.45       0.8848   0.8961   0.7765   0.9913   0.9932   0.8163  
0.50       0.8848   0.8961   0.7765   0.9913   0.9932   0.8163  
0.55       0.8848   0.8961   0.7765   0.9913   0.9932   0.8163  
0.60       0.8848   0.8961   0.7765   0.9913   0.9932   0.8163  
0.65       0.8848   0.8961   0.7765   0.9913   0.9932   0.8163  
0.70       0.8848   0.8961   0.7765   0.9913   0.9932   0.8163  
0.75       0.8848   0.8961   0.7765   0.9913   0.9932   0.8163  
0.80       0.8848   0.8961   0.7765   0.9913   0.9932   0.8163  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8848, F1=0.8961, Normal Recall=0.7765, Normal Precision=0.9913, Attack Recall=0.9932, Attack Precision=0.8163

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
0.15       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318   <--
0.20       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.25       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.30       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.35       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.40       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.45       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.50       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.55       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.60       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.65       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.70       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.75       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.80       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7994, F1=0.4973, Normal Recall=0.7780, Normal Precision=0.9989, Attack Recall=0.9923, Attack Precision=0.3318

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
0.15       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284   <--
0.20       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.25       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.30       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.35       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.40       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.45       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.50       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.55       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.60       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.65       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.70       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.75       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.80       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8213, F1=0.6897, Normal Recall=0.7785, Normal Precision=0.9977, Attack Recall=0.9927, Attack Precision=0.5284

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
0.15       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573   <--
0.20       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.25       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.30       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.35       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.40       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.45       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.50       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.55       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.60       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.65       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.70       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.75       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.80       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8425, F1=0.7909, Normal Recall=0.7782, Normal Precision=0.9960, Attack Recall=0.9927, Attack Precision=0.6573

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
0.15       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486   <--
0.20       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.25       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.30       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.35       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.40       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.45       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.50       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.55       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.60       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.65       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.70       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.75       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.80       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8637, F1=0.8535, Normal Recall=0.7777, Normal Precision=0.9938, Attack Recall=0.9927, Attack Precision=0.7486

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
0.15       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156   <--
0.20       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.25       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.30       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.35       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.40       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.45       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.50       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.55       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.60       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.65       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.70       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.75       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.80       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8842, F1=0.8955, Normal Recall=0.7756, Normal Precision=0.9906, Attack Recall=0.9927, Attack Precision=0.8156

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
0.15       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318   <--
0.20       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.25       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.30       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.35       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.40       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.45       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.50       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.55       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.60       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.65       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.70       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.75       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
0.80       0.7994   0.4973   0.7780   0.9989   0.9923   0.3318  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7994, F1=0.4973, Normal Recall=0.7780, Normal Precision=0.9989, Attack Recall=0.9923, Attack Precision=0.3318

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
0.15       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284   <--
0.20       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.25       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.30       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.35       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.40       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.45       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.50       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.55       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.60       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.65       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.70       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.75       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
0.80       0.8213   0.6897   0.7785   0.9977   0.9927   0.5284  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8213, F1=0.6897, Normal Recall=0.7785, Normal Precision=0.9977, Attack Recall=0.9927, Attack Precision=0.5284

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
0.15       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573   <--
0.20       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.25       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.30       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.35       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.40       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.45       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.50       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.55       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.60       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.65       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.70       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.75       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
0.80       0.8425   0.7909   0.7782   0.9960   0.9927   0.6573  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8425, F1=0.7909, Normal Recall=0.7782, Normal Precision=0.9960, Attack Recall=0.9927, Attack Precision=0.6573

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
0.15       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486   <--
0.20       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.25       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.30       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.35       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.40       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.45       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.50       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.55       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.60       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.65       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.70       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.75       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
0.80       0.8637   0.8535   0.7777   0.9938   0.9927   0.7486  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8637, F1=0.8535, Normal Recall=0.7777, Normal Precision=0.9938, Attack Recall=0.9927, Attack Precision=0.7486

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
0.15       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156   <--
0.20       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.25       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.30       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.35       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.40       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.45       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.50       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.55       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.60       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.65       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.70       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.75       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
0.80       0.8842   0.8955   0.7756   0.9906   0.9927   0.8156  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8842, F1=0.8955, Normal Recall=0.7756, Normal Precision=0.9906, Attack Recall=0.9927, Attack Precision=0.8156

```

