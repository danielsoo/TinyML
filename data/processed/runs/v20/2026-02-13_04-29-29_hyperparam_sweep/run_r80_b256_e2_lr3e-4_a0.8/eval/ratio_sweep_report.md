# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e2_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-18 21:03:40 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7582 | 0.7746 | 0.7902 | 0.8073 | 0.8236 | 0.8397 | 0.8550 | 0.8707 | 0.8874 | 0.9044 | 0.9204 |
| QAT+Prune only | 0.8250 | 0.8398 | 0.8538 | 0.8677 | 0.8837 | 0.8959 | 0.9125 | 0.9259 | 0.9408 | 0.9550 | 0.9698 |
| QAT+PTQ | 0.8249 | 0.8398 | 0.8538 | 0.8678 | 0.8837 | 0.8958 | 0.9126 | 0.9258 | 0.9408 | 0.9549 | 0.9698 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8249 | 0.8398 | 0.8538 | 0.8678 | 0.8837 | 0.8958 | 0.9126 | 0.9258 | 0.9408 | 0.9549 | 0.9698 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.4497 | 0.6369 | 0.7413 | 0.8068 | 0.8517 | 0.8839 | 0.9088 | 0.9290 | 0.9454 | 0.9585 |
| QAT+Prune only | 0.0000 | 0.5476 | 0.7263 | 0.8148 | 0.8696 | 0.9030 | 0.9301 | 0.9482 | 0.9632 | 0.9748 | 0.9847 |
| QAT+PTQ | 0.0000 | 0.5477 | 0.7263 | 0.8148 | 0.8696 | 0.9029 | 0.9301 | 0.9482 | 0.9632 | 0.9748 | 0.9847 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5477 | 0.7263 | 0.8148 | 0.8696 | 0.9029 | 0.9301 | 0.9482 | 0.9632 | 0.9748 | 0.9847 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7582 | 0.7584 | 0.7576 | 0.7588 | 0.7591 | 0.7590 | 0.7569 | 0.7549 | 0.7556 | 0.7604 | 0.0000 |
| QAT+Prune only | 0.8250 | 0.8254 | 0.8248 | 0.8239 | 0.8262 | 0.8219 | 0.8266 | 0.8233 | 0.8247 | 0.8213 | 0.0000 |
| QAT+PTQ | 0.8249 | 0.8254 | 0.8248 | 0.8241 | 0.8262 | 0.8217 | 0.8268 | 0.8232 | 0.8247 | 0.8208 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8249 | 0.8254 | 0.8248 | 0.8241 | 0.8262 | 0.8217 | 0.8268 | 0.8232 | 0.8247 | 0.8208 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7582 | 0.0000 | 0.0000 | 0.0000 | 0.7582 | 1.0000 |
| 90 | 10 | 299,940 | 0.7746 | 0.2975 | 0.9210 | 0.4497 | 0.7584 | 0.9886 |
| 80 | 20 | 291,350 | 0.7902 | 0.4870 | 0.9204 | 0.6369 | 0.7576 | 0.9744 |
| 70 | 30 | 194,230 | 0.8073 | 0.6205 | 0.9204 | 0.7413 | 0.7588 | 0.9570 |
| 60 | 40 | 145,675 | 0.8236 | 0.7181 | 0.9204 | 0.8068 | 0.7591 | 0.9346 |
| 50 | 50 | 116,540 | 0.8397 | 0.7925 | 0.9204 | 0.8517 | 0.7590 | 0.9050 |
| 40 | 60 | 97,115 | 0.8550 | 0.8503 | 0.9204 | 0.8839 | 0.7569 | 0.8637 |
| 30 | 70 | 83,240 | 0.8707 | 0.8976 | 0.9204 | 0.9088 | 0.7549 | 0.8025 |
| 20 | 80 | 72,835 | 0.8874 | 0.9377 | 0.9204 | 0.9290 | 0.7556 | 0.7035 |
| 10 | 90 | 64,740 | 0.9044 | 0.9719 | 0.9204 | 0.9454 | 0.7604 | 0.5148 |
| 0 | 100 | 58,270 | 0.9204 | 1.0000 | 0.9204 | 0.9585 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8250 | 0.0000 | 0.0000 | 0.0000 | 0.8250 | 1.0000 |
| 90 | 10 | 299,940 | 0.8398 | 0.3815 | 0.9696 | 0.5476 | 0.8254 | 0.9959 |
| 80 | 20 | 291,350 | 0.8538 | 0.5806 | 0.9698 | 0.7263 | 0.8248 | 0.9909 |
| 70 | 30 | 194,230 | 0.8677 | 0.7025 | 0.9698 | 0.8148 | 0.8239 | 0.9845 |
| 60 | 40 | 145,675 | 0.8837 | 0.7882 | 0.9698 | 0.8696 | 0.8262 | 0.9762 |
| 50 | 50 | 116,540 | 0.8959 | 0.8449 | 0.9698 | 0.9030 | 0.8219 | 0.9646 |
| 40 | 60 | 97,115 | 0.9125 | 0.8935 | 0.9698 | 0.9301 | 0.8266 | 0.9481 |
| 30 | 70 | 83,240 | 0.9259 | 0.9276 | 0.9698 | 0.9482 | 0.8233 | 0.9212 |
| 20 | 80 | 72,835 | 0.9408 | 0.9568 | 0.9698 | 0.9632 | 0.8247 | 0.8723 |
| 10 | 90 | 64,740 | 0.9550 | 0.9799 | 0.9698 | 0.9748 | 0.8213 | 0.7514 |
| 0 | 100 | 58,270 | 0.9698 | 1.0000 | 0.9698 | 0.9847 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8249 | 0.0000 | 0.0000 | 0.0000 | 0.8249 | 1.0000 |
| 90 | 10 | 299,940 | 0.8398 | 0.3816 | 0.9696 | 0.5477 | 0.8254 | 0.9959 |
| 80 | 20 | 291,350 | 0.8538 | 0.5806 | 0.9698 | 0.7263 | 0.8248 | 0.9909 |
| 70 | 30 | 194,230 | 0.8678 | 0.7026 | 0.9698 | 0.8148 | 0.8241 | 0.9845 |
| 60 | 40 | 145,675 | 0.8837 | 0.7882 | 0.9698 | 0.8696 | 0.8262 | 0.9762 |
| 50 | 50 | 116,540 | 0.8958 | 0.8447 | 0.9698 | 0.9029 | 0.8217 | 0.9645 |
| 40 | 60 | 97,115 | 0.9126 | 0.8936 | 0.9698 | 0.9301 | 0.8268 | 0.9480 |
| 30 | 70 | 83,240 | 0.9258 | 0.9275 | 0.9698 | 0.9482 | 0.8232 | 0.9211 |
| 20 | 80 | 72,835 | 0.9408 | 0.9568 | 0.9698 | 0.9632 | 0.8247 | 0.8722 |
| 10 | 90 | 64,740 | 0.9549 | 0.9799 | 0.9698 | 0.9748 | 0.8208 | 0.7512 |
| 0 | 100 | 58,270 | 0.9698 | 1.0000 | 0.9698 | 0.9847 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8249 | 0.0000 | 0.0000 | 0.0000 | 0.8249 | 1.0000 |
| 90 | 10 | 299,940 | 0.8398 | 0.3816 | 0.9696 | 0.5477 | 0.8254 | 0.9959 |
| 80 | 20 | 291,350 | 0.8538 | 0.5806 | 0.9698 | 0.7263 | 0.8248 | 0.9909 |
| 70 | 30 | 194,230 | 0.8678 | 0.7026 | 0.9698 | 0.8148 | 0.8241 | 0.9845 |
| 60 | 40 | 145,675 | 0.8837 | 0.7882 | 0.9698 | 0.8696 | 0.8262 | 0.9762 |
| 50 | 50 | 116,540 | 0.8958 | 0.8447 | 0.9698 | 0.9029 | 0.8217 | 0.9645 |
| 40 | 60 | 97,115 | 0.9126 | 0.8936 | 0.9698 | 0.9301 | 0.8268 | 0.9480 |
| 30 | 70 | 83,240 | 0.9258 | 0.9275 | 0.9698 | 0.9482 | 0.8232 | 0.9211 |
| 20 | 80 | 72,835 | 0.9408 | 0.9568 | 0.9698 | 0.9632 | 0.8247 | 0.8722 |
| 10 | 90 | 64,740 | 0.9549 | 0.9799 | 0.9698 | 0.9748 | 0.8208 | 0.7512 |
| 0 | 100 | 58,270 | 0.9698 | 1.0000 | 0.9698 | 0.9847 | 0.0000 | 0.0000 |


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
0.15       0.7747   0.4499   0.7584   0.9886   0.9215   0.2976   <--
0.20       0.7747   0.4499   0.7584   0.9886   0.9215   0.2976  
0.25       0.7747   0.4499   0.7584   0.9886   0.9215   0.2976  
0.30       0.7747   0.4499   0.7584   0.9886   0.9215   0.2976  
0.35       0.7747   0.4499   0.7584   0.9886   0.9215   0.2976  
0.40       0.7747   0.4499   0.7584   0.9886   0.9215   0.2976  
0.45       0.7747   0.4499   0.7584   0.9886   0.9215   0.2976  
0.50       0.7747   0.4499   0.7584   0.9886   0.9215   0.2976  
0.55       0.7747   0.4499   0.7584   0.9886   0.9215   0.2976  
0.60       0.7747   0.4499   0.7584   0.9886   0.9215   0.2976  
0.65       0.7747   0.4499   0.7584   0.9886   0.9215   0.2976  
0.70       0.7747   0.4499   0.7584   0.9886   0.9215   0.2976  
0.75       0.7747   0.4499   0.7584   0.9886   0.9215   0.2976  
0.80       0.7747   0.4499   0.7584   0.9886   0.9215   0.2976  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7747, F1=0.4499, Normal Recall=0.7584, Normal Precision=0.9886, Attack Recall=0.9215, Attack Precision=0.2976

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
0.15       0.7909   0.6378   0.7585   0.9744   0.9204   0.4879   <--
0.20       0.7909   0.6378   0.7585   0.9744   0.9204   0.4879  
0.25       0.7909   0.6378   0.7585   0.9744   0.9204   0.4879  
0.30       0.7909   0.6378   0.7585   0.9744   0.9204   0.4879  
0.35       0.7909   0.6378   0.7585   0.9744   0.9204   0.4879  
0.40       0.7909   0.6378   0.7585   0.9744   0.9204   0.4879  
0.45       0.7909   0.6378   0.7585   0.9744   0.9204   0.4879  
0.50       0.7909   0.6378   0.7585   0.9744   0.9204   0.4879  
0.55       0.7909   0.6378   0.7585   0.9744   0.9204   0.4879  
0.60       0.7909   0.6378   0.7585   0.9744   0.9204   0.4879  
0.65       0.7909   0.6378   0.7585   0.9744   0.9204   0.4879  
0.70       0.7909   0.6378   0.7585   0.9744   0.9204   0.4879  
0.75       0.7909   0.6378   0.7585   0.9744   0.9204   0.4879  
0.80       0.7909   0.6378   0.7585   0.9744   0.9204   0.4879  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7909, F1=0.6378, Normal Recall=0.7585, Normal Precision=0.9744, Attack Recall=0.9204, Attack Precision=0.4879

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
0.15       0.8071   0.7411   0.7586   0.9569   0.9204   0.6203   <--
0.20       0.8071   0.7411   0.7586   0.9569   0.9204   0.6203  
0.25       0.8071   0.7411   0.7586   0.9569   0.9204   0.6203  
0.30       0.8071   0.7411   0.7586   0.9569   0.9204   0.6203  
0.35       0.8071   0.7411   0.7586   0.9569   0.9204   0.6203  
0.40       0.8071   0.7411   0.7586   0.9569   0.9204   0.6203  
0.45       0.8071   0.7411   0.7586   0.9569   0.9204   0.6203  
0.50       0.8071   0.7411   0.7586   0.9569   0.9204   0.6203  
0.55       0.8071   0.7411   0.7586   0.9569   0.9204   0.6203  
0.60       0.8071   0.7411   0.7586   0.9569   0.9204   0.6203  
0.65       0.8071   0.7411   0.7586   0.9569   0.9204   0.6203  
0.70       0.8071   0.7411   0.7586   0.9569   0.9204   0.6203  
0.75       0.8071   0.7411   0.7586   0.9569   0.9204   0.6203  
0.80       0.8071   0.7411   0.7586   0.9569   0.9204   0.6203  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8071, F1=0.7411, Normal Recall=0.7586, Normal Precision=0.9569, Attack Recall=0.9204, Attack Precision=0.6203

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
0.15       0.8232   0.8064   0.7584   0.9346   0.9204   0.7175   <--
0.20       0.8232   0.8064   0.7584   0.9346   0.9204   0.7175  
0.25       0.8232   0.8064   0.7584   0.9346   0.9204   0.7175  
0.30       0.8232   0.8064   0.7584   0.9346   0.9204   0.7175  
0.35       0.8232   0.8064   0.7584   0.9346   0.9204   0.7175  
0.40       0.8232   0.8064   0.7584   0.9346   0.9204   0.7175  
0.45       0.8232   0.8064   0.7584   0.9346   0.9204   0.7175  
0.50       0.8232   0.8064   0.7584   0.9346   0.9204   0.7175  
0.55       0.8232   0.8064   0.7584   0.9346   0.9204   0.7175  
0.60       0.8232   0.8064   0.7584   0.9346   0.9204   0.7175  
0.65       0.8232   0.8064   0.7584   0.9346   0.9204   0.7175  
0.70       0.8232   0.8064   0.7584   0.9346   0.9204   0.7175  
0.75       0.8232   0.8064   0.7584   0.9346   0.9204   0.7175  
0.80       0.8232   0.8064   0.7584   0.9346   0.9204   0.7175  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8232, F1=0.8064, Normal Recall=0.7584, Normal Precision=0.9346, Attack Recall=0.9204, Attack Precision=0.7175

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
0.15       0.8389   0.8511   0.7575   0.9049   0.9204   0.7915   <--
0.20       0.8389   0.8511   0.7575   0.9049   0.9204   0.7915  
0.25       0.8389   0.8511   0.7575   0.9049   0.9204   0.7915  
0.30       0.8389   0.8511   0.7575   0.9049   0.9204   0.7915  
0.35       0.8389   0.8511   0.7575   0.9049   0.9204   0.7915  
0.40       0.8389   0.8511   0.7575   0.9049   0.9204   0.7915  
0.45       0.8389   0.8511   0.7575   0.9049   0.9204   0.7915  
0.50       0.8389   0.8511   0.7575   0.9049   0.9204   0.7915  
0.55       0.8389   0.8511   0.7575   0.9049   0.9204   0.7915  
0.60       0.8389   0.8511   0.7575   0.9049   0.9204   0.7915  
0.65       0.8389   0.8511   0.7575   0.9049   0.9204   0.7915  
0.70       0.8389   0.8511   0.7575   0.9049   0.9204   0.7915  
0.75       0.8389   0.8511   0.7575   0.9049   0.9204   0.7915  
0.80       0.8389   0.8511   0.7575   0.9049   0.9204   0.7915  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8389, F1=0.8511, Normal Recall=0.7575, Normal Precision=0.9049, Attack Recall=0.9204, Attack Precision=0.7915

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
0.15       0.8399   0.5479   0.8254   0.9960   0.9704   0.3817   <--
0.20       0.8399   0.5479   0.8254   0.9960   0.9704   0.3817  
0.25       0.8399   0.5479   0.8254   0.9960   0.9704   0.3817  
0.30       0.8399   0.5479   0.8254   0.9960   0.9704   0.3817  
0.35       0.8399   0.5479   0.8254   0.9960   0.9704   0.3817  
0.40       0.8399   0.5479   0.8254   0.9960   0.9704   0.3817  
0.45       0.8399   0.5479   0.8254   0.9960   0.9704   0.3817  
0.50       0.8399   0.5479   0.8254   0.9960   0.9704   0.3817  
0.55       0.8399   0.5479   0.8254   0.9960   0.9704   0.3817  
0.60       0.8399   0.5479   0.8254   0.9960   0.9704   0.3817  
0.65       0.8399   0.5479   0.8254   0.9960   0.9704   0.3817  
0.70       0.8399   0.5479   0.8254   0.9960   0.9704   0.3817  
0.75       0.8399   0.5479   0.8254   0.9960   0.9704   0.3817  
0.80       0.8399   0.5479   0.8254   0.9960   0.9704   0.3817  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8399, F1=0.5479, Normal Recall=0.8254, Normal Precision=0.9960, Attack Recall=0.9704, Attack Precision=0.3817

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
0.15       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812   <--
0.20       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.25       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.30       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.35       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.40       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.45       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.50       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.55       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.60       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.65       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.70       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.75       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.80       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8542, F1=0.7268, Normal Recall=0.8253, Normal Precision=0.9909, Attack Recall=0.9698, Attack Precision=0.5812

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
0.15       0.8686   0.8157   0.8252   0.9846   0.9698   0.7039   <--
0.20       0.8686   0.8157   0.8252   0.9846   0.9698   0.7039  
0.25       0.8686   0.8157   0.8252   0.9846   0.9698   0.7039  
0.30       0.8686   0.8157   0.8252   0.9846   0.9698   0.7039  
0.35       0.8686   0.8157   0.8252   0.9846   0.9698   0.7039  
0.40       0.8686   0.8157   0.8252   0.9846   0.9698   0.7039  
0.45       0.8686   0.8157   0.8252   0.9846   0.9698   0.7039  
0.50       0.8686   0.8157   0.8252   0.9846   0.9698   0.7039  
0.55       0.8686   0.8157   0.8252   0.9846   0.9698   0.7039  
0.60       0.8686   0.8157   0.8252   0.9846   0.9698   0.7039  
0.65       0.8686   0.8157   0.8252   0.9846   0.9698   0.7039  
0.70       0.8686   0.8157   0.8252   0.9846   0.9698   0.7039  
0.75       0.8686   0.8157   0.8252   0.9846   0.9698   0.7039  
0.80       0.8686   0.8157   0.8252   0.9846   0.9698   0.7039  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8686, F1=0.8157, Normal Recall=0.8252, Normal Precision=0.9846, Attack Recall=0.9698, Attack Precision=0.7039

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
0.15       0.8830   0.8690   0.8252   0.9762   0.9698   0.7872   <--
0.20       0.8830   0.8690   0.8252   0.9762   0.9698   0.7872  
0.25       0.8830   0.8690   0.8252   0.9762   0.9698   0.7872  
0.30       0.8830   0.8690   0.8252   0.9762   0.9698   0.7872  
0.35       0.8830   0.8690   0.8252   0.9762   0.9698   0.7872  
0.40       0.8830   0.8690   0.8252   0.9762   0.9698   0.7872  
0.45       0.8830   0.8690   0.8252   0.9762   0.9698   0.7872  
0.50       0.8830   0.8690   0.8252   0.9762   0.9698   0.7872  
0.55       0.8830   0.8690   0.8252   0.9762   0.9698   0.7872  
0.60       0.8830   0.8690   0.8252   0.9762   0.9698   0.7872  
0.65       0.8830   0.8690   0.8252   0.9762   0.9698   0.7872  
0.70       0.8830   0.8690   0.8252   0.9762   0.9698   0.7872  
0.75       0.8830   0.8690   0.8252   0.9762   0.9698   0.7872  
0.80       0.8830   0.8690   0.8252   0.9762   0.9698   0.7872  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8830, F1=0.8690, Normal Recall=0.8252, Normal Precision=0.9762, Attack Recall=0.9698, Attack Precision=0.7872

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
0.15       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475   <--
0.20       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.25       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.30       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.35       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.40       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.45       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.50       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.55       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.60       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.65       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.70       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.75       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.80       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8976, F1=0.9045, Normal Recall=0.8255, Normal Precision=0.9647, Attack Recall=0.9698, Attack Precision=0.8475

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
0.15       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818   <--
0.20       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.25       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.30       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.35       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.40       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.45       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.50       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.55       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.60       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.65       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.70       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.75       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.80       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8399, F1=0.5480, Normal Recall=0.8254, Normal Precision=0.9960, Attack Recall=0.9704, Attack Precision=0.3818

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
0.15       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812   <--
0.20       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.25       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.30       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.35       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.40       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.45       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.50       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.55       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.60       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.65       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.70       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.75       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.80       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8542, F1=0.7268, Normal Recall=0.8253, Normal Precision=0.9909, Attack Recall=0.9698, Attack Precision=0.5812

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
0.15       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039   <--
0.20       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.25       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.30       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.35       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.40       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.45       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.50       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.55       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.60       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.65       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.70       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.75       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.80       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8685, F1=0.8157, Normal Recall=0.8251, Normal Precision=0.9846, Attack Recall=0.9698, Attack Precision=0.7039

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
0.15       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870   <--
0.20       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.25       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.30       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.35       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.40       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.45       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.50       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.55       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.60       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.65       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.70       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.75       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.80       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8829, F1=0.8689, Normal Recall=0.8250, Normal Precision=0.9762, Attack Recall=0.9698, Attack Precision=0.7870

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
0.15       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475   <--
0.20       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.25       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.30       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.35       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.40       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.45       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.50       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.55       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.60       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.65       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.70       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.75       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.80       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8976, F1=0.9045, Normal Recall=0.8255, Normal Precision=0.9647, Attack Recall=0.9698, Attack Precision=0.8475

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
0.15       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818   <--
0.20       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.25       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.30       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.35       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.40       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.45       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.50       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.55       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.60       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.65       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.70       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.75       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
0.80       0.8399   0.5480   0.8254   0.9960   0.9704   0.3818  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8399, F1=0.5480, Normal Recall=0.8254, Normal Precision=0.9960, Attack Recall=0.9704, Attack Precision=0.3818

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
0.15       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812   <--
0.20       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.25       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.30       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.35       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.40       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.45       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.50       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.55       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.60       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.65       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.70       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.75       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
0.80       0.8542   0.7268   0.8253   0.9909   0.9698   0.5812  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8542, F1=0.7268, Normal Recall=0.8253, Normal Precision=0.9909, Attack Recall=0.9698, Attack Precision=0.5812

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
0.15       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039   <--
0.20       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.25       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.30       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.35       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.40       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.45       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.50       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.55       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.60       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.65       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.70       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.75       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
0.80       0.8685   0.8157   0.8251   0.9846   0.9698   0.7039  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8685, F1=0.8157, Normal Recall=0.8251, Normal Precision=0.9846, Attack Recall=0.9698, Attack Precision=0.7039

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
0.15       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870   <--
0.20       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.25       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.30       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.35       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.40       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.45       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.50       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.55       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.60       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.65       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.70       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.75       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
0.80       0.8829   0.8689   0.8250   0.9762   0.9698   0.7870  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8829, F1=0.8689, Normal Recall=0.8250, Normal Precision=0.9762, Attack Recall=0.9698, Attack Precision=0.7870

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
0.15       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475   <--
0.20       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.25       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.30       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.35       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.40       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.45       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.50       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.55       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.60       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.65       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.70       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.75       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
0.80       0.8976   0.9045   0.8255   0.9647   0.9698   0.8475  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8976, F1=0.9045, Normal Recall=0.8255, Normal Precision=0.9647, Attack Recall=0.9698, Attack Precision=0.8475

```

