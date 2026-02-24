# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e2_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-13 23:15:26 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7862 | 0.7670 | 0.7502 | 0.7326 | 0.7151 | 0.6990 | 0.6812 | 0.6653 | 0.6470 | 0.6298 | 0.6131 |
| QAT+Prune only | 0.9651 | 0.9481 | 0.9312 | 0.9145 | 0.8971 | 0.8800 | 0.8635 | 0.8462 | 0.8292 | 0.8123 | 0.7953 |
| QAT+PTQ | 0.9653 | 0.9482 | 0.9311 | 0.9144 | 0.8969 | 0.8798 | 0.8632 | 0.8457 | 0.8287 | 0.8117 | 0.7946 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9653 | 0.9482 | 0.9311 | 0.9144 | 0.8969 | 0.8798 | 0.8632 | 0.8457 | 0.8287 | 0.8117 | 0.7946 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3445 | 0.4954 | 0.5791 | 0.6326 | 0.6707 | 0.6977 | 0.7195 | 0.7353 | 0.7488 | 0.7601 |
| QAT+Prune only | 0.0000 | 0.7537 | 0.8221 | 0.8481 | 0.8608 | 0.8689 | 0.8749 | 0.8786 | 0.8817 | 0.8841 | 0.8860 |
| QAT+PTQ | 0.0000 | 0.7538 | 0.8219 | 0.8478 | 0.8604 | 0.8686 | 0.8745 | 0.8782 | 0.8812 | 0.8837 | 0.8856 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.7538 | 0.8219 | 0.8478 | 0.8604 | 0.8686 | 0.8745 | 0.8782 | 0.8812 | 0.8837 | 0.8856 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7862 | 0.7842 | 0.7845 | 0.7838 | 0.7832 | 0.7850 | 0.7834 | 0.7873 | 0.7825 | 0.7804 | 0.0000 |
| QAT+Prune only | 0.9651 | 0.9652 | 0.9651 | 0.9656 | 0.9649 | 0.9648 | 0.9657 | 0.9648 | 0.9647 | 0.9651 | 0.0000 |
| QAT+PTQ | 0.9653 | 0.9653 | 0.9652 | 0.9657 | 0.9650 | 0.9649 | 0.9660 | 0.9647 | 0.9647 | 0.9654 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9653 | 0.9653 | 0.9652 | 0.9657 | 0.9650 | 0.9649 | 0.9660 | 0.9647 | 0.9647 | 0.9654 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7862 | 0.0000 | 0.0000 | 0.0000 | 0.7862 | 1.0000 |
| 90 | 10 | 299,940 | 0.7670 | 0.2397 | 0.6122 | 0.3445 | 0.7842 | 0.9479 |
| 80 | 20 | 291,350 | 0.7502 | 0.4156 | 0.6131 | 0.4954 | 0.7845 | 0.8902 |
| 70 | 30 | 194,230 | 0.7326 | 0.5486 | 0.6131 | 0.5791 | 0.7838 | 0.8254 |
| 60 | 40 | 145,675 | 0.7151 | 0.6534 | 0.6131 | 0.6326 | 0.7832 | 0.7522 |
| 50 | 50 | 116,540 | 0.6990 | 0.7404 | 0.6131 | 0.6707 | 0.7850 | 0.6698 |
| 40 | 60 | 97,115 | 0.6812 | 0.8093 | 0.6131 | 0.6977 | 0.7834 | 0.5744 |
| 30 | 70 | 83,240 | 0.6653 | 0.8706 | 0.6131 | 0.7195 | 0.7873 | 0.4658 |
| 20 | 80 | 72,835 | 0.6470 | 0.9185 | 0.6131 | 0.7353 | 0.7825 | 0.3358 |
| 10 | 90 | 64,740 | 0.6298 | 0.9617 | 0.6131 | 0.7488 | 0.7804 | 0.1831 |
| 0 | 100 | 58,270 | 0.6131 | 1.0000 | 0.6131 | 0.7601 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9651 | 0.0000 | 0.0000 | 0.0000 | 0.9651 | 1.0000 |
| 90 | 10 | 299,940 | 0.9481 | 0.7172 | 0.7941 | 0.7537 | 0.9652 | 0.9768 |
| 80 | 20 | 291,350 | 0.9312 | 0.8508 | 0.7953 | 0.8221 | 0.9651 | 0.9497 |
| 70 | 30 | 194,230 | 0.9145 | 0.9084 | 0.7953 | 0.8481 | 0.9656 | 0.9167 |
| 60 | 40 | 145,675 | 0.8971 | 0.9379 | 0.7953 | 0.8608 | 0.9649 | 0.8761 |
| 50 | 50 | 116,540 | 0.8800 | 0.9576 | 0.7953 | 0.8689 | 0.9648 | 0.8250 |
| 40 | 60 | 97,115 | 0.8635 | 0.9720 | 0.7953 | 0.8749 | 0.9657 | 0.7588 |
| 30 | 70 | 83,240 | 0.8462 | 0.9814 | 0.7953 | 0.8786 | 0.9648 | 0.6689 |
| 20 | 80 | 72,835 | 0.8292 | 0.9890 | 0.7953 | 0.8817 | 0.9647 | 0.5410 |
| 10 | 90 | 64,740 | 0.8123 | 0.9951 | 0.7954 | 0.8841 | 0.9651 | 0.3438 |
| 0 | 100 | 58,270 | 0.7953 | 1.0000 | 0.7953 | 0.8860 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9653 | 0.0000 | 0.0000 | 0.0000 | 0.9653 | 1.0000 |
| 90 | 10 | 299,940 | 0.9482 | 0.7178 | 0.7936 | 0.7538 | 0.9653 | 0.9768 |
| 80 | 20 | 291,350 | 0.9311 | 0.8510 | 0.7946 | 0.8219 | 0.9652 | 0.9495 |
| 70 | 30 | 194,230 | 0.9144 | 0.9085 | 0.7946 | 0.8478 | 0.9657 | 0.9165 |
| 60 | 40 | 145,675 | 0.8969 | 0.9381 | 0.7946 | 0.8604 | 0.9650 | 0.8758 |
| 50 | 50 | 116,540 | 0.8798 | 0.9577 | 0.7946 | 0.8686 | 0.9649 | 0.8245 |
| 40 | 60 | 97,115 | 0.8632 | 0.9723 | 0.7946 | 0.8745 | 0.9660 | 0.7582 |
| 30 | 70 | 83,240 | 0.8457 | 0.9813 | 0.7946 | 0.8782 | 0.9647 | 0.6681 |
| 20 | 80 | 72,835 | 0.8287 | 0.9890 | 0.7946 | 0.8812 | 0.9647 | 0.5401 |
| 10 | 90 | 64,740 | 0.8117 | 0.9952 | 0.7946 | 0.8837 | 0.9654 | 0.3431 |
| 0 | 100 | 58,270 | 0.7946 | 1.0000 | 0.7946 | 0.8856 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9653 | 0.0000 | 0.0000 | 0.0000 | 0.9653 | 1.0000 |
| 90 | 10 | 299,940 | 0.9482 | 0.7178 | 0.7936 | 0.7538 | 0.9653 | 0.9768 |
| 80 | 20 | 291,350 | 0.9311 | 0.8510 | 0.7946 | 0.8219 | 0.9652 | 0.9495 |
| 70 | 30 | 194,230 | 0.9144 | 0.9085 | 0.7946 | 0.8478 | 0.9657 | 0.9165 |
| 60 | 40 | 145,675 | 0.8969 | 0.9381 | 0.7946 | 0.8604 | 0.9650 | 0.8758 |
| 50 | 50 | 116,540 | 0.8798 | 0.9577 | 0.7946 | 0.8686 | 0.9649 | 0.8245 |
| 40 | 60 | 97,115 | 0.8632 | 0.9723 | 0.7946 | 0.8745 | 0.9660 | 0.7582 |
| 30 | 70 | 83,240 | 0.8457 | 0.9813 | 0.7946 | 0.8782 | 0.9647 | 0.6681 |
| 20 | 80 | 72,835 | 0.8287 | 0.9890 | 0.7946 | 0.8812 | 0.9647 | 0.5401 |
| 10 | 90 | 64,740 | 0.8117 | 0.9952 | 0.7946 | 0.8837 | 0.9654 | 0.3431 |
| 0 | 100 | 58,270 | 0.7946 | 1.0000 | 0.7946 | 0.8856 | 0.0000 | 0.0000 |


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
0.15       0.7670   0.3444   0.7842   0.9479   0.6121   0.2396   <--
0.20       0.7670   0.3444   0.7842   0.9479   0.6121   0.2396  
0.25       0.7670   0.3444   0.7842   0.9479   0.6121   0.2396  
0.30       0.7670   0.3444   0.7842   0.9479   0.6121   0.2396  
0.35       0.7670   0.3444   0.7842   0.9479   0.6121   0.2396  
0.40       0.7670   0.3444   0.7842   0.9479   0.6121   0.2396  
0.45       0.7670   0.3444   0.7842   0.9479   0.6121   0.2396  
0.50       0.7670   0.3444   0.7842   0.9479   0.6121   0.2396  
0.55       0.7670   0.3444   0.7842   0.9479   0.6121   0.2396  
0.60       0.7670   0.3444   0.7842   0.9479   0.6121   0.2396  
0.65       0.7670   0.3444   0.7842   0.9479   0.6121   0.2396  
0.70       0.7670   0.3444   0.7842   0.9479   0.6121   0.2396  
0.75       0.7670   0.3444   0.7842   0.9479   0.6121   0.2396  
0.80       0.7670   0.3444   0.7842   0.9479   0.6121   0.2396  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7670, F1=0.3444, Normal Recall=0.7842, Normal Precision=0.9479, Attack Recall=0.6121, Attack Precision=0.2396

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
0.15       0.7499   0.4951   0.7841   0.8902   0.6131   0.4151   <--
0.20       0.7499   0.4951   0.7841   0.8902   0.6131   0.4151  
0.25       0.7499   0.4951   0.7841   0.8902   0.6131   0.4151  
0.30       0.7499   0.4951   0.7841   0.8902   0.6131   0.4151  
0.35       0.7499   0.4951   0.7841   0.8902   0.6131   0.4151  
0.40       0.7499   0.4951   0.7841   0.8902   0.6131   0.4151  
0.45       0.7499   0.4951   0.7841   0.8902   0.6131   0.4151  
0.50       0.7499   0.4951   0.7841   0.8902   0.6131   0.4151  
0.55       0.7499   0.4951   0.7841   0.8902   0.6131   0.4151  
0.60       0.7499   0.4951   0.7841   0.8902   0.6131   0.4151  
0.65       0.7499   0.4951   0.7841   0.8902   0.6131   0.4151  
0.70       0.7499   0.4951   0.7841   0.8902   0.6131   0.4151  
0.75       0.7499   0.4951   0.7841   0.8902   0.6131   0.4151  
0.80       0.7499   0.4951   0.7841   0.8902   0.6131   0.4151  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7499, F1=0.4951, Normal Recall=0.7841, Normal Precision=0.8902, Attack Recall=0.6131, Attack Precision=0.4151

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
0.15       0.7341   0.5804   0.7860   0.8258   0.6131   0.5511   <--
0.20       0.7341   0.5804   0.7860   0.8258   0.6131   0.5511  
0.25       0.7341   0.5804   0.7860   0.8258   0.6131   0.5511  
0.30       0.7341   0.5804   0.7860   0.8258   0.6131   0.5511  
0.35       0.7341   0.5804   0.7860   0.8258   0.6131   0.5511  
0.40       0.7341   0.5804   0.7860   0.8258   0.6131   0.5511  
0.45       0.7341   0.5804   0.7860   0.8258   0.6131   0.5511  
0.50       0.7341   0.5804   0.7860   0.8258   0.6131   0.5511  
0.55       0.7341   0.5804   0.7860   0.8258   0.6131   0.5511  
0.60       0.7341   0.5804   0.7860   0.8258   0.6131   0.5511  
0.65       0.7341   0.5804   0.7860   0.8258   0.6131   0.5511  
0.70       0.7341   0.5804   0.7860   0.8258   0.6131   0.5511  
0.75       0.7341   0.5804   0.7860   0.8258   0.6131   0.5511  
0.80       0.7341   0.5804   0.7860   0.8258   0.6131   0.5511  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7341, F1=0.5804, Normal Recall=0.7860, Normal Precision=0.8258, Attack Recall=0.6131, Attack Precision=0.5511

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
0.15       0.7170   0.6341   0.7863   0.7530   0.6131   0.6567   <--
0.20       0.7170   0.6341   0.7863   0.7530   0.6131   0.6567  
0.25       0.7170   0.6341   0.7863   0.7530   0.6131   0.6567  
0.30       0.7170   0.6341   0.7863   0.7530   0.6131   0.6567  
0.35       0.7170   0.6341   0.7863   0.7530   0.6131   0.6567  
0.40       0.7170   0.6341   0.7863   0.7530   0.6131   0.6567  
0.45       0.7170   0.6341   0.7863   0.7530   0.6131   0.6567  
0.50       0.7170   0.6341   0.7863   0.7530   0.6131   0.6567  
0.55       0.7170   0.6341   0.7863   0.7530   0.6131   0.6567  
0.60       0.7170   0.6341   0.7863   0.7530   0.6131   0.6567  
0.65       0.7170   0.6341   0.7863   0.7530   0.6131   0.6567  
0.70       0.7170   0.6341   0.7863   0.7530   0.6131   0.6567  
0.75       0.7170   0.6341   0.7863   0.7530   0.6131   0.6567  
0.80       0.7170   0.6341   0.7863   0.7530   0.6131   0.6567  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7170, F1=0.6341, Normal Recall=0.7863, Normal Precision=0.7530, Attack Recall=0.6131, Attack Precision=0.6567

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
0.15       0.7000   0.6714   0.7869   0.6704   0.6131   0.7420   <--
0.20       0.7000   0.6714   0.7869   0.6704   0.6131   0.7420  
0.25       0.7000   0.6714   0.7869   0.6704   0.6131   0.7420  
0.30       0.7000   0.6714   0.7869   0.6704   0.6131   0.7420  
0.35       0.7000   0.6714   0.7869   0.6704   0.6131   0.7420  
0.40       0.7000   0.6714   0.7869   0.6704   0.6131   0.7420  
0.45       0.7000   0.6714   0.7869   0.6704   0.6131   0.7420  
0.50       0.7000   0.6714   0.7869   0.6704   0.6131   0.7420  
0.55       0.7000   0.6714   0.7869   0.6704   0.6131   0.7420  
0.60       0.7000   0.6714   0.7869   0.6704   0.6131   0.7420  
0.65       0.7000   0.6714   0.7869   0.6704   0.6131   0.7420  
0.70       0.7000   0.6714   0.7869   0.6704   0.6131   0.7420  
0.75       0.7000   0.6714   0.7869   0.6704   0.6131   0.7420  
0.80       0.7000   0.6714   0.7869   0.6704   0.6131   0.7420  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7000, F1=0.6714, Normal Recall=0.7869, Normal Precision=0.6704, Attack Recall=0.6131, Attack Precision=0.7420

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
0.15       0.9486   0.7565   0.9652   0.9774   0.7989   0.7184   <--
0.20       0.9486   0.7565   0.9652   0.9774   0.7989   0.7184  
0.25       0.9486   0.7565   0.9652   0.9774   0.7989   0.7184  
0.30       0.9486   0.7565   0.9652   0.9774   0.7989   0.7184  
0.35       0.9486   0.7565   0.9652   0.9774   0.7989   0.7184  
0.40       0.9486   0.7565   0.9652   0.9774   0.7989   0.7184  
0.45       0.9486   0.7565   0.9652   0.9774   0.7989   0.7184  
0.50       0.9486   0.7565   0.9652   0.9774   0.7989   0.7184  
0.55       0.9486   0.7565   0.9652   0.9774   0.7989   0.7184  
0.60       0.9486   0.7565   0.9652   0.9774   0.7989   0.7184  
0.65       0.9486   0.7565   0.9652   0.9774   0.7989   0.7184  
0.70       0.9486   0.7565   0.9652   0.9774   0.7989   0.7184  
0.75       0.9486   0.7565   0.9652   0.9774   0.7989   0.7184  
0.80       0.9486   0.7565   0.9652   0.9774   0.7989   0.7184  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9486, F1=0.7565, Normal Recall=0.9652, Normal Precision=0.9774, Attack Recall=0.7989, Attack Precision=0.7184

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
0.15       0.9314   0.8225   0.9654   0.9497   0.7953   0.8516   <--
0.20       0.9314   0.8225   0.9654   0.9497   0.7953   0.8516  
0.25       0.9314   0.8225   0.9654   0.9497   0.7953   0.8516  
0.30       0.9314   0.8225   0.9654   0.9497   0.7953   0.8516  
0.35       0.9314   0.8225   0.9654   0.9497   0.7953   0.8516  
0.40       0.9314   0.8225   0.9654   0.9497   0.7953   0.8516  
0.45       0.9314   0.8225   0.9654   0.9497   0.7953   0.8516  
0.50       0.9314   0.8225   0.9654   0.9497   0.7953   0.8516  
0.55       0.9314   0.8225   0.9654   0.9497   0.7953   0.8516  
0.60       0.9314   0.8225   0.9654   0.9497   0.7953   0.8516  
0.65       0.9314   0.8225   0.9654   0.9497   0.7953   0.8516  
0.70       0.9314   0.8225   0.9654   0.9497   0.7953   0.8516  
0.75       0.9314   0.8225   0.9654   0.9497   0.7953   0.8516  
0.80       0.9314   0.8225   0.9654   0.9497   0.7953   0.8516  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9314, F1=0.8225, Normal Recall=0.9654, Normal Precision=0.9497, Attack Recall=0.7953, Attack Precision=0.8516

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
0.15       0.9143   0.8477   0.9653   0.9167   0.7953   0.9075   <--
0.20       0.9143   0.8477   0.9653   0.9167   0.7953   0.9075  
0.25       0.9143   0.8477   0.9653   0.9167   0.7953   0.9075  
0.30       0.9143   0.8477   0.9653   0.9167   0.7953   0.9075  
0.35       0.9143   0.8477   0.9653   0.9167   0.7953   0.9075  
0.40       0.9143   0.8477   0.9653   0.9167   0.7953   0.9075  
0.45       0.9143   0.8477   0.9653   0.9167   0.7953   0.9075  
0.50       0.9143   0.8477   0.9653   0.9167   0.7953   0.9075  
0.55       0.9143   0.8477   0.9653   0.9167   0.7953   0.9075  
0.60       0.9143   0.8477   0.9653   0.9167   0.7953   0.9075  
0.65       0.9143   0.8477   0.9653   0.9167   0.7953   0.9075  
0.70       0.9143   0.8477   0.9653   0.9167   0.7953   0.9075  
0.75       0.9143   0.8477   0.9653   0.9167   0.7953   0.9075  
0.80       0.9143   0.8477   0.9653   0.9167   0.7953   0.9075  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9143, F1=0.8477, Normal Recall=0.9653, Normal Precision=0.9167, Attack Recall=0.7953, Attack Precision=0.9075

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
0.15       0.8974   0.8611   0.9654   0.8762   0.7953   0.9387   <--
0.20       0.8974   0.8611   0.9654   0.8762   0.7953   0.9387  
0.25       0.8974   0.8611   0.9654   0.8762   0.7953   0.9387  
0.30       0.8974   0.8611   0.9654   0.8762   0.7953   0.9387  
0.35       0.8974   0.8611   0.9654   0.8762   0.7953   0.9387  
0.40       0.8974   0.8611   0.9654   0.8762   0.7953   0.9387  
0.45       0.8974   0.8611   0.9654   0.8762   0.7953   0.9387  
0.50       0.8974   0.8611   0.9654   0.8762   0.7953   0.9387  
0.55       0.8974   0.8611   0.9654   0.8762   0.7953   0.9387  
0.60       0.8974   0.8611   0.9654   0.8762   0.7953   0.9387  
0.65       0.8974   0.8611   0.9654   0.8762   0.7953   0.9387  
0.70       0.8974   0.8611   0.9654   0.8762   0.7953   0.9387  
0.75       0.8974   0.8611   0.9654   0.8762   0.7953   0.9387  
0.80       0.8974   0.8611   0.9654   0.8762   0.7953   0.9387  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8974, F1=0.8611, Normal Recall=0.9654, Normal Precision=0.8762, Attack Recall=0.7953, Attack Precision=0.9387

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
0.15       0.8804   0.8693   0.9655   0.8251   0.7953   0.9585   <--
0.20       0.8804   0.8693   0.9655   0.8251   0.7953   0.9585  
0.25       0.8804   0.8693   0.9655   0.8251   0.7953   0.9585  
0.30       0.8804   0.8693   0.9655   0.8251   0.7953   0.9585  
0.35       0.8804   0.8693   0.9655   0.8251   0.7953   0.9585  
0.40       0.8804   0.8693   0.9655   0.8251   0.7953   0.9585  
0.45       0.8804   0.8693   0.9655   0.8251   0.7953   0.9585  
0.50       0.8804   0.8693   0.9655   0.8251   0.7953   0.9585  
0.55       0.8804   0.8693   0.9655   0.8251   0.7953   0.9585  
0.60       0.8804   0.8693   0.9655   0.8251   0.7953   0.9585  
0.65       0.8804   0.8693   0.9655   0.8251   0.7953   0.9585  
0.70       0.8804   0.8693   0.9655   0.8251   0.7953   0.9585  
0.75       0.8804   0.8693   0.9655   0.8251   0.7953   0.9585  
0.80       0.8804   0.8693   0.9655   0.8251   0.7953   0.9585  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8804, F1=0.8693, Normal Recall=0.9655, Normal Precision=0.8251, Attack Recall=0.7953, Attack Precision=0.9585

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
0.15       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189   <--
0.20       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.25       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.30       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.35       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.40       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.45       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.50       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.55       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.60       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.65       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.70       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.75       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.80       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9486, F1=0.7564, Normal Recall=0.9653, Normal Precision=0.9773, Attack Recall=0.7981, Attack Precision=0.7189

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
0.15       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519   <--
0.20       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.25       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.30       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.35       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.40       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.45       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.50       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.55       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.60       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.65       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.70       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.75       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.80       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9313, F1=0.8223, Normal Recall=0.9655, Normal Precision=0.9495, Attack Recall=0.7946, Attack Precision=0.8519

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
0.15       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079   <--
0.20       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.25       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.30       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.35       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.40       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.45       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.50       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.55       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.60       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.65       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.70       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.75       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.80       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9142, F1=0.8475, Normal Recall=0.9654, Normal Precision=0.9165, Attack Recall=0.7946, Attack Precision=0.9079

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
0.15       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391   <--
0.20       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.25       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.30       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.35       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.40       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.45       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.50       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.55       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.60       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.65       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.70       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.75       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.80       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8972, F1=0.8609, Normal Recall=0.9657, Normal Precision=0.8758, Attack Recall=0.7946, Attack Precision=0.9391

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
0.15       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588   <--
0.20       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.25       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.30       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.35       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.40       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.45       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.50       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.55       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.60       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.65       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.70       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.75       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.80       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8802, F1=0.8690, Normal Recall=0.9659, Normal Precision=0.8247, Attack Recall=0.7946, Attack Precision=0.9588

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
0.15       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189   <--
0.20       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.25       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.30       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.35       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.40       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.45       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.50       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.55       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.60       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.65       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.70       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.75       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
0.80       0.9486   0.7564   0.9653   0.9773   0.7981   0.7189  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9486, F1=0.7564, Normal Recall=0.9653, Normal Precision=0.9773, Attack Recall=0.7981, Attack Precision=0.7189

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
0.15       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519   <--
0.20       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.25       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.30       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.35       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.40       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.45       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.50       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.55       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.60       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.65       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.70       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.75       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
0.80       0.9313   0.8223   0.9655   0.9495   0.7946   0.8519  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9313, F1=0.8223, Normal Recall=0.9655, Normal Precision=0.9495, Attack Recall=0.7946, Attack Precision=0.8519

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
0.15       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079   <--
0.20       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.25       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.30       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.35       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.40       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.45       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.50       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.55       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.60       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.65       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.70       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.75       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
0.80       0.9142   0.8475   0.9654   0.9165   0.7946   0.9079  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9142, F1=0.8475, Normal Recall=0.9654, Normal Precision=0.9165, Attack Recall=0.7946, Attack Precision=0.9079

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
0.15       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391   <--
0.20       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.25       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.30       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.35       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.40       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.45       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.50       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.55       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.60       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.65       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.70       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.75       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
0.80       0.8972   0.8609   0.9657   0.8758   0.7946   0.9391  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8972, F1=0.8609, Normal Recall=0.9657, Normal Precision=0.8758, Attack Recall=0.7946, Attack Precision=0.9391

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
0.15       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588   <--
0.20       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.25       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.30       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.35       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.40       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.45       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.50       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.55       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.60       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.65       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.70       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.75       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
0.80       0.8802   0.8690   0.9659   0.8247   0.7946   0.9588  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8802, F1=0.8690, Normal Recall=0.9659, Normal Precision=0.8247, Attack Recall=0.7946, Attack Precision=0.9588

```

