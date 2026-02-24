# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e1_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-16 09:21:31 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3234 | 0.3907 | 0.4575 | 0.5256 | 0.5926 | 0.6593 | 0.7283 | 0.7950 | 0.8614 | 0.9304 | 0.9967 |
| QAT+Prune only | 0.6959 | 0.7273 | 0.7571 | 0.7878 | 0.8179 | 0.8476 | 0.8780 | 0.9090 | 0.9389 | 0.9690 | 0.9997 |
| QAT+PTQ | 0.6957 | 0.7271 | 0.7569 | 0.7878 | 0.8177 | 0.8475 | 0.8780 | 0.9092 | 0.9389 | 0.9690 | 0.9997 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6957 | 0.7271 | 0.7569 | 0.7878 | 0.8177 | 0.8475 | 0.8780 | 0.9092 | 0.9389 | 0.9690 | 0.9997 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2465 | 0.4236 | 0.5576 | 0.6618 | 0.7453 | 0.8149 | 0.8719 | 0.9201 | 0.9627 | 0.9984 |
| QAT+Prune only | 0.0000 | 0.4230 | 0.6221 | 0.7387 | 0.8146 | 0.8677 | 0.9077 | 0.9390 | 0.9632 | 0.9831 | 0.9999 |
| QAT+PTQ | 0.0000 | 0.4229 | 0.6220 | 0.7386 | 0.8144 | 0.8676 | 0.9077 | 0.9391 | 0.9632 | 0.9831 | 0.9999 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4229 | 0.6220 | 0.7386 | 0.8144 | 0.8676 | 0.9077 | 0.9391 | 0.9632 | 0.9831 | 0.9999 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3234 | 0.3233 | 0.3227 | 0.3236 | 0.3231 | 0.3219 | 0.3257 | 0.3244 | 0.3202 | 0.3335 | 0.0000 |
| QAT+Prune only | 0.6959 | 0.6970 | 0.6965 | 0.6970 | 0.6967 | 0.6954 | 0.6954 | 0.6974 | 0.6955 | 0.6926 | 0.0000 |
| QAT+PTQ | 0.6957 | 0.6968 | 0.6962 | 0.6969 | 0.6964 | 0.6952 | 0.6954 | 0.6978 | 0.6953 | 0.6928 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6957 | 0.6968 | 0.6962 | 0.6969 | 0.6964 | 0.6952 | 0.6954 | 0.6978 | 0.6953 | 0.6928 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3234 | 0.0000 | 0.0000 | 0.0000 | 0.3234 | 1.0000 |
| 90 | 10 | 299,940 | 0.3907 | 0.1406 | 0.9967 | 0.2465 | 0.3233 | 0.9989 |
| 80 | 20 | 291,350 | 0.4575 | 0.2690 | 0.9967 | 0.4236 | 0.3227 | 0.9975 |
| 70 | 30 | 194,230 | 0.5256 | 0.3871 | 0.9967 | 0.5576 | 0.3236 | 0.9957 |
| 60 | 40 | 145,675 | 0.5926 | 0.4954 | 0.9967 | 0.6618 | 0.3231 | 0.9933 |
| 50 | 50 | 116,540 | 0.6593 | 0.5951 | 0.9967 | 0.7453 | 0.3219 | 0.9900 |
| 40 | 60 | 97,115 | 0.7283 | 0.6892 | 0.9967 | 0.8149 | 0.3257 | 0.9852 |
| 30 | 70 | 83,240 | 0.7950 | 0.7749 | 0.9967 | 0.8719 | 0.3244 | 0.9771 |
| 20 | 80 | 72,835 | 0.8614 | 0.8543 | 0.9967 | 0.9201 | 0.3202 | 0.9609 |
| 10 | 90 | 64,740 | 0.9304 | 0.9308 | 0.9967 | 0.9627 | 0.3335 | 0.9191 |
| 0 | 100 | 58,270 | 0.9967 | 1.0000 | 0.9967 | 0.9984 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6959 | 0.0000 | 0.0000 | 0.0000 | 0.6959 | 1.0000 |
| 90 | 10 | 299,940 | 0.7273 | 0.2683 | 0.9997 | 0.4230 | 0.6970 | 1.0000 |
| 80 | 20 | 291,350 | 0.7571 | 0.4516 | 0.9997 | 0.6221 | 0.6965 | 0.9999 |
| 70 | 30 | 194,230 | 0.7878 | 0.5858 | 0.9997 | 0.7387 | 0.6970 | 0.9998 |
| 60 | 40 | 145,675 | 0.8179 | 0.6873 | 0.9997 | 0.8146 | 0.6967 | 0.9998 |
| 50 | 50 | 116,540 | 0.8476 | 0.7665 | 0.9997 | 0.8677 | 0.6954 | 0.9996 |
| 40 | 60 | 97,115 | 0.8780 | 0.8312 | 0.9997 | 0.9077 | 0.6954 | 0.9994 |
| 30 | 70 | 83,240 | 0.9090 | 0.8852 | 0.9997 | 0.9390 | 0.6974 | 0.9991 |
| 20 | 80 | 72,835 | 0.9389 | 0.9293 | 0.9997 | 0.9632 | 0.6955 | 0.9985 |
| 10 | 90 | 64,740 | 0.9690 | 0.9670 | 0.9997 | 0.9831 | 0.6926 | 0.9967 |
| 0 | 100 | 58,270 | 0.9997 | 1.0000 | 0.9997 | 0.9999 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6957 | 0.0000 | 0.0000 | 0.0000 | 0.6957 | 1.0000 |
| 90 | 10 | 299,940 | 0.7271 | 0.2681 | 0.9997 | 0.4229 | 0.6968 | 1.0000 |
| 80 | 20 | 291,350 | 0.7569 | 0.4514 | 0.9997 | 0.6220 | 0.6962 | 0.9999 |
| 70 | 30 | 194,230 | 0.7878 | 0.5857 | 0.9997 | 0.7386 | 0.6969 | 0.9998 |
| 60 | 40 | 145,675 | 0.8177 | 0.6870 | 0.9997 | 0.8144 | 0.6964 | 0.9998 |
| 50 | 50 | 116,540 | 0.8475 | 0.7664 | 0.9997 | 0.8676 | 0.6952 | 0.9996 |
| 40 | 60 | 97,115 | 0.8780 | 0.8312 | 0.9997 | 0.9077 | 0.6954 | 0.9994 |
| 30 | 70 | 83,240 | 0.9092 | 0.8853 | 0.9997 | 0.9391 | 0.6978 | 0.9991 |
| 20 | 80 | 72,835 | 0.9389 | 0.9292 | 0.9997 | 0.9632 | 0.6953 | 0.9985 |
| 10 | 90 | 64,740 | 0.9690 | 0.9670 | 0.9997 | 0.9831 | 0.6928 | 0.9967 |
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
| 100 | 0 | 100,000 | 0.6957 | 0.0000 | 0.0000 | 0.0000 | 0.6957 | 1.0000 |
| 90 | 10 | 299,940 | 0.7271 | 0.2681 | 0.9997 | 0.4229 | 0.6968 | 1.0000 |
| 80 | 20 | 291,350 | 0.7569 | 0.4514 | 0.9997 | 0.6220 | 0.6962 | 0.9999 |
| 70 | 30 | 194,230 | 0.7878 | 0.5857 | 0.9997 | 0.7386 | 0.6969 | 0.9998 |
| 60 | 40 | 145,675 | 0.8177 | 0.6870 | 0.9997 | 0.8144 | 0.6964 | 0.9998 |
| 50 | 50 | 116,540 | 0.8475 | 0.7664 | 0.9997 | 0.8676 | 0.6952 | 0.9996 |
| 40 | 60 | 97,115 | 0.8780 | 0.8312 | 0.9997 | 0.9077 | 0.6954 | 0.9994 |
| 30 | 70 | 83,240 | 0.9092 | 0.8853 | 0.9997 | 0.9391 | 0.6978 | 0.9991 |
| 20 | 80 | 72,835 | 0.9389 | 0.9292 | 0.9997 | 0.9632 | 0.6953 | 0.9985 |
| 10 | 90 | 64,740 | 0.9690 | 0.9670 | 0.9997 | 0.9831 | 0.6928 | 0.9967 |
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
0.15       0.3907   0.2465   0.3233   0.9989   0.9969   0.1407   <--
0.20       0.3907   0.2465   0.3233   0.9989   0.9969   0.1407  
0.25       0.3907   0.2465   0.3233   0.9989   0.9969   0.1407  
0.30       0.3907   0.2465   0.3233   0.9989   0.9969   0.1407  
0.35       0.3907   0.2465   0.3233   0.9989   0.9969   0.1407  
0.40       0.3907   0.2465   0.3233   0.9989   0.9969   0.1407  
0.45       0.3907   0.2465   0.3233   0.9989   0.9969   0.1407  
0.50       0.3907   0.2465   0.3233   0.9989   0.9969   0.1407  
0.55       0.3907   0.2465   0.3233   0.9989   0.9969   0.1407  
0.60       0.3907   0.2465   0.3233   0.9989   0.9969   0.1407  
0.65       0.3907   0.2465   0.3233   0.9989   0.9969   0.1407  
0.70       0.3907   0.2465   0.3233   0.9989   0.9969   0.1407  
0.75       0.3907   0.2465   0.3233   0.9989   0.9969   0.1407  
0.80       0.3907   0.2465   0.3233   0.9989   0.9969   0.1407  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3907, F1=0.2465, Normal Recall=0.3233, Normal Precision=0.9989, Attack Recall=0.9969, Attack Precision=0.1407

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
0.15       0.4580   0.4238   0.3233   0.9975   0.9967   0.2691   <--
0.20       0.4580   0.4238   0.3233   0.9975   0.9967   0.2691  
0.25       0.4580   0.4238   0.3233   0.9975   0.9967   0.2691  
0.30       0.4580   0.4238   0.3233   0.9975   0.9967   0.2691  
0.35       0.4580   0.4238   0.3233   0.9975   0.9967   0.2691  
0.40       0.4580   0.4238   0.3233   0.9975   0.9967   0.2691  
0.45       0.4580   0.4238   0.3233   0.9975   0.9967   0.2691  
0.50       0.4580   0.4238   0.3233   0.9975   0.9967   0.2691  
0.55       0.4580   0.4238   0.3233   0.9975   0.9967   0.2691  
0.60       0.4580   0.4238   0.3233   0.9975   0.9967   0.2691  
0.65       0.4580   0.4238   0.3233   0.9975   0.9967   0.2691  
0.70       0.4580   0.4238   0.3233   0.9975   0.9967   0.2691  
0.75       0.4580   0.4238   0.3233   0.9975   0.9967   0.2691  
0.80       0.4580   0.4238   0.3233   0.9975   0.9967   0.2691  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4580, F1=0.4238, Normal Recall=0.3233, Normal Precision=0.9975, Attack Recall=0.9967, Attack Precision=0.2691

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
0.15       0.5263   0.5580   0.3247   0.9957   0.9967   0.3875   <--
0.20       0.5263   0.5580   0.3247   0.9957   0.9967   0.3875  
0.25       0.5263   0.5580   0.3247   0.9957   0.9967   0.3875  
0.30       0.5263   0.5580   0.3247   0.9957   0.9967   0.3875  
0.35       0.5263   0.5580   0.3247   0.9957   0.9967   0.3875  
0.40       0.5263   0.5580   0.3247   0.9957   0.9967   0.3875  
0.45       0.5263   0.5580   0.3247   0.9957   0.9967   0.3875  
0.50       0.5263   0.5580   0.3247   0.9957   0.9967   0.3875  
0.55       0.5263   0.5580   0.3247   0.9957   0.9967   0.3875  
0.60       0.5263   0.5580   0.3247   0.9957   0.9967   0.3875  
0.65       0.5263   0.5580   0.3247   0.9957   0.9967   0.3875  
0.70       0.5263   0.5580   0.3247   0.9957   0.9967   0.3875  
0.75       0.5263   0.5580   0.3247   0.9957   0.9967   0.3875  
0.80       0.5263   0.5580   0.3247   0.9957   0.9967   0.3875  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5263, F1=0.5580, Normal Recall=0.3247, Normal Precision=0.9957, Attack Recall=0.9967, Attack Precision=0.3875

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
0.15       0.5927   0.6619   0.3234   0.9933   0.9967   0.4955   <--
0.20       0.5927   0.6619   0.3234   0.9933   0.9967   0.4955  
0.25       0.5927   0.6619   0.3234   0.9933   0.9967   0.4955  
0.30       0.5927   0.6619   0.3234   0.9933   0.9967   0.4955  
0.35       0.5927   0.6619   0.3234   0.9933   0.9967   0.4955  
0.40       0.5927   0.6619   0.3234   0.9933   0.9967   0.4955  
0.45       0.5927   0.6619   0.3234   0.9933   0.9967   0.4955  
0.50       0.5927   0.6619   0.3234   0.9933   0.9967   0.4955  
0.55       0.5927   0.6619   0.3234   0.9933   0.9967   0.4955  
0.60       0.5927   0.6619   0.3234   0.9933   0.9967   0.4955  
0.65       0.5927   0.6619   0.3234   0.9933   0.9967   0.4955  
0.70       0.5927   0.6619   0.3234   0.9933   0.9967   0.4955  
0.75       0.5927   0.6619   0.3234   0.9933   0.9967   0.4955  
0.80       0.5927   0.6619   0.3234   0.9933   0.9967   0.4955  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5927, F1=0.6619, Normal Recall=0.3234, Normal Precision=0.9933, Attack Recall=0.9967, Attack Precision=0.4955

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
0.15       0.6595   0.7454   0.3223   0.9900   0.9967   0.5953   <--
0.20       0.6595   0.7454   0.3223   0.9900   0.9967   0.5953  
0.25       0.6595   0.7454   0.3223   0.9900   0.9967   0.5953  
0.30       0.6595   0.7454   0.3223   0.9900   0.9967   0.5953  
0.35       0.6595   0.7454   0.3223   0.9900   0.9967   0.5953  
0.40       0.6595   0.7454   0.3223   0.9900   0.9967   0.5953  
0.45       0.6595   0.7454   0.3223   0.9900   0.9967   0.5953  
0.50       0.6595   0.7454   0.3223   0.9900   0.9967   0.5953  
0.55       0.6595   0.7454   0.3223   0.9900   0.9967   0.5953  
0.60       0.6595   0.7454   0.3223   0.9900   0.9967   0.5953  
0.65       0.6595   0.7454   0.3223   0.9900   0.9967   0.5953  
0.70       0.6595   0.7454   0.3223   0.9900   0.9967   0.5953  
0.75       0.6595   0.7454   0.3223   0.9900   0.9967   0.5953  
0.80       0.6595   0.7454   0.3223   0.9900   0.9967   0.5953  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6595, F1=0.7454, Normal Recall=0.3223, Normal Precision=0.9900, Attack Recall=0.9967, Attack Precision=0.5953

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
0.15       0.7273   0.4230   0.6970   1.0000   0.9998   0.2683   <--
0.20       0.7273   0.4230   0.6970   1.0000   0.9998   0.2683  
0.25       0.7273   0.4230   0.6970   1.0000   0.9998   0.2683  
0.30       0.7273   0.4230   0.6970   1.0000   0.9998   0.2683  
0.35       0.7273   0.4230   0.6970   1.0000   0.9998   0.2683  
0.40       0.7273   0.4230   0.6970   1.0000   0.9998   0.2683  
0.45       0.7273   0.4230   0.6970   1.0000   0.9998   0.2683  
0.50       0.7273   0.4230   0.6970   1.0000   0.9998   0.2683  
0.55       0.7273   0.4230   0.6970   1.0000   0.9998   0.2683  
0.60       0.7273   0.4230   0.6970   1.0000   0.9998   0.2683  
0.65       0.7273   0.4230   0.6970   1.0000   0.9998   0.2683  
0.70       0.7273   0.4230   0.6970   1.0000   0.9998   0.2683  
0.75       0.7273   0.4230   0.6970   1.0000   0.9998   0.2683  
0.80       0.7273   0.4230   0.6970   1.0000   0.9998   0.2683  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7273, F1=0.4230, Normal Recall=0.6970, Normal Precision=1.0000, Attack Recall=0.9998, Attack Precision=0.2683

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
0.15       0.7579   0.6229   0.6975   0.9999   0.9997   0.4524   <--
0.20       0.7579   0.6229   0.6975   0.9999   0.9997   0.4524  
0.25       0.7579   0.6229   0.6975   0.9999   0.9997   0.4524  
0.30       0.7579   0.6229   0.6975   0.9999   0.9997   0.4524  
0.35       0.7579   0.6229   0.6975   0.9999   0.9997   0.4524  
0.40       0.7579   0.6229   0.6975   0.9999   0.9997   0.4524  
0.45       0.7579   0.6229   0.6975   0.9999   0.9997   0.4524  
0.50       0.7579   0.6229   0.6975   0.9999   0.9997   0.4524  
0.55       0.7579   0.6229   0.6975   0.9999   0.9997   0.4524  
0.60       0.7579   0.6229   0.6975   0.9999   0.9997   0.4524  
0.65       0.7579   0.6229   0.6975   0.9999   0.9997   0.4524  
0.70       0.7579   0.6229   0.6975   0.9999   0.9997   0.4524  
0.75       0.7579   0.6229   0.6975   0.9999   0.9997   0.4524  
0.80       0.7579   0.6229   0.6975   0.9999   0.9997   0.4524  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7579, F1=0.6229, Normal Recall=0.6975, Normal Precision=0.9999, Attack Recall=0.9997, Attack Precision=0.4524

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
0.15       0.7875   0.7384   0.6965   0.9998   0.9997   0.5854   <--
0.20       0.7875   0.7384   0.6965   0.9998   0.9997   0.5854  
0.25       0.7875   0.7384   0.6965   0.9998   0.9997   0.5854  
0.30       0.7875   0.7384   0.6965   0.9998   0.9997   0.5854  
0.35       0.7875   0.7384   0.6965   0.9998   0.9997   0.5854  
0.40       0.7875   0.7384   0.6965   0.9998   0.9997   0.5854  
0.45       0.7875   0.7384   0.6965   0.9998   0.9997   0.5854  
0.50       0.7875   0.7384   0.6965   0.9998   0.9997   0.5854  
0.55       0.7875   0.7384   0.6965   0.9998   0.9997   0.5854  
0.60       0.7875   0.7384   0.6965   0.9998   0.9997   0.5854  
0.65       0.7875   0.7384   0.6965   0.9998   0.9997   0.5854  
0.70       0.7875   0.7384   0.6965   0.9998   0.9997   0.5854  
0.75       0.7875   0.7384   0.6965   0.9998   0.9997   0.5854  
0.80       0.7875   0.7384   0.6965   0.9998   0.9997   0.5854  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7875, F1=0.7384, Normal Recall=0.6965, Normal Precision=0.9998, Attack Recall=0.9997, Attack Precision=0.5854

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
0.15       0.8174   0.8142   0.6959   0.9998   0.9997   0.6867   <--
0.20       0.8174   0.8142   0.6959   0.9998   0.9997   0.6867  
0.25       0.8174   0.8142   0.6959   0.9998   0.9997   0.6867  
0.30       0.8174   0.8142   0.6959   0.9998   0.9997   0.6867  
0.35       0.8174   0.8142   0.6959   0.9998   0.9997   0.6867  
0.40       0.8174   0.8142   0.6959   0.9998   0.9997   0.6867  
0.45       0.8174   0.8142   0.6959   0.9998   0.9997   0.6867  
0.50       0.8174   0.8142   0.6959   0.9998   0.9997   0.6867  
0.55       0.8174   0.8142   0.6959   0.9998   0.9997   0.6867  
0.60       0.8174   0.8142   0.6959   0.9998   0.9997   0.6867  
0.65       0.8174   0.8142   0.6959   0.9998   0.9997   0.6867  
0.70       0.8174   0.8142   0.6959   0.9998   0.9997   0.6867  
0.75       0.8174   0.8142   0.6959   0.9998   0.9997   0.6867  
0.80       0.8174   0.8142   0.6959   0.9998   0.9997   0.6867  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8174, F1=0.8142, Normal Recall=0.6959, Normal Precision=0.9998, Attack Recall=0.9997, Attack Precision=0.6867

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
0.15       0.8465   0.8669   0.6933   0.9996   0.9997   0.7652   <--
0.20       0.8465   0.8669   0.6933   0.9996   0.9997   0.7652  
0.25       0.8465   0.8669   0.6933   0.9996   0.9997   0.7652  
0.30       0.8465   0.8669   0.6933   0.9996   0.9997   0.7652  
0.35       0.8465   0.8669   0.6933   0.9996   0.9997   0.7652  
0.40       0.8465   0.8669   0.6933   0.9996   0.9997   0.7652  
0.45       0.8465   0.8669   0.6933   0.9996   0.9997   0.7652  
0.50       0.8465   0.8669   0.6933   0.9996   0.9997   0.7652  
0.55       0.8465   0.8669   0.6933   0.9996   0.9997   0.7652  
0.60       0.8465   0.8669   0.6933   0.9996   0.9997   0.7652  
0.65       0.8465   0.8669   0.6933   0.9996   0.9997   0.7652  
0.70       0.8465   0.8669   0.6933   0.9996   0.9997   0.7652  
0.75       0.8465   0.8669   0.6933   0.9996   0.9997   0.7652  
0.80       0.8465   0.8669   0.6933   0.9996   0.9997   0.7652  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8465, F1=0.8669, Normal Recall=0.6933, Normal Precision=0.9996, Attack Recall=0.9997, Attack Precision=0.7652

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
0.15       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682   <--
0.20       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.25       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.30       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.35       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.40       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.45       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.50       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.55       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.60       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.65       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.70       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.75       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.80       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7271, F1=0.4229, Normal Recall=0.6968, Normal Precision=1.0000, Attack Recall=0.9998, Attack Precision=0.2682

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
0.15       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522   <--
0.20       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.25       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.30       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.35       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.40       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.45       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.50       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.55       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.60       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.65       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.70       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.75       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.80       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7577, F1=0.6227, Normal Recall=0.6972, Normal Precision=0.9999, Attack Recall=0.9997, Attack Precision=0.4522

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
0.15       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853   <--
0.20       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.25       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.30       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.35       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.40       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.45       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.50       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.55       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.60       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.65       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.70       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.75       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.80       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7874, F1=0.7383, Normal Recall=0.6964, Normal Precision=0.9998, Attack Recall=0.9997, Attack Precision=0.5853

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
0.15       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865   <--
0.20       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.25       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.30       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.35       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.40       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.45       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.50       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.55       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.60       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.65       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.70       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.75       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.80       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8173, F1=0.8140, Normal Recall=0.6956, Normal Precision=0.9998, Attack Recall=0.9997, Attack Precision=0.6865

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
0.15       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650   <--
0.20       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.25       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.30       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.35       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.40       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.45       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.50       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.55       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.60       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.65       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.70       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.75       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.80       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8463, F1=0.8668, Normal Recall=0.6929, Normal Precision=0.9996, Attack Recall=0.9997, Attack Precision=0.7650

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
0.15       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682   <--
0.20       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.25       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.30       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.35       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.40       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.45       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.50       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.55       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.60       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.65       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.70       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.75       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
0.80       0.7271   0.4229   0.6968   1.0000   0.9998   0.2682  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7271, F1=0.4229, Normal Recall=0.6968, Normal Precision=1.0000, Attack Recall=0.9998, Attack Precision=0.2682

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
0.15       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522   <--
0.20       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.25       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.30       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.35       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.40       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.45       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.50       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.55       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.60       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.65       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.70       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.75       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
0.80       0.7577   0.6227   0.6972   0.9999   0.9997   0.4522  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7577, F1=0.6227, Normal Recall=0.6972, Normal Precision=0.9999, Attack Recall=0.9997, Attack Precision=0.4522

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
0.15       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853   <--
0.20       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.25       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.30       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.35       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.40       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.45       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.50       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.55       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.60       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.65       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.70       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.75       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
0.80       0.7874   0.7383   0.6964   0.9998   0.9997   0.5853  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7874, F1=0.7383, Normal Recall=0.6964, Normal Precision=0.9998, Attack Recall=0.9997, Attack Precision=0.5853

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
0.15       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865   <--
0.20       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.25       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.30       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.35       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.40       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.45       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.50       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.55       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.60       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.65       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.70       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.75       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
0.80       0.8173   0.8140   0.6956   0.9998   0.9997   0.6865  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8173, F1=0.8140, Normal Recall=0.6956, Normal Precision=0.9998, Attack Recall=0.9997, Attack Precision=0.6865

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
0.15       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650   <--
0.20       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.25       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.30       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.35       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.40       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.45       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.50       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.55       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.60       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.65       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.70       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.75       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
0.80       0.8463   0.8668   0.6929   0.9996   0.9997   0.7650  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8463, F1=0.8668, Normal Recall=0.6929, Normal Precision=0.9996, Attack Recall=0.9997, Attack Precision=0.7650

```

