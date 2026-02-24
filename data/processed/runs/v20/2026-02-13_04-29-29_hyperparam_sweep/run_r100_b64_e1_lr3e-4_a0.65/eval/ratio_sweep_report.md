# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e1_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-19 09:23:26 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8484 | 0.8556 | 0.8633 | 0.8734 | 0.8806 | 0.8890 | 0.8971 | 0.9068 | 0.9143 | 0.9227 | 0.9312 |
| QAT+Prune only | 0.7332 | 0.7598 | 0.7858 | 0.8132 | 0.8393 | 0.8645 | 0.8915 | 0.9186 | 0.9450 | 0.9712 | 0.9977 |
| QAT+PTQ | 0.7288 | 0.7561 | 0.7825 | 0.8102 | 0.8368 | 0.8625 | 0.8898 | 0.9176 | 0.9441 | 0.9709 | 0.9977 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7288 | 0.7561 | 0.7825 | 0.8102 | 0.8368 | 0.8625 | 0.8898 | 0.9176 | 0.9441 | 0.9709 | 0.9977 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.5636 | 0.7316 | 0.8153 | 0.8619 | 0.8935 | 0.9157 | 0.9333 | 0.9456 | 0.9559 | 0.9644 |
| QAT+Prune only | 0.0000 | 0.4538 | 0.6508 | 0.7621 | 0.8324 | 0.8804 | 0.9169 | 0.9449 | 0.9667 | 0.9842 | 0.9988 |
| QAT+PTQ | 0.0000 | 0.4499 | 0.6472 | 0.7593 | 0.8303 | 0.8789 | 0.9157 | 0.9443 | 0.9661 | 0.9840 | 0.9988 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4499 | 0.6472 | 0.7593 | 0.8303 | 0.8789 | 0.9157 | 0.9443 | 0.9661 | 0.9840 | 0.9988 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8484 | 0.8471 | 0.8464 | 0.8486 | 0.8469 | 0.8467 | 0.8460 | 0.8497 | 0.8464 | 0.8457 | 0.0000 |
| QAT+Prune only | 0.7332 | 0.7334 | 0.7329 | 0.7341 | 0.7337 | 0.7313 | 0.7323 | 0.7340 | 0.7340 | 0.7331 | 0.0000 |
| QAT+PTQ | 0.7288 | 0.7292 | 0.7287 | 0.7299 | 0.7296 | 0.7273 | 0.7279 | 0.7308 | 0.7295 | 0.7292 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7288 | 0.7292 | 0.7287 | 0.7299 | 0.7296 | 0.7273 | 0.7279 | 0.7308 | 0.7295 | 0.7292 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8484 | 0.0000 | 0.0000 | 0.0000 | 0.8484 | 1.0000 |
| 90 | 10 | 299,940 | 0.8556 | 0.4039 | 0.9322 | 0.5636 | 0.8471 | 0.9912 |
| 80 | 20 | 291,350 | 0.8633 | 0.6024 | 0.9312 | 0.7316 | 0.8464 | 0.9801 |
| 70 | 30 | 194,230 | 0.8734 | 0.7250 | 0.9312 | 0.8153 | 0.8486 | 0.9664 |
| 60 | 40 | 145,675 | 0.8806 | 0.8022 | 0.9312 | 0.8619 | 0.8469 | 0.9486 |
| 50 | 50 | 116,540 | 0.8890 | 0.8587 | 0.9312 | 0.8935 | 0.8467 | 0.9249 |
| 40 | 60 | 97,115 | 0.8971 | 0.9007 | 0.9312 | 0.9157 | 0.8460 | 0.8913 |
| 30 | 70 | 83,240 | 0.9068 | 0.9353 | 0.9312 | 0.9333 | 0.8497 | 0.8411 |
| 20 | 80 | 72,835 | 0.9143 | 0.9604 | 0.9312 | 0.9456 | 0.8464 | 0.7547 |
| 10 | 90 | 64,740 | 0.9227 | 0.9819 | 0.9312 | 0.9559 | 0.8457 | 0.5773 |
| 0 | 100 | 58,270 | 0.9312 | 1.0000 | 0.9312 | 0.9644 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7332 | 0.0000 | 0.0000 | 0.0000 | 0.7332 | 1.0000 |
| 90 | 10 | 299,940 | 0.7598 | 0.2937 | 0.9977 | 0.4538 | 0.7334 | 0.9996 |
| 80 | 20 | 291,350 | 0.7858 | 0.4829 | 0.9977 | 0.6508 | 0.7329 | 0.9992 |
| 70 | 30 | 194,230 | 0.8132 | 0.6166 | 0.9977 | 0.7621 | 0.7341 | 0.9987 |
| 60 | 40 | 145,675 | 0.8393 | 0.7141 | 0.9977 | 0.8324 | 0.7337 | 0.9979 |
| 50 | 50 | 116,540 | 0.8645 | 0.7878 | 0.9977 | 0.8804 | 0.7313 | 0.9969 |
| 40 | 60 | 97,115 | 0.8915 | 0.8483 | 0.9977 | 0.9169 | 0.7323 | 0.9953 |
| 30 | 70 | 83,240 | 0.9186 | 0.8975 | 0.9977 | 0.9449 | 0.7340 | 0.9927 |
| 20 | 80 | 72,835 | 0.9450 | 0.9375 | 0.9977 | 0.9667 | 0.7340 | 0.9876 |
| 10 | 90 | 64,740 | 0.9712 | 0.9711 | 0.9977 | 0.9842 | 0.7331 | 0.9725 |
| 0 | 100 | 58,270 | 0.9977 | 1.0000 | 0.9977 | 0.9988 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7288 | 0.0000 | 0.0000 | 0.0000 | 0.7288 | 1.0000 |
| 90 | 10 | 299,940 | 0.7561 | 0.2905 | 0.9977 | 0.4499 | 0.7292 | 0.9996 |
| 80 | 20 | 291,350 | 0.7825 | 0.4790 | 0.9977 | 0.6472 | 0.7287 | 0.9992 |
| 70 | 30 | 194,230 | 0.8102 | 0.6129 | 0.9977 | 0.7593 | 0.7299 | 0.9987 |
| 60 | 40 | 145,675 | 0.8368 | 0.7109 | 0.9977 | 0.8303 | 0.7296 | 0.9979 |
| 50 | 50 | 116,540 | 0.8625 | 0.7854 | 0.9977 | 0.8789 | 0.7273 | 0.9968 |
| 40 | 60 | 97,115 | 0.8898 | 0.8462 | 0.9977 | 0.9157 | 0.7279 | 0.9953 |
| 30 | 70 | 83,240 | 0.9176 | 0.8963 | 0.9977 | 0.9443 | 0.7308 | 0.9927 |
| 20 | 80 | 72,835 | 0.9441 | 0.9365 | 0.9977 | 0.9661 | 0.7295 | 0.9875 |
| 10 | 90 | 64,740 | 0.9709 | 0.9707 | 0.9977 | 0.9840 | 0.7292 | 0.9724 |
| 0 | 100 | 58,270 | 0.9977 | 1.0000 | 0.9977 | 0.9988 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7288 | 0.0000 | 0.0000 | 0.0000 | 0.7288 | 1.0000 |
| 90 | 10 | 299,940 | 0.7561 | 0.2905 | 0.9977 | 0.4499 | 0.7292 | 0.9996 |
| 80 | 20 | 291,350 | 0.7825 | 0.4790 | 0.9977 | 0.6472 | 0.7287 | 0.9992 |
| 70 | 30 | 194,230 | 0.8102 | 0.6129 | 0.9977 | 0.7593 | 0.7299 | 0.9987 |
| 60 | 40 | 145,675 | 0.8368 | 0.7109 | 0.9977 | 0.8303 | 0.7296 | 0.9979 |
| 50 | 50 | 116,540 | 0.8625 | 0.7854 | 0.9977 | 0.8789 | 0.7273 | 0.9968 |
| 40 | 60 | 97,115 | 0.8898 | 0.8462 | 0.9977 | 0.9157 | 0.7279 | 0.9953 |
| 30 | 70 | 83,240 | 0.9176 | 0.8963 | 0.9977 | 0.9443 | 0.7308 | 0.9927 |
| 20 | 80 | 72,835 | 0.9441 | 0.9365 | 0.9977 | 0.9661 | 0.7295 | 0.9875 |
| 10 | 90 | 64,740 | 0.9709 | 0.9707 | 0.9977 | 0.9840 | 0.7292 | 0.9724 |
| 0 | 100 | 58,270 | 0.9977 | 1.0000 | 0.9977 | 0.9988 | 0.0000 | 0.0000 |


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
0.15       0.8555   0.5632   0.8471   0.9911   0.9314   0.4036   <--
0.20       0.8555   0.5632   0.8471   0.9911   0.9314   0.4036  
0.25       0.8555   0.5632   0.8471   0.9911   0.9314   0.4036  
0.30       0.8555   0.5632   0.8471   0.9911   0.9314   0.4036  
0.35       0.8555   0.5632   0.8471   0.9911   0.9314   0.4036  
0.40       0.8555   0.5632   0.8471   0.9911   0.9314   0.4036  
0.45       0.8555   0.5632   0.8471   0.9911   0.9314   0.4036  
0.50       0.8555   0.5632   0.8471   0.9911   0.9314   0.4036  
0.55       0.8555   0.5632   0.8471   0.9911   0.9314   0.4036  
0.60       0.8555   0.5632   0.8471   0.9911   0.9314   0.4036  
0.65       0.8555   0.5632   0.8471   0.9911   0.9314   0.4036  
0.70       0.8555   0.5632   0.8471   0.9911   0.9314   0.4036  
0.75       0.8555   0.5632   0.8471   0.9911   0.9314   0.4036  
0.80       0.8555   0.5632   0.8471   0.9911   0.9314   0.4036  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8555, F1=0.5632, Normal Recall=0.8471, Normal Precision=0.9911, Attack Recall=0.9314, Attack Precision=0.4036

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
0.15       0.8643   0.7329   0.8475   0.9801   0.9312   0.6042   <--
0.20       0.8643   0.7329   0.8475   0.9801   0.9312   0.6042  
0.25       0.8643   0.7329   0.8475   0.9801   0.9312   0.6042  
0.30       0.8643   0.7329   0.8475   0.9801   0.9312   0.6042  
0.35       0.8643   0.7329   0.8475   0.9801   0.9312   0.6042  
0.40       0.8643   0.7329   0.8475   0.9801   0.9312   0.6042  
0.45       0.8643   0.7329   0.8475   0.9801   0.9312   0.6042  
0.50       0.8643   0.7329   0.8475   0.9801   0.9312   0.6042  
0.55       0.8643   0.7329   0.8475   0.9801   0.9312   0.6042  
0.60       0.8643   0.7329   0.8475   0.9801   0.9312   0.6042  
0.65       0.8643   0.7329   0.8475   0.9801   0.9312   0.6042  
0.70       0.8643   0.7329   0.8475   0.9801   0.9312   0.6042  
0.75       0.8643   0.7329   0.8475   0.9801   0.9312   0.6042  
0.80       0.8643   0.7329   0.8475   0.9801   0.9312   0.6042  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8643, F1=0.7329, Normal Recall=0.8475, Normal Precision=0.9801, Attack Recall=0.9312, Attack Precision=0.6042

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
0.15       0.8733   0.8152   0.8485   0.9664   0.9312   0.7249   <--
0.20       0.8733   0.8152   0.8485   0.9664   0.9312   0.7249  
0.25       0.8733   0.8152   0.8485   0.9664   0.9312   0.7249  
0.30       0.8733   0.8152   0.8485   0.9664   0.9312   0.7249  
0.35       0.8733   0.8152   0.8485   0.9664   0.9312   0.7249  
0.40       0.8733   0.8152   0.8485   0.9664   0.9312   0.7249  
0.45       0.8733   0.8152   0.8485   0.9664   0.9312   0.7249  
0.50       0.8733   0.8152   0.8485   0.9664   0.9312   0.7249  
0.55       0.8733   0.8152   0.8485   0.9664   0.9312   0.7249  
0.60       0.8733   0.8152   0.8485   0.9664   0.9312   0.7249  
0.65       0.8733   0.8152   0.8485   0.9664   0.9312   0.7249  
0.70       0.8733   0.8152   0.8485   0.9664   0.9312   0.7249  
0.75       0.8733   0.8152   0.8485   0.9664   0.9312   0.7249  
0.80       0.8733   0.8152   0.8485   0.9664   0.9312   0.7249  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8733, F1=0.8152, Normal Recall=0.8485, Normal Precision=0.9664, Attack Recall=0.9312, Attack Precision=0.7249

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
0.15       0.8816   0.8628   0.8485   0.9487   0.9312   0.8038   <--
0.20       0.8816   0.8628   0.8485   0.9487   0.9312   0.8038  
0.25       0.8816   0.8628   0.8485   0.9487   0.9312   0.8038  
0.30       0.8816   0.8628   0.8485   0.9487   0.9312   0.8038  
0.35       0.8816   0.8628   0.8485   0.9487   0.9312   0.8038  
0.40       0.8816   0.8628   0.8485   0.9487   0.9312   0.8038  
0.45       0.8816   0.8628   0.8485   0.9487   0.9312   0.8038  
0.50       0.8816   0.8628   0.8485   0.9487   0.9312   0.8038  
0.55       0.8816   0.8628   0.8485   0.9487   0.9312   0.8038  
0.60       0.8816   0.8628   0.8485   0.9487   0.9312   0.8038  
0.65       0.8816   0.8628   0.8485   0.9487   0.9312   0.8038  
0.70       0.8816   0.8628   0.8485   0.9487   0.9312   0.8038  
0.75       0.8816   0.8628   0.8485   0.9487   0.9312   0.8038  
0.80       0.8816   0.8628   0.8485   0.9487   0.9312   0.8038  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8816, F1=0.8628, Normal Recall=0.8485, Normal Precision=0.9487, Attack Recall=0.9312, Attack Precision=0.8038

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
0.15       0.8896   0.8940   0.8481   0.9250   0.9312   0.8597   <--
0.20       0.8896   0.8940   0.8481   0.9250   0.9312   0.8597  
0.25       0.8896   0.8940   0.8481   0.9250   0.9312   0.8597  
0.30       0.8896   0.8940   0.8481   0.9250   0.9312   0.8597  
0.35       0.8896   0.8940   0.8481   0.9250   0.9312   0.8597  
0.40       0.8896   0.8940   0.8481   0.9250   0.9312   0.8597  
0.45       0.8896   0.8940   0.8481   0.9250   0.9312   0.8597  
0.50       0.8896   0.8940   0.8481   0.9250   0.9312   0.8597  
0.55       0.8896   0.8940   0.8481   0.9250   0.9312   0.8597  
0.60       0.8896   0.8940   0.8481   0.9250   0.9312   0.8597  
0.65       0.8896   0.8940   0.8481   0.9250   0.9312   0.8597  
0.70       0.8896   0.8940   0.8481   0.9250   0.9312   0.8597  
0.75       0.8896   0.8940   0.8481   0.9250   0.9312   0.8597  
0.80       0.8896   0.8940   0.8481   0.9250   0.9312   0.8597  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8896, F1=0.8940, Normal Recall=0.8481, Normal Precision=0.9250, Attack Recall=0.9312, Attack Precision=0.8597

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
0.15       0.7599   0.4539   0.7334   0.9997   0.9980   0.2937   <--
0.20       0.7599   0.4539   0.7334   0.9997   0.9980   0.2937  
0.25       0.7599   0.4539   0.7334   0.9997   0.9980   0.2937  
0.30       0.7599   0.4539   0.7334   0.9997   0.9980   0.2937  
0.35       0.7599   0.4539   0.7334   0.9997   0.9980   0.2937  
0.40       0.7599   0.4539   0.7334   0.9997   0.9980   0.2937  
0.45       0.7599   0.4539   0.7334   0.9997   0.9980   0.2937  
0.50       0.7599   0.4539   0.7334   0.9997   0.9980   0.2937  
0.55       0.7599   0.4539   0.7334   0.9997   0.9980   0.2937  
0.60       0.7599   0.4539   0.7334   0.9997   0.9980   0.2937  
0.65       0.7599   0.4539   0.7334   0.9997   0.9980   0.2937  
0.70       0.7599   0.4539   0.7334   0.9997   0.9980   0.2937  
0.75       0.7599   0.4539   0.7334   0.9997   0.9980   0.2937  
0.80       0.7599   0.4539   0.7334   0.9997   0.9980   0.2937  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7599, F1=0.4539, Normal Recall=0.7334, Normal Precision=0.9997, Attack Recall=0.9980, Attack Precision=0.2937

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
0.15       0.7866   0.6516   0.7339   0.9992   0.9977   0.4838   <--
0.20       0.7866   0.6516   0.7339   0.9992   0.9977   0.4838  
0.25       0.7866   0.6516   0.7339   0.9992   0.9977   0.4838  
0.30       0.7866   0.6516   0.7339   0.9992   0.9977   0.4838  
0.35       0.7866   0.6516   0.7339   0.9992   0.9977   0.4838  
0.40       0.7866   0.6516   0.7339   0.9992   0.9977   0.4838  
0.45       0.7866   0.6516   0.7339   0.9992   0.9977   0.4838  
0.50       0.7866   0.6516   0.7339   0.9992   0.9977   0.4838  
0.55       0.7866   0.6516   0.7339   0.9992   0.9977   0.4838  
0.60       0.7866   0.6516   0.7339   0.9992   0.9977   0.4838  
0.65       0.7866   0.6516   0.7339   0.9992   0.9977   0.4838  
0.70       0.7866   0.6516   0.7339   0.9992   0.9977   0.4838  
0.75       0.7866   0.6516   0.7339   0.9992   0.9977   0.4838  
0.80       0.7866   0.6516   0.7339   0.9992   0.9977   0.4838  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7866, F1=0.6516, Normal Recall=0.7339, Normal Precision=0.9992, Attack Recall=0.9977, Attack Precision=0.4838

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
0.15       0.8129   0.7619   0.7337   0.9987   0.9977   0.6162   <--
0.20       0.8129   0.7619   0.7337   0.9987   0.9977   0.6162  
0.25       0.8129   0.7619   0.7337   0.9987   0.9977   0.6162  
0.30       0.8129   0.7619   0.7337   0.9987   0.9977   0.6162  
0.35       0.8129   0.7619   0.7337   0.9987   0.9977   0.6162  
0.40       0.8129   0.7619   0.7337   0.9987   0.9977   0.6162  
0.45       0.8129   0.7619   0.7337   0.9987   0.9977   0.6162  
0.50       0.8129   0.7619   0.7337   0.9987   0.9977   0.6162  
0.55       0.8129   0.7619   0.7337   0.9987   0.9977   0.6162  
0.60       0.8129   0.7619   0.7337   0.9987   0.9977   0.6162  
0.65       0.8129   0.7619   0.7337   0.9987   0.9977   0.6162  
0.70       0.8129   0.7619   0.7337   0.9987   0.9977   0.6162  
0.75       0.8129   0.7619   0.7337   0.9987   0.9977   0.6162  
0.80       0.8129   0.7619   0.7337   0.9987   0.9977   0.6162  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8129, F1=0.7619, Normal Recall=0.7337, Normal Precision=0.9987, Attack Recall=0.9977, Attack Precision=0.6162

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
0.15       0.8389   0.8320   0.7330   0.9979   0.9977   0.7136   <--
0.20       0.8389   0.8320   0.7330   0.9979   0.9977   0.7136  
0.25       0.8389   0.8320   0.7330   0.9979   0.9977   0.7136  
0.30       0.8389   0.8320   0.7330   0.9979   0.9977   0.7136  
0.35       0.8389   0.8320   0.7330   0.9979   0.9977   0.7136  
0.40       0.8389   0.8320   0.7330   0.9979   0.9977   0.7136  
0.45       0.8389   0.8320   0.7330   0.9979   0.9977   0.7136  
0.50       0.8389   0.8320   0.7330   0.9979   0.9977   0.7136  
0.55       0.8389   0.8320   0.7330   0.9979   0.9977   0.7136  
0.60       0.8389   0.8320   0.7330   0.9979   0.9977   0.7136  
0.65       0.8389   0.8320   0.7330   0.9979   0.9977   0.7136  
0.70       0.8389   0.8320   0.7330   0.9979   0.9977   0.7136  
0.75       0.8389   0.8320   0.7330   0.9979   0.9977   0.7136  
0.80       0.8389   0.8320   0.7330   0.9979   0.9977   0.7136  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8389, F1=0.8320, Normal Recall=0.7330, Normal Precision=0.9979, Attack Recall=0.9977, Attack Precision=0.7136

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
0.15       0.8650   0.8808   0.7323   0.9969   0.9977   0.7885   <--
0.20       0.8650   0.8808   0.7323   0.9969   0.9977   0.7885  
0.25       0.8650   0.8808   0.7323   0.9969   0.9977   0.7885  
0.30       0.8650   0.8808   0.7323   0.9969   0.9977   0.7885  
0.35       0.8650   0.8808   0.7323   0.9969   0.9977   0.7885  
0.40       0.8650   0.8808   0.7323   0.9969   0.9977   0.7885  
0.45       0.8650   0.8808   0.7323   0.9969   0.9977   0.7885  
0.50       0.8650   0.8808   0.7323   0.9969   0.9977   0.7885  
0.55       0.8650   0.8808   0.7323   0.9969   0.9977   0.7885  
0.60       0.8650   0.8808   0.7323   0.9969   0.9977   0.7885  
0.65       0.8650   0.8808   0.7323   0.9969   0.9977   0.7885  
0.70       0.8650   0.8808   0.7323   0.9969   0.9977   0.7885  
0.75       0.8650   0.8808   0.7323   0.9969   0.9977   0.7885  
0.80       0.8650   0.8808   0.7323   0.9969   0.9977   0.7885  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8650, F1=0.8808, Normal Recall=0.7323, Normal Precision=0.9969, Attack Recall=0.9977, Attack Precision=0.7885

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
0.15       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905   <--
0.20       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.25       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.30       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.35       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.40       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.45       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.50       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.55       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.60       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.65       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.70       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.75       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.80       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7561, F1=0.4501, Normal Recall=0.7292, Normal Precision=0.9997, Attack Recall=0.9980, Attack Precision=0.2905

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
0.15       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798   <--
0.20       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.25       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.30       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.35       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.40       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.45       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.50       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.55       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.60       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.65       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.70       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.75       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.80       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7832, F1=0.6480, Normal Recall=0.7296, Normal Precision=0.9992, Attack Recall=0.9977, Attack Precision=0.4798

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
0.15       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124   <--
0.20       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.25       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.30       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.35       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.40       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.45       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.50       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.55       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.60       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.65       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.70       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.75       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.80       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8099, F1=0.7589, Normal Recall=0.7294, Normal Precision=0.9987, Attack Recall=0.9977, Attack Precision=0.6124

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
0.15       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101   <--
0.20       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.25       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.30       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.35       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.40       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.45       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.50       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.55       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.60       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.65       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.70       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.75       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.80       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8362, F1=0.8297, Normal Recall=0.7285, Normal Precision=0.9979, Attack Recall=0.9977, Attack Precision=0.7101

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
0.15       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856   <--
0.20       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.25       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.30       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.35       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.40       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.45       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.50       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.55       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.60       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.65       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.70       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.75       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.80       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8627, F1=0.8790, Normal Recall=0.7277, Normal Precision=0.9968, Attack Recall=0.9977, Attack Precision=0.7856

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
0.15       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905   <--
0.20       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.25       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.30       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.35       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.40       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.45       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.50       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.55       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.60       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.65       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.70       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.75       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
0.80       0.7561   0.4501   0.7292   0.9997   0.9980   0.2905  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7561, F1=0.4501, Normal Recall=0.7292, Normal Precision=0.9997, Attack Recall=0.9980, Attack Precision=0.2905

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
0.15       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798   <--
0.20       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.25       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.30       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.35       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.40       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.45       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.50       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.55       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.60       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.65       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.70       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.75       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
0.80       0.7832   0.6480   0.7296   0.9992   0.9977   0.4798  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7832, F1=0.6480, Normal Recall=0.7296, Normal Precision=0.9992, Attack Recall=0.9977, Attack Precision=0.4798

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
0.15       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124   <--
0.20       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.25       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.30       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.35       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.40       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.45       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.50       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.55       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.60       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.65       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.70       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.75       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
0.80       0.8099   0.7589   0.7294   0.9987   0.9977   0.6124  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8099, F1=0.7589, Normal Recall=0.7294, Normal Precision=0.9987, Attack Recall=0.9977, Attack Precision=0.6124

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
0.15       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101   <--
0.20       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.25       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.30       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.35       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.40       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.45       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.50       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.55       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.60       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.65       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.70       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.75       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
0.80       0.8362   0.8297   0.7285   0.9979   0.9977   0.7101  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8362, F1=0.8297, Normal Recall=0.7285, Normal Precision=0.9979, Attack Recall=0.9977, Attack Precision=0.7101

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
0.15       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856   <--
0.20       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.25       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.30       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.35       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.40       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.45       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.50       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.55       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.60       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.65       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.70       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.75       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
0.80       0.8627   0.8790   0.7277   0.9968   0.9977   0.7856  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8627, F1=0.8790, Normal Recall=0.7277, Normal Precision=0.9968, Attack Recall=0.9977, Attack Precision=0.7856

```

