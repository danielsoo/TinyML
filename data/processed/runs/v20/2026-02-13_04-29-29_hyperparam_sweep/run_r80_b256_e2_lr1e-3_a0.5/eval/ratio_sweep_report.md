# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e2_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-19 03:07:47 |

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
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9085 | 0.9048 | 0.9008 | 0.8984 | 0.8943 | 0.8901 | 0.8874 | 0.8834 | 0.8798 | 0.8766 | 0.8729 |
| QAT+Prune only | 0.3963 | 0.4568 | 0.5169 | 0.5779 | 0.6385 | 0.6970 | 0.7582 | 0.8173 | 0.8768 | 0.9387 | 0.9979 |
| QAT+PTQ | 0.3974 | 0.4576 | 0.5176 | 0.5784 | 0.6392 | 0.6975 | 0.7584 | 0.8175 | 0.8770 | 0.9389 | 0.9979 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.3974 | 0.4576 | 0.5176 | 0.5784 | 0.6392 | 0.6975 | 0.7584 | 0.8175 | 0.8770 | 0.9389 | 0.9979 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6469 | 0.7788 | 0.8375 | 0.8686 | 0.8882 | 0.9030 | 0.9129 | 0.9207 | 0.9272 | 0.9321 |
| QAT+Prune only | 0.0000 | 0.2687 | 0.4525 | 0.5865 | 0.6883 | 0.7671 | 0.8320 | 0.8844 | 0.9284 | 0.9670 | 0.9990 |
| QAT+PTQ | 0.0000 | 0.2690 | 0.4528 | 0.5868 | 0.6887 | 0.7674 | 0.8321 | 0.8845 | 0.9285 | 0.9671 | 0.9989 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2690 | 0.4528 | 0.5868 | 0.6887 | 0.7674 | 0.8321 | 0.8845 | 0.9285 | 0.9671 | 0.9989 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9085 | 0.9084 | 0.9078 | 0.9093 | 0.9086 | 0.9073 | 0.9092 | 0.9080 | 0.9074 | 0.9095 | 0.0000 |
| QAT+Prune only | 0.3963 | 0.3967 | 0.3967 | 0.3979 | 0.3989 | 0.3962 | 0.3985 | 0.3960 | 0.3923 | 0.4058 | 0.0000 |
| QAT+PTQ | 0.3974 | 0.3975 | 0.3975 | 0.3987 | 0.4001 | 0.3972 | 0.3992 | 0.3966 | 0.3934 | 0.4078 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.3974 | 0.3975 | 0.3975 | 0.3987 | 0.4001 | 0.3972 | 0.3992 | 0.3966 | 0.3934 | 0.4078 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9085 | 0.0000 | 0.0000 | 0.0000 | 0.9085 | 1.0000 |
| 90 | 10 | 299,940 | 0.9048 | 0.5140 | 0.8724 | 0.6469 | 0.9084 | 0.9846 |
| 80 | 20 | 291,350 | 0.9008 | 0.7031 | 0.8729 | 0.7788 | 0.9078 | 0.9662 |
| 70 | 30 | 194,230 | 0.8984 | 0.8049 | 0.8729 | 0.8375 | 0.9093 | 0.9435 |
| 60 | 40 | 145,675 | 0.8943 | 0.8643 | 0.8729 | 0.8686 | 0.9086 | 0.9147 |
| 50 | 50 | 116,540 | 0.8901 | 0.9040 | 0.8729 | 0.8882 | 0.9073 | 0.8771 |
| 40 | 60 | 97,115 | 0.8874 | 0.9352 | 0.8729 | 0.9030 | 0.9092 | 0.8267 |
| 30 | 70 | 83,240 | 0.8834 | 0.9568 | 0.8729 | 0.9129 | 0.9080 | 0.7538 |
| 20 | 80 | 72,835 | 0.8798 | 0.9742 | 0.8729 | 0.9207 | 0.9074 | 0.6409 |
| 10 | 90 | 64,740 | 0.8766 | 0.9886 | 0.8729 | 0.9272 | 0.9095 | 0.4429 |
| 0 | 100 | 58,270 | 0.8729 | 1.0000 | 0.8729 | 0.9321 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3963 | 0.0000 | 0.0000 | 0.0000 | 0.3963 | 1.0000 |
| 90 | 10 | 299,940 | 0.4568 | 0.1553 | 0.9980 | 0.2687 | 0.3967 | 0.9994 |
| 80 | 20 | 291,350 | 0.5169 | 0.2925 | 0.9979 | 0.4525 | 0.3967 | 0.9987 |
| 70 | 30 | 194,230 | 0.5779 | 0.4153 | 0.9979 | 0.5865 | 0.3979 | 0.9978 |
| 60 | 40 | 145,675 | 0.6385 | 0.5253 | 0.9979 | 0.6883 | 0.3989 | 0.9965 |
| 50 | 50 | 116,540 | 0.6970 | 0.6230 | 0.9979 | 0.7671 | 0.3962 | 0.9947 |
| 40 | 60 | 97,115 | 0.7582 | 0.7134 | 0.9979 | 0.8320 | 0.3985 | 0.9922 |
| 30 | 70 | 83,240 | 0.8173 | 0.7940 | 0.9979 | 0.8844 | 0.3960 | 0.9878 |
| 20 | 80 | 72,835 | 0.8768 | 0.8679 | 0.9979 | 0.9284 | 0.3923 | 0.9791 |
| 10 | 90 | 64,740 | 0.9387 | 0.9379 | 0.9979 | 0.9670 | 0.4058 | 0.9556 |
| 0 | 100 | 58,270 | 0.9979 | 1.0000 | 0.9979 | 0.9990 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3974 | 0.0000 | 0.0000 | 0.0000 | 0.3974 | 1.0000 |
| 90 | 10 | 299,940 | 0.4576 | 0.1554 | 0.9980 | 0.2690 | 0.3975 | 0.9994 |
| 80 | 20 | 291,350 | 0.5176 | 0.2928 | 0.9979 | 0.4528 | 0.3975 | 0.9987 |
| 70 | 30 | 194,230 | 0.5784 | 0.4156 | 0.9979 | 0.5868 | 0.3987 | 0.9977 |
| 60 | 40 | 145,675 | 0.6392 | 0.5258 | 0.9979 | 0.6887 | 0.4001 | 0.9965 |
| 50 | 50 | 116,540 | 0.6975 | 0.6234 | 0.9979 | 0.7674 | 0.3972 | 0.9947 |
| 40 | 60 | 97,115 | 0.7584 | 0.7136 | 0.9979 | 0.8321 | 0.3992 | 0.9921 |
| 30 | 70 | 83,240 | 0.8175 | 0.7942 | 0.9979 | 0.8845 | 0.3966 | 0.9877 |
| 20 | 80 | 72,835 | 0.8770 | 0.8681 | 0.9979 | 0.9285 | 0.3934 | 0.9790 |
| 10 | 90 | 64,740 | 0.9389 | 0.9381 | 0.9979 | 0.9671 | 0.4078 | 0.9555 |
| 0 | 100 | 58,270 | 0.9979 | 1.0000 | 0.9979 | 0.9989 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.3974 | 0.0000 | 0.0000 | 0.0000 | 0.3974 | 1.0000 |
| 90 | 10 | 299,940 | 0.4576 | 0.1554 | 0.9980 | 0.2690 | 0.3975 | 0.9994 |
| 80 | 20 | 291,350 | 0.5176 | 0.2928 | 0.9979 | 0.4528 | 0.3975 | 0.9987 |
| 70 | 30 | 194,230 | 0.5784 | 0.4156 | 0.9979 | 0.5868 | 0.3987 | 0.9977 |
| 60 | 40 | 145,675 | 0.6392 | 0.5258 | 0.9979 | 0.6887 | 0.4001 | 0.9965 |
| 50 | 50 | 116,540 | 0.6975 | 0.6234 | 0.9979 | 0.7674 | 0.3972 | 0.9947 |
| 40 | 60 | 97,115 | 0.7584 | 0.7136 | 0.9979 | 0.8321 | 0.3992 | 0.9921 |
| 30 | 70 | 83,240 | 0.8175 | 0.7942 | 0.9979 | 0.8845 | 0.3966 | 0.9877 |
| 20 | 80 | 72,835 | 0.8770 | 0.8681 | 0.9979 | 0.9285 | 0.3934 | 0.9790 |
| 10 | 90 | 64,740 | 0.9389 | 0.9381 | 0.9979 | 0.9671 | 0.4078 | 0.9555 |
| 0 | 100 | 58,270 | 0.9979 | 1.0000 | 0.9979 | 0.9989 | 0.0000 | 0.0000 |


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
0.15       0.9050   0.6482   0.9084   0.9849   0.8750   0.5148   <--
0.20       0.9050   0.6482   0.9084   0.9849   0.8750   0.5148  
0.25       0.9050   0.6482   0.9084   0.9849   0.8750   0.5148  
0.30       0.9050   0.6482   0.9084   0.9849   0.8750   0.5148  
0.35       0.9050   0.6482   0.9084   0.9849   0.8750   0.5148  
0.40       0.9050   0.6482   0.9084   0.9849   0.8750   0.5148  
0.45       0.9050   0.6482   0.9084   0.9849   0.8750   0.5148  
0.50       0.9050   0.6482   0.9084   0.9849   0.8750   0.5148  
0.55       0.9050   0.6482   0.9084   0.9849   0.8750   0.5148  
0.60       0.9050   0.6482   0.9084   0.9849   0.8750   0.5148  
0.65       0.9050   0.6482   0.9084   0.9849   0.8750   0.5148  
0.70       0.9050   0.6482   0.9084   0.9849   0.8750   0.5148  
0.75       0.9050   0.6482   0.9084   0.9849   0.8750   0.5148  
0.80       0.9050   0.6482   0.9084   0.9849   0.8750   0.5148  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9050, F1=0.6482, Normal Recall=0.9084, Normal Precision=0.9849, Attack Recall=0.8750, Attack Precision=0.5148

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
0.15       0.9015   0.7799   0.9086   0.9662   0.8729   0.7049   <--
0.20       0.9015   0.7799   0.9086   0.9662   0.8729   0.7049  
0.25       0.9015   0.7799   0.9086   0.9662   0.8729   0.7049  
0.30       0.9015   0.7799   0.9086   0.9662   0.8729   0.7049  
0.35       0.9015   0.7799   0.9086   0.9662   0.8729   0.7049  
0.40       0.9015   0.7799   0.9086   0.9662   0.8729   0.7049  
0.45       0.9015   0.7799   0.9086   0.9662   0.8729   0.7049  
0.50       0.9015   0.7799   0.9086   0.9662   0.8729   0.7049  
0.55       0.9015   0.7799   0.9086   0.9662   0.8729   0.7049  
0.60       0.9015   0.7799   0.9086   0.9662   0.8729   0.7049  
0.65       0.9015   0.7799   0.9086   0.9662   0.8729   0.7049  
0.70       0.9015   0.7799   0.9086   0.9662   0.8729   0.7049  
0.75       0.9015   0.7799   0.9086   0.9662   0.8729   0.7049  
0.80       0.9015   0.7799   0.9086   0.9662   0.8729   0.7049  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9015, F1=0.7799, Normal Recall=0.9086, Normal Precision=0.9662, Attack Recall=0.8729, Attack Precision=0.7049

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
0.15       0.8983   0.8374   0.9092   0.9435   0.8729   0.8046   <--
0.20       0.8983   0.8374   0.9092   0.9435   0.8729   0.8046  
0.25       0.8983   0.8374   0.9092   0.9435   0.8729   0.8046  
0.30       0.8983   0.8374   0.9092   0.9435   0.8729   0.8046  
0.35       0.8983   0.8374   0.9092   0.9435   0.8729   0.8046  
0.40       0.8983   0.8374   0.9092   0.9435   0.8729   0.8046  
0.45       0.8983   0.8374   0.9092   0.9435   0.8729   0.8046  
0.50       0.8983   0.8374   0.9092   0.9435   0.8729   0.8046  
0.55       0.8983   0.8374   0.9092   0.9435   0.8729   0.8046  
0.60       0.8983   0.8374   0.9092   0.9435   0.8729   0.8046  
0.65       0.8983   0.8374   0.9092   0.9435   0.8729   0.8046  
0.70       0.8983   0.8374   0.9092   0.9435   0.8729   0.8046  
0.75       0.8983   0.8374   0.9092   0.9435   0.8729   0.8046  
0.80       0.8983   0.8374   0.9092   0.9435   0.8729   0.8046  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8983, F1=0.8374, Normal Recall=0.9092, Normal Precision=0.9435, Attack Recall=0.8729, Attack Precision=0.8046

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
0.15       0.8945   0.8687   0.9088   0.9147   0.8729   0.8646   <--
0.20       0.8945   0.8687   0.9088   0.9147   0.8729   0.8646  
0.25       0.8945   0.8687   0.9088   0.9147   0.8729   0.8646  
0.30       0.8945   0.8687   0.9088   0.9147   0.8729   0.8646  
0.35       0.8945   0.8687   0.9088   0.9147   0.8729   0.8646  
0.40       0.8945   0.8687   0.9088   0.9147   0.8729   0.8646  
0.45       0.8945   0.8687   0.9088   0.9147   0.8729   0.8646  
0.50       0.8945   0.8687   0.9088   0.9147   0.8729   0.8646  
0.55       0.8945   0.8687   0.9088   0.9147   0.8729   0.8646  
0.60       0.8945   0.8687   0.9088   0.9147   0.8729   0.8646  
0.65       0.8945   0.8687   0.9088   0.9147   0.8729   0.8646  
0.70       0.8945   0.8687   0.9088   0.9147   0.8729   0.8646  
0.75       0.8945   0.8687   0.9088   0.9147   0.8729   0.8646  
0.80       0.8945   0.8687   0.9088   0.9147   0.8729   0.8646  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8945, F1=0.8687, Normal Recall=0.9088, Normal Precision=0.9147, Attack Recall=0.8729, Attack Precision=0.8646

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
0.15       0.8910   0.8890   0.9091   0.8773   0.8729   0.9057   <--
0.20       0.8910   0.8890   0.9091   0.8773   0.8729   0.9057  
0.25       0.8910   0.8890   0.9091   0.8773   0.8729   0.9057  
0.30       0.8910   0.8890   0.9091   0.8773   0.8729   0.9057  
0.35       0.8910   0.8890   0.9091   0.8773   0.8729   0.9057  
0.40       0.8910   0.8890   0.9091   0.8773   0.8729   0.9057  
0.45       0.8910   0.8890   0.9091   0.8773   0.8729   0.9057  
0.50       0.8910   0.8890   0.9091   0.8773   0.8729   0.9057  
0.55       0.8910   0.8890   0.9091   0.8773   0.8729   0.9057  
0.60       0.8910   0.8890   0.9091   0.8773   0.8729   0.9057  
0.65       0.8910   0.8890   0.9091   0.8773   0.8729   0.9057  
0.70       0.8910   0.8890   0.9091   0.8773   0.8729   0.9057  
0.75       0.8910   0.8890   0.9091   0.8773   0.8729   0.9057  
0.80       0.8910   0.8890   0.9091   0.8773   0.8729   0.9057  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8910, F1=0.8890, Normal Recall=0.9091, Normal Precision=0.8773, Attack Recall=0.8729, Attack Precision=0.9057

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
0.15       0.4568   0.2687   0.3967   0.9994   0.9980   0.1553   <--
0.20       0.4568   0.2687   0.3967   0.9994   0.9980   0.1553  
0.25       0.4568   0.2687   0.3967   0.9994   0.9980   0.1553  
0.30       0.4568   0.2687   0.3967   0.9994   0.9980   0.1553  
0.35       0.4568   0.2687   0.3967   0.9994   0.9980   0.1553  
0.40       0.4568   0.2687   0.3967   0.9994   0.9980   0.1553  
0.45       0.4568   0.2687   0.3967   0.9994   0.9980   0.1553  
0.50       0.4568   0.2687   0.3967   0.9994   0.9980   0.1553  
0.55       0.4568   0.2687   0.3967   0.9994   0.9980   0.1553  
0.60       0.4568   0.2687   0.3967   0.9994   0.9980   0.1553  
0.65       0.4568   0.2687   0.3967   0.9994   0.9980   0.1553  
0.70       0.4568   0.2687   0.3967   0.9994   0.9980   0.1553  
0.75       0.4568   0.2687   0.3967   0.9994   0.9980   0.1553  
0.80       0.4568   0.2687   0.3967   0.9994   0.9980   0.1553  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4568, F1=0.2687, Normal Recall=0.3967, Normal Precision=0.9994, Attack Recall=0.9980, Attack Precision=0.1553

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
0.15       0.5170   0.4525   0.3968   0.9987   0.9979   0.2926   <--
0.20       0.5170   0.4525   0.3968   0.9987   0.9979   0.2926  
0.25       0.5170   0.4525   0.3968   0.9987   0.9979   0.2926  
0.30       0.5170   0.4525   0.3968   0.9987   0.9979   0.2926  
0.35       0.5170   0.4525   0.3968   0.9987   0.9979   0.2926  
0.40       0.5170   0.4525   0.3968   0.9987   0.9979   0.2926  
0.45       0.5170   0.4525   0.3968   0.9987   0.9979   0.2926  
0.50       0.5170   0.4525   0.3968   0.9987   0.9979   0.2926  
0.55       0.5170   0.4525   0.3968   0.9987   0.9979   0.2926  
0.60       0.5170   0.4525   0.3968   0.9987   0.9979   0.2926  
0.65       0.5170   0.4525   0.3968   0.9987   0.9979   0.2926  
0.70       0.5170   0.4525   0.3968   0.9987   0.9979   0.2926  
0.75       0.5170   0.4525   0.3968   0.9987   0.9979   0.2926  
0.80       0.5170   0.4525   0.3968   0.9987   0.9979   0.2926  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5170, F1=0.4525, Normal Recall=0.3968, Normal Precision=0.9987, Attack Recall=0.9979, Attack Precision=0.2926

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
0.15       0.5773   0.5862   0.3971   0.9977   0.9979   0.4150   <--
0.20       0.5773   0.5862   0.3971   0.9977   0.9979   0.4150  
0.25       0.5773   0.5862   0.3971   0.9977   0.9979   0.4150  
0.30       0.5773   0.5862   0.3971   0.9977   0.9979   0.4150  
0.35       0.5773   0.5862   0.3971   0.9977   0.9979   0.4150  
0.40       0.5773   0.5862   0.3971   0.9977   0.9979   0.4150  
0.45       0.5773   0.5862   0.3971   0.9977   0.9979   0.4150  
0.50       0.5773   0.5862   0.3971   0.9977   0.9979   0.4150  
0.55       0.5773   0.5862   0.3971   0.9977   0.9979   0.4150  
0.60       0.5773   0.5862   0.3971   0.9977   0.9979   0.4150  
0.65       0.5773   0.5862   0.3971   0.9977   0.9979   0.4150  
0.70       0.5773   0.5862   0.3971   0.9977   0.9979   0.4150  
0.75       0.5773   0.5862   0.3971   0.9977   0.9979   0.4150  
0.80       0.5773   0.5862   0.3971   0.9977   0.9979   0.4150  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5773, F1=0.5862, Normal Recall=0.3971, Normal Precision=0.9977, Attack Recall=0.9979, Attack Precision=0.4150

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
0.15       0.6368   0.6873   0.3961   0.9965   0.9979   0.5242   <--
0.20       0.6368   0.6873   0.3961   0.9965   0.9979   0.5242  
0.25       0.6368   0.6873   0.3961   0.9965   0.9979   0.5242  
0.30       0.6368   0.6873   0.3961   0.9965   0.9979   0.5242  
0.35       0.6368   0.6873   0.3961   0.9965   0.9979   0.5242  
0.40       0.6368   0.6873   0.3961   0.9965   0.9979   0.5242  
0.45       0.6368   0.6873   0.3961   0.9965   0.9979   0.5242  
0.50       0.6368   0.6873   0.3961   0.9965   0.9979   0.5242  
0.55       0.6368   0.6873   0.3961   0.9965   0.9979   0.5242  
0.60       0.6368   0.6873   0.3961   0.9965   0.9979   0.5242  
0.65       0.6368   0.6873   0.3961   0.9965   0.9979   0.5242  
0.70       0.6368   0.6873   0.3961   0.9965   0.9979   0.5242  
0.75       0.6368   0.6873   0.3961   0.9965   0.9979   0.5242  
0.80       0.6368   0.6873   0.3961   0.9965   0.9979   0.5242  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6368, F1=0.6873, Normal Recall=0.3961, Normal Precision=0.9965, Attack Recall=0.9979, Attack Precision=0.5242

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
0.15       0.6962   0.7666   0.3945   0.9947   0.9979   0.6224   <--
0.20       0.6962   0.7666   0.3945   0.9947   0.9979   0.6224  
0.25       0.6962   0.7666   0.3945   0.9947   0.9979   0.6224  
0.30       0.6962   0.7666   0.3945   0.9947   0.9979   0.6224  
0.35       0.6962   0.7666   0.3945   0.9947   0.9979   0.6224  
0.40       0.6962   0.7666   0.3945   0.9947   0.9979   0.6224  
0.45       0.6962   0.7666   0.3945   0.9947   0.9979   0.6224  
0.50       0.6962   0.7666   0.3945   0.9947   0.9979   0.6224  
0.55       0.6962   0.7666   0.3945   0.9947   0.9979   0.6224  
0.60       0.6962   0.7666   0.3945   0.9947   0.9979   0.6224  
0.65       0.6962   0.7666   0.3945   0.9947   0.9979   0.6224  
0.70       0.6962   0.7666   0.3945   0.9947   0.9979   0.6224  
0.75       0.6962   0.7666   0.3945   0.9947   0.9979   0.6224  
0.80       0.6962   0.7666   0.3945   0.9947   0.9979   0.6224  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6962, F1=0.7666, Normal Recall=0.3945, Normal Precision=0.9947, Attack Recall=0.9979, Attack Precision=0.6224

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
0.15       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554   <--
0.20       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.25       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.30       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.35       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.40       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.45       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.50       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.55       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.60       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.65       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.70       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.75       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.80       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4576, F1=0.2690, Normal Recall=0.3975, Normal Precision=0.9994, Attack Recall=0.9979, Attack Precision=0.1554

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
0.15       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929   <--
0.20       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.25       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.30       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.35       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.40       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.45       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.50       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.55       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.60       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.65       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.70       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.75       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.80       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5177, F1=0.4529, Normal Recall=0.3977, Normal Precision=0.9987, Attack Recall=0.9979, Attack Precision=0.2929

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
0.15       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154   <--
0.20       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.25       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.30       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.35       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.40       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.45       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.50       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.55       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.60       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.65       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.70       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.75       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.80       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5781, F1=0.5866, Normal Recall=0.3981, Normal Precision=0.9977, Attack Recall=0.9979, Attack Precision=0.4154

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
0.15       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246   <--
0.20       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.25       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.30       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.35       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.40       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.45       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.50       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.55       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.60       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.65       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.70       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.75       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.80       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6375, F1=0.6877, Normal Recall=0.3972, Normal Precision=0.9965, Attack Recall=0.9979, Attack Precision=0.5246

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
0.15       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228   <--
0.20       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.25       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.30       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.35       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.40       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.45       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.50       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.55       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.60       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.65       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.70       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.75       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.80       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6967, F1=0.7669, Normal Recall=0.3956, Normal Precision=0.9947, Attack Recall=0.9979, Attack Precision=0.6228

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
0.15       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554   <--
0.20       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.25       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.30       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.35       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.40       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.45       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.50       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.55       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.60       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.65       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.70       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.75       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
0.80       0.4576   0.2690   0.3975   0.9994   0.9979   0.1554  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4576, F1=0.2690, Normal Recall=0.3975, Normal Precision=0.9994, Attack Recall=0.9979, Attack Precision=0.1554

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
0.15       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929   <--
0.20       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.25       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.30       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.35       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.40       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.45       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.50       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.55       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.60       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.65       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.70       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.75       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
0.80       0.5177   0.4529   0.3977   0.9987   0.9979   0.2929  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5177, F1=0.4529, Normal Recall=0.3977, Normal Precision=0.9987, Attack Recall=0.9979, Attack Precision=0.2929

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
0.15       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154   <--
0.20       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.25       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.30       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.35       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.40       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.45       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.50       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.55       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.60       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.65       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.70       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.75       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
0.80       0.5781   0.5866   0.3981   0.9977   0.9979   0.4154  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5781, F1=0.5866, Normal Recall=0.3981, Normal Precision=0.9977, Attack Recall=0.9979, Attack Precision=0.4154

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
0.15       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246   <--
0.20       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.25       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.30       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.35       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.40       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.45       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.50       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.55       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.60       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.65       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.70       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.75       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
0.80       0.6375   0.6877   0.3972   0.9965   0.9979   0.5246  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6375, F1=0.6877, Normal Recall=0.3972, Normal Precision=0.9965, Attack Recall=0.9979, Attack Precision=0.5246

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
0.15       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228   <--
0.20       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.25       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.30       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.35       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.40       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.45       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.50       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.55       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.60       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.65       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.70       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.75       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
0.80       0.6967   0.7669   0.3956   0.9947   0.9979   0.6228  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6967, F1=0.7669, Normal Recall=0.3956, Normal Precision=0.9947, Attack Recall=0.9979, Attack Precision=0.6228

```

