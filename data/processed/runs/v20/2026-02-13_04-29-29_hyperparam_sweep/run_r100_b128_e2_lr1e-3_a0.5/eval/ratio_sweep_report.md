# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e2_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-21 21:09:04 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5116 | 0.5511 | 0.5886 | 0.6261 | 0.6641 | 0.7014 | 0.7393 | 0.7786 | 0.8145 | 0.8526 | 0.8907 |
| QAT+Prune only | 0.7942 | 0.8150 | 0.8341 | 0.8540 | 0.8735 | 0.8914 | 0.9121 | 0.9322 | 0.9515 | 0.9707 | 0.9906 |
| QAT+PTQ | 0.7953 | 0.8162 | 0.8351 | 0.8549 | 0.8743 | 0.8922 | 0.9127 | 0.9325 | 0.9517 | 0.9708 | 0.9906 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7953 | 0.8162 | 0.8351 | 0.8549 | 0.8743 | 0.8922 | 0.9127 | 0.9325 | 0.9517 | 0.9708 | 0.9906 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2844 | 0.4641 | 0.5883 | 0.6796 | 0.7489 | 0.8039 | 0.8492 | 0.8848 | 0.9158 | 0.9422 |
| QAT+Prune only | 0.0000 | 0.5172 | 0.7049 | 0.8028 | 0.8624 | 0.9012 | 0.9312 | 0.9534 | 0.9703 | 0.9838 | 0.9953 |
| QAT+PTQ | 0.0000 | 0.5189 | 0.7062 | 0.8038 | 0.8631 | 0.9019 | 0.9316 | 0.9536 | 0.9704 | 0.9839 | 0.9953 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5189 | 0.7062 | 0.8038 | 0.8631 | 0.9019 | 0.9316 | 0.9536 | 0.9704 | 0.9839 | 0.9953 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5116 | 0.5132 | 0.5131 | 0.5127 | 0.5131 | 0.5121 | 0.5123 | 0.5170 | 0.5096 | 0.5100 | 0.0000 |
| QAT+Prune only | 0.7942 | 0.7955 | 0.7949 | 0.7954 | 0.7954 | 0.7921 | 0.7943 | 0.7960 | 0.7948 | 0.7909 | 0.0000 |
| QAT+PTQ | 0.7953 | 0.7968 | 0.7962 | 0.7967 | 0.7967 | 0.7939 | 0.7958 | 0.7969 | 0.7961 | 0.7921 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7953 | 0.7968 | 0.7962 | 0.7967 | 0.7967 | 0.7939 | 0.7958 | 0.7969 | 0.7961 | 0.7921 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5116 | 0.0000 | 0.0000 | 0.0000 | 0.5116 | 1.0000 |
| 90 | 10 | 299,940 | 0.5511 | 0.1692 | 0.8920 | 0.2844 | 0.5132 | 0.9772 |
| 80 | 20 | 291,350 | 0.5886 | 0.3138 | 0.8907 | 0.4641 | 0.5131 | 0.9494 |
| 70 | 30 | 194,230 | 0.6261 | 0.4392 | 0.8907 | 0.5883 | 0.5127 | 0.9163 |
| 60 | 40 | 145,675 | 0.6641 | 0.5495 | 0.8907 | 0.6796 | 0.5131 | 0.8756 |
| 50 | 50 | 116,540 | 0.7014 | 0.6461 | 0.8907 | 0.7489 | 0.5121 | 0.8241 |
| 40 | 60 | 97,115 | 0.7393 | 0.7326 | 0.8907 | 0.8039 | 0.5123 | 0.7575 |
| 30 | 70 | 83,240 | 0.7786 | 0.8114 | 0.8907 | 0.8492 | 0.5170 | 0.6696 |
| 20 | 80 | 72,835 | 0.8145 | 0.8790 | 0.8907 | 0.8848 | 0.5096 | 0.5382 |
| 10 | 90 | 64,740 | 0.8526 | 0.9424 | 0.8907 | 0.9158 | 0.5100 | 0.3414 |
| 0 | 100 | 58,270 | 0.8907 | 1.0000 | 0.8907 | 0.9422 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7942 | 0.0000 | 0.0000 | 0.0000 | 0.7942 | 1.0000 |
| 90 | 10 | 299,940 | 0.8150 | 0.3499 | 0.9909 | 0.5172 | 0.7955 | 0.9987 |
| 80 | 20 | 291,350 | 0.8341 | 0.5470 | 0.9906 | 0.7049 | 0.7949 | 0.9971 |
| 70 | 30 | 194,230 | 0.8540 | 0.6748 | 0.9906 | 0.8028 | 0.7954 | 0.9950 |
| 60 | 40 | 145,675 | 0.8735 | 0.7635 | 0.9906 | 0.8624 | 0.7954 | 0.9922 |
| 50 | 50 | 116,540 | 0.8914 | 0.8265 | 0.9906 | 0.9012 | 0.7921 | 0.9883 |
| 40 | 60 | 97,115 | 0.9121 | 0.8784 | 0.9906 | 0.9312 | 0.7943 | 0.9826 |
| 30 | 70 | 83,240 | 0.9322 | 0.9189 | 0.9906 | 0.9534 | 0.7960 | 0.9733 |
| 20 | 80 | 72,835 | 0.9515 | 0.9508 | 0.9906 | 0.9703 | 0.7948 | 0.9550 |
| 10 | 90 | 64,740 | 0.9707 | 0.9771 | 0.9906 | 0.9838 | 0.7909 | 0.9036 |
| 0 | 100 | 58,270 | 0.9906 | 1.0000 | 0.9906 | 0.9953 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7953 | 0.0000 | 0.0000 | 0.0000 | 0.7953 | 1.0000 |
| 90 | 10 | 299,940 | 0.8162 | 0.3514 | 0.9909 | 0.5189 | 0.7968 | 0.9987 |
| 80 | 20 | 291,350 | 0.8351 | 0.5486 | 0.9906 | 0.7062 | 0.7962 | 0.9971 |
| 70 | 30 | 194,230 | 0.8549 | 0.6762 | 0.9906 | 0.8038 | 0.7967 | 0.9950 |
| 60 | 40 | 145,675 | 0.8743 | 0.7646 | 0.9906 | 0.8631 | 0.7967 | 0.9922 |
| 50 | 50 | 116,540 | 0.8922 | 0.8278 | 0.9906 | 0.9019 | 0.7939 | 0.9883 |
| 40 | 60 | 97,115 | 0.9127 | 0.8792 | 0.9906 | 0.9316 | 0.7958 | 0.9826 |
| 30 | 70 | 83,240 | 0.9325 | 0.9192 | 0.9906 | 0.9536 | 0.7969 | 0.9733 |
| 20 | 80 | 72,835 | 0.9517 | 0.9511 | 0.9906 | 0.9704 | 0.7961 | 0.9550 |
| 10 | 90 | 64,740 | 0.9708 | 0.9772 | 0.9906 | 0.9839 | 0.7921 | 0.9038 |
| 0 | 100 | 58,270 | 0.9906 | 1.0000 | 0.9906 | 0.9953 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7953 | 0.0000 | 0.0000 | 0.0000 | 0.7953 | 1.0000 |
| 90 | 10 | 299,940 | 0.8162 | 0.3514 | 0.9909 | 0.5189 | 0.7968 | 0.9987 |
| 80 | 20 | 291,350 | 0.8351 | 0.5486 | 0.9906 | 0.7062 | 0.7962 | 0.9971 |
| 70 | 30 | 194,230 | 0.8549 | 0.6762 | 0.9906 | 0.8038 | 0.7967 | 0.9950 |
| 60 | 40 | 145,675 | 0.8743 | 0.7646 | 0.9906 | 0.8631 | 0.7967 | 0.9922 |
| 50 | 50 | 116,540 | 0.8922 | 0.8278 | 0.9906 | 0.9019 | 0.7939 | 0.9883 |
| 40 | 60 | 97,115 | 0.9127 | 0.8792 | 0.9906 | 0.9316 | 0.7958 | 0.9826 |
| 30 | 70 | 83,240 | 0.9325 | 0.9192 | 0.9906 | 0.9536 | 0.7969 | 0.9733 |
| 20 | 80 | 72,835 | 0.9517 | 0.9511 | 0.9906 | 0.9704 | 0.7961 | 0.9550 |
| 10 | 90 | 64,740 | 0.9708 | 0.9772 | 0.9906 | 0.9839 | 0.7921 | 0.9038 |
| 0 | 100 | 58,270 | 0.9906 | 1.0000 | 0.9906 | 0.9953 | 0.0000 | 0.0000 |


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
0.15       0.5512   0.2845   0.5132   0.9772   0.8924   0.1692   <--
0.20       0.5512   0.2845   0.5132   0.9772   0.8924   0.1692  
0.25       0.5512   0.2845   0.5132   0.9772   0.8924   0.1692  
0.30       0.5512   0.2845   0.5132   0.9772   0.8924   0.1692  
0.35       0.5512   0.2845   0.5132   0.9772   0.8924   0.1692  
0.40       0.5512   0.2845   0.5132   0.9772   0.8924   0.1692  
0.45       0.5512   0.2845   0.5132   0.9772   0.8924   0.1692  
0.50       0.5512   0.2845   0.5132   0.9772   0.8924   0.1692  
0.55       0.5512   0.2845   0.5132   0.9772   0.8924   0.1692  
0.60       0.5512   0.2845   0.5132   0.9772   0.8924   0.1692  
0.65       0.5512   0.2845   0.5132   0.9772   0.8924   0.1692  
0.70       0.5512   0.2845   0.5132   0.9772   0.8924   0.1692  
0.75       0.5512   0.2845   0.5132   0.9772   0.8924   0.1692  
0.80       0.5512   0.2845   0.5132   0.9772   0.8924   0.1692  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5512, F1=0.2845, Normal Recall=0.5132, Normal Precision=0.9772, Attack Recall=0.8924, Attack Precision=0.1692

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
0.15       0.5887   0.4641   0.5132   0.9494   0.8907   0.3138   <--
0.20       0.5887   0.4641   0.5132   0.9494   0.8907   0.3138  
0.25       0.5887   0.4641   0.5132   0.9494   0.8907   0.3138  
0.30       0.5887   0.4641   0.5132   0.9494   0.8907   0.3138  
0.35       0.5887   0.4641   0.5132   0.9494   0.8907   0.3138  
0.40       0.5887   0.4641   0.5132   0.9494   0.8907   0.3138  
0.45       0.5887   0.4641   0.5132   0.9494   0.8907   0.3138  
0.50       0.5887   0.4641   0.5132   0.9494   0.8907   0.3138  
0.55       0.5887   0.4641   0.5132   0.9494   0.8907   0.3138  
0.60       0.5887   0.4641   0.5132   0.9494   0.8907   0.3138  
0.65       0.5887   0.4641   0.5132   0.9494   0.8907   0.3138  
0.70       0.5887   0.4641   0.5132   0.9494   0.8907   0.3138  
0.75       0.5887   0.4641   0.5132   0.9494   0.8907   0.3138  
0.80       0.5887   0.4641   0.5132   0.9494   0.8907   0.3138  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5887, F1=0.4641, Normal Recall=0.5132, Normal Precision=0.9494, Attack Recall=0.8907, Attack Precision=0.3138

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
0.15       0.6263   0.5885   0.5130   0.9163   0.8907   0.4394   <--
0.20       0.6263   0.5885   0.5130   0.9163   0.8907   0.4394  
0.25       0.6263   0.5885   0.5130   0.9163   0.8907   0.4394  
0.30       0.6263   0.5885   0.5130   0.9163   0.8907   0.4394  
0.35       0.6263   0.5885   0.5130   0.9163   0.8907   0.4394  
0.40       0.6263   0.5885   0.5130   0.9163   0.8907   0.4394  
0.45       0.6263   0.5885   0.5130   0.9163   0.8907   0.4394  
0.50       0.6263   0.5885   0.5130   0.9163   0.8907   0.4394  
0.55       0.6263   0.5885   0.5130   0.9163   0.8907   0.4394  
0.60       0.6263   0.5885   0.5130   0.9163   0.8907   0.4394  
0.65       0.6263   0.5885   0.5130   0.9163   0.8907   0.4394  
0.70       0.6263   0.5885   0.5130   0.9163   0.8907   0.4394  
0.75       0.6263   0.5885   0.5130   0.9163   0.8907   0.4394  
0.80       0.6263   0.5885   0.5130   0.9163   0.8907   0.4394  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6263, F1=0.5885, Normal Recall=0.5130, Normal Precision=0.9163, Attack Recall=0.8907, Attack Precision=0.4394

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
0.15       0.6635   0.6792   0.5121   0.8754   0.8907   0.5489   <--
0.20       0.6635   0.6792   0.5121   0.8754   0.8907   0.5489  
0.25       0.6635   0.6792   0.5121   0.8754   0.8907   0.5489  
0.30       0.6635   0.6792   0.5121   0.8754   0.8907   0.5489  
0.35       0.6635   0.6792   0.5121   0.8754   0.8907   0.5489  
0.40       0.6635   0.6792   0.5121   0.8754   0.8907   0.5489  
0.45       0.6635   0.6792   0.5121   0.8754   0.8907   0.5489  
0.50       0.6635   0.6792   0.5121   0.8754   0.8907   0.5489  
0.55       0.6635   0.6792   0.5121   0.8754   0.8907   0.5489  
0.60       0.6635   0.6792   0.5121   0.8754   0.8907   0.5489  
0.65       0.6635   0.6792   0.5121   0.8754   0.8907   0.5489  
0.70       0.6635   0.6792   0.5121   0.8754   0.8907   0.5489  
0.75       0.6635   0.6792   0.5121   0.8754   0.8907   0.5489  
0.80       0.6635   0.6792   0.5121   0.8754   0.8907   0.5489  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6635, F1=0.6792, Normal Recall=0.5121, Normal Precision=0.8754, Attack Recall=0.8907, Attack Precision=0.5489

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
0.15       0.7022   0.7495   0.5138   0.8246   0.8907   0.6469   <--
0.20       0.7022   0.7495   0.5138   0.8246   0.8907   0.6469  
0.25       0.7022   0.7495   0.5138   0.8246   0.8907   0.6469  
0.30       0.7022   0.7495   0.5138   0.8246   0.8907   0.6469  
0.35       0.7022   0.7495   0.5138   0.8246   0.8907   0.6469  
0.40       0.7022   0.7495   0.5138   0.8246   0.8907   0.6469  
0.45       0.7022   0.7495   0.5138   0.8246   0.8907   0.6469  
0.50       0.7022   0.7495   0.5138   0.8246   0.8907   0.6469  
0.55       0.7022   0.7495   0.5138   0.8246   0.8907   0.6469  
0.60       0.7022   0.7495   0.5138   0.8246   0.8907   0.6469  
0.65       0.7022   0.7495   0.5138   0.8246   0.8907   0.6469  
0.70       0.7022   0.7495   0.5138   0.8246   0.8907   0.6469  
0.75       0.7022   0.7495   0.5138   0.8246   0.8907   0.6469  
0.80       0.7022   0.7495   0.5138   0.8246   0.8907   0.6469  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7022, F1=0.7495, Normal Recall=0.5138, Normal Precision=0.8246, Attack Recall=0.8907, Attack Precision=0.6469

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
0.15       0.8150   0.5174   0.7955   0.9988   0.9913   0.3500   <--
0.20       0.8150   0.5174   0.7955   0.9988   0.9913   0.3500  
0.25       0.8150   0.5174   0.7955   0.9988   0.9913   0.3500  
0.30       0.8150   0.5174   0.7955   0.9988   0.9913   0.3500  
0.35       0.8150   0.5174   0.7955   0.9988   0.9913   0.3500  
0.40       0.8150   0.5174   0.7955   0.9988   0.9913   0.3500  
0.45       0.8150   0.5174   0.7955   0.9988   0.9913   0.3500  
0.50       0.8150   0.5174   0.7955   0.9988   0.9913   0.3500  
0.55       0.8150   0.5174   0.7955   0.9988   0.9913   0.3500  
0.60       0.8150   0.5174   0.7955   0.9988   0.9913   0.3500  
0.65       0.8150   0.5174   0.7955   0.9988   0.9913   0.3500  
0.70       0.8150   0.5174   0.7955   0.9988   0.9913   0.3500  
0.75       0.8150   0.5174   0.7955   0.9988   0.9913   0.3500  
0.80       0.8150   0.5174   0.7955   0.9988   0.9913   0.3500  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8150, F1=0.5174, Normal Recall=0.7955, Normal Precision=0.9988, Attack Recall=0.9913, Attack Precision=0.3500

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
0.15       0.8345   0.7054   0.7955   0.9971   0.9906   0.5477   <--
0.20       0.8345   0.7054   0.7955   0.9971   0.9906   0.5477  
0.25       0.8345   0.7054   0.7955   0.9971   0.9906   0.5477  
0.30       0.8345   0.7054   0.7955   0.9971   0.9906   0.5477  
0.35       0.8345   0.7054   0.7955   0.9971   0.9906   0.5477  
0.40       0.8345   0.7054   0.7955   0.9971   0.9906   0.5477  
0.45       0.8345   0.7054   0.7955   0.9971   0.9906   0.5477  
0.50       0.8345   0.7054   0.7955   0.9971   0.9906   0.5477  
0.55       0.8345   0.7054   0.7955   0.9971   0.9906   0.5477  
0.60       0.8345   0.7054   0.7955   0.9971   0.9906   0.5477  
0.65       0.8345   0.7054   0.7955   0.9971   0.9906   0.5477  
0.70       0.8345   0.7054   0.7955   0.9971   0.9906   0.5477  
0.75       0.8345   0.7054   0.7955   0.9971   0.9906   0.5477  
0.80       0.8345   0.7054   0.7955   0.9971   0.9906   0.5477  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8345, F1=0.7054, Normal Recall=0.7955, Normal Precision=0.9971, Attack Recall=0.9906, Attack Precision=0.5477

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
0.15       0.8536   0.8024   0.7949   0.9950   0.9906   0.6743   <--
0.20       0.8536   0.8024   0.7949   0.9950   0.9906   0.6743  
0.25       0.8536   0.8024   0.7949   0.9950   0.9906   0.6743  
0.30       0.8536   0.8024   0.7949   0.9950   0.9906   0.6743  
0.35       0.8536   0.8024   0.7949   0.9950   0.9906   0.6743  
0.40       0.8536   0.8024   0.7949   0.9950   0.9906   0.6743  
0.45       0.8536   0.8024   0.7949   0.9950   0.9906   0.6743  
0.50       0.8536   0.8024   0.7949   0.9950   0.9906   0.6743  
0.55       0.8536   0.8024   0.7949   0.9950   0.9906   0.6743  
0.60       0.8536   0.8024   0.7949   0.9950   0.9906   0.6743  
0.65       0.8536   0.8024   0.7949   0.9950   0.9906   0.6743  
0.70       0.8536   0.8024   0.7949   0.9950   0.9906   0.6743  
0.75       0.8536   0.8024   0.7949   0.9950   0.9906   0.6743  
0.80       0.8536   0.8024   0.7949   0.9950   0.9906   0.6743  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8536, F1=0.8024, Normal Recall=0.7949, Normal Precision=0.9950, Attack Recall=0.9906, Attack Precision=0.6743

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
0.15       0.8727   0.8616   0.7941   0.9922   0.9906   0.7623   <--
0.20       0.8727   0.8616   0.7941   0.9922   0.9906   0.7623  
0.25       0.8727   0.8616   0.7941   0.9922   0.9906   0.7623  
0.30       0.8727   0.8616   0.7941   0.9922   0.9906   0.7623  
0.35       0.8727   0.8616   0.7941   0.9922   0.9906   0.7623  
0.40       0.8727   0.8616   0.7941   0.9922   0.9906   0.7623  
0.45       0.8727   0.8616   0.7941   0.9922   0.9906   0.7623  
0.50       0.8727   0.8616   0.7941   0.9922   0.9906   0.7623  
0.55       0.8727   0.8616   0.7941   0.9922   0.9906   0.7623  
0.60       0.8727   0.8616   0.7941   0.9922   0.9906   0.7623  
0.65       0.8727   0.8616   0.7941   0.9922   0.9906   0.7623  
0.70       0.8727   0.8616   0.7941   0.9922   0.9906   0.7623  
0.75       0.8727   0.8616   0.7941   0.9922   0.9906   0.7623  
0.80       0.8727   0.8616   0.7941   0.9922   0.9906   0.7623  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8727, F1=0.8616, Normal Recall=0.7941, Normal Precision=0.9922, Attack Recall=0.9906, Attack Precision=0.7623

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
0.15       0.8915   0.9013   0.7924   0.9883   0.9906   0.8268   <--
0.20       0.8915   0.9013   0.7924   0.9883   0.9906   0.8268  
0.25       0.8915   0.9013   0.7924   0.9883   0.9906   0.8268  
0.30       0.8915   0.9013   0.7924   0.9883   0.9906   0.8268  
0.35       0.8915   0.9013   0.7924   0.9883   0.9906   0.8268  
0.40       0.8915   0.9013   0.7924   0.9883   0.9906   0.8268  
0.45       0.8915   0.9013   0.7924   0.9883   0.9906   0.8268  
0.50       0.8915   0.9013   0.7924   0.9883   0.9906   0.8268  
0.55       0.8915   0.9013   0.7924   0.9883   0.9906   0.8268  
0.60       0.8915   0.9013   0.7924   0.9883   0.9906   0.8268  
0.65       0.8915   0.9013   0.7924   0.9883   0.9906   0.8268  
0.70       0.8915   0.9013   0.7924   0.9883   0.9906   0.8268  
0.75       0.8915   0.9013   0.7924   0.9883   0.9906   0.8268  
0.80       0.8915   0.9013   0.7924   0.9883   0.9906   0.8268  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8915, F1=0.9013, Normal Recall=0.7924, Normal Precision=0.9883, Attack Recall=0.9906, Attack Precision=0.8268

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
0.15       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515   <--
0.20       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.25       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.30       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.35       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.40       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.45       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.50       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.55       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.60       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.65       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.70       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.75       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.80       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8163, F1=0.5190, Normal Recall=0.7968, Normal Precision=0.9988, Attack Recall=0.9913, Attack Precision=0.3515

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
0.15       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493   <--
0.20       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.25       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.30       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.35       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.40       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.45       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.50       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.55       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.60       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.65       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.70       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.75       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.80       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8356, F1=0.7068, Normal Recall=0.7968, Normal Precision=0.9971, Attack Recall=0.9906, Attack Precision=0.5493

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
0.15       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756   <--
0.20       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.25       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.30       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.35       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.40       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.45       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.50       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.55       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.60       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.65       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.70       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.75       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.80       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8545, F1=0.8033, Normal Recall=0.7961, Normal Precision=0.9950, Attack Recall=0.9906, Attack Precision=0.6756

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
0.15       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633   <--
0.20       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.25       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.30       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.35       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.40       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.45       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.50       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.55       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.60       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.65       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.70       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.75       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.80       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8734, F1=0.8623, Normal Recall=0.7953, Normal Precision=0.9922, Attack Recall=0.9906, Attack Precision=0.7633

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
0.15       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277   <--
0.20       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.25       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.30       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.35       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.40       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.45       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.50       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.55       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.60       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.65       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.70       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.75       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.80       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8922, F1=0.9019, Normal Recall=0.7938, Normal Precision=0.9883, Attack Recall=0.9906, Attack Precision=0.8277

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
0.15       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515   <--
0.20       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.25       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.30       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.35       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.40       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.45       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.50       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.55       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.60       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.65       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.70       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.75       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
0.80       0.8163   0.5190   0.7968   0.9988   0.9913   0.3515  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8163, F1=0.5190, Normal Recall=0.7968, Normal Precision=0.9988, Attack Recall=0.9913, Attack Precision=0.3515

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
0.15       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493   <--
0.20       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.25       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.30       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.35       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.40       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.45       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.50       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.55       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.60       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.65       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.70       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.75       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
0.80       0.8356   0.7068   0.7968   0.9971   0.9906   0.5493  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8356, F1=0.7068, Normal Recall=0.7968, Normal Precision=0.9971, Attack Recall=0.9906, Attack Precision=0.5493

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
0.15       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756   <--
0.20       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.25       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.30       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.35       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.40       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.45       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.50       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.55       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.60       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.65       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.70       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.75       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
0.80       0.8545   0.8033   0.7961   0.9950   0.9906   0.6756  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8545, F1=0.8033, Normal Recall=0.7961, Normal Precision=0.9950, Attack Recall=0.9906, Attack Precision=0.6756

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
0.15       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633   <--
0.20       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.25       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.30       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.35       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.40       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.45       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.50       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.55       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.60       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.65       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.70       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.75       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
0.80       0.8734   0.8623   0.7953   0.9922   0.9906   0.7633  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8734, F1=0.8623, Normal Recall=0.7953, Normal Precision=0.9922, Attack Recall=0.9906, Attack Precision=0.7633

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
0.15       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277   <--
0.20       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.25       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.30       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.35       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.40       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.45       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.50       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.55       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.60       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.65       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.70       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.75       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
0.80       0.8922   0.9019   0.7938   0.9883   0.9906   0.8277  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8922, F1=0.9019, Normal Recall=0.7938, Normal Precision=0.9883, Attack Recall=0.9906, Attack Precision=0.8277

```

