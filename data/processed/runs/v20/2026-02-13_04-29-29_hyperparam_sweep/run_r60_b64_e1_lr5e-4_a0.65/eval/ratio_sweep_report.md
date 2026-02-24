# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-13 09:58:53 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 60 |
| **Local epochs** | 1 |
| **Batch size** | 64 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0670 | 0.1606 | 0.2541 | 0.3475 | 0.4401 | 0.5337 | 0.6273 | 0.7200 | 0.8134 | 0.9068 | 1.0000 |
| QAT+Prune only | 0.8523 | 0.8666 | 0.8790 | 0.8920 | 0.9051 | 0.9170 | 0.9307 | 0.9426 | 0.9559 | 0.9672 | 0.9808 |
| QAT+PTQ | 0.8520 | 0.8662 | 0.8787 | 0.8916 | 0.9047 | 0.9167 | 0.9302 | 0.9422 | 0.9555 | 0.9668 | 0.9804 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8520 | 0.8662 | 0.8787 | 0.8916 | 0.9047 | 0.9167 | 0.9302 | 0.9422 | 0.9555 | 0.9668 | 0.9804 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1924 | 0.3491 | 0.4790 | 0.5883 | 0.6820 | 0.7630 | 0.8333 | 0.8955 | 0.9508 | 1.0000 |
| QAT+Prune only | 0.0000 | 0.5953 | 0.7643 | 0.8450 | 0.8921 | 0.9220 | 0.9444 | 0.9599 | 0.9727 | 0.9818 | 0.9903 |
| QAT+PTQ | 0.0000 | 0.5945 | 0.7637 | 0.8444 | 0.8916 | 0.9217 | 0.9440 | 0.9596 | 0.9724 | 0.9815 | 0.9901 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5945 | 0.7637 | 0.8444 | 0.8916 | 0.9217 | 0.9440 | 0.9596 | 0.9724 | 0.9815 | 0.9901 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0670 | 0.0674 | 0.0676 | 0.0679 | 0.0668 | 0.0674 | 0.0682 | 0.0669 | 0.0671 | 0.0687 | 0.0000 |
| QAT+Prune only | 0.8523 | 0.8538 | 0.8536 | 0.8540 | 0.8547 | 0.8532 | 0.8555 | 0.8536 | 0.8564 | 0.8454 | 0.0000 |
| QAT+PTQ | 0.8520 | 0.8534 | 0.8533 | 0.8536 | 0.8542 | 0.8531 | 0.8549 | 0.8531 | 0.8558 | 0.8448 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8520 | 0.8534 | 0.8533 | 0.8536 | 0.8542 | 0.8531 | 0.8549 | 0.8531 | 0.8558 | 0.8448 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0670 | 0.0000 | 0.0000 | 0.0000 | 0.0670 | 1.0000 |
| 90 | 10 | 299,940 | 0.1606 | 0.1064 | 0.9999 | 0.1924 | 0.0674 | 0.9999 |
| 80 | 20 | 291,350 | 0.2541 | 0.2114 | 1.0000 | 0.3491 | 0.0676 | 0.9999 |
| 70 | 30 | 194,230 | 0.3475 | 0.3150 | 1.0000 | 0.4790 | 0.0679 | 0.9998 |
| 60 | 40 | 145,675 | 0.4401 | 0.4167 | 1.0000 | 0.5883 | 0.0668 | 0.9997 |
| 50 | 50 | 116,540 | 0.5337 | 0.5174 | 1.0000 | 0.6820 | 0.0674 | 0.9995 |
| 40 | 60 | 97,115 | 0.6273 | 0.6168 | 1.0000 | 0.7630 | 0.0682 | 0.9992 |
| 30 | 70 | 83,240 | 0.7200 | 0.7143 | 1.0000 | 0.8333 | 0.0669 | 0.9988 |
| 20 | 80 | 72,835 | 0.8134 | 0.8109 | 1.0000 | 0.8955 | 0.0671 | 0.9980 |
| 10 | 90 | 64,740 | 0.9068 | 0.9062 | 1.0000 | 0.9508 | 0.0687 | 0.9955 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8523 | 0.0000 | 0.0000 | 0.0000 | 0.8523 | 1.0000 |
| 90 | 10 | 299,940 | 0.8666 | 0.4272 | 0.9814 | 0.5953 | 0.8538 | 0.9976 |
| 80 | 20 | 291,350 | 0.8790 | 0.6261 | 0.9808 | 0.7643 | 0.8536 | 0.9944 |
| 70 | 30 | 194,230 | 0.8920 | 0.7422 | 0.9808 | 0.8450 | 0.8540 | 0.9904 |
| 60 | 40 | 145,675 | 0.9051 | 0.8182 | 0.9808 | 0.8921 | 0.8547 | 0.9852 |
| 50 | 50 | 116,540 | 0.9170 | 0.8698 | 0.9808 | 0.9220 | 0.8532 | 0.9779 |
| 40 | 60 | 97,115 | 0.9307 | 0.9106 | 0.9808 | 0.9444 | 0.8555 | 0.9674 |
| 30 | 70 | 83,240 | 0.9426 | 0.9399 | 0.9808 | 0.9599 | 0.8536 | 0.9500 |
| 20 | 80 | 72,835 | 0.9559 | 0.9647 | 0.9808 | 0.9727 | 0.8564 | 0.9176 |
| 10 | 90 | 64,740 | 0.9672 | 0.9828 | 0.9808 | 0.9818 | 0.8454 | 0.8300 |
| 0 | 100 | 58,270 | 0.9808 | 1.0000 | 0.9808 | 0.9903 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8520 | 0.0000 | 0.0000 | 0.0000 | 0.8520 | 1.0000 |
| 90 | 10 | 299,940 | 0.8662 | 0.4265 | 0.9808 | 0.5945 | 0.8534 | 0.9975 |
| 80 | 20 | 291,350 | 0.8787 | 0.6255 | 0.9804 | 0.7637 | 0.8533 | 0.9943 |
| 70 | 30 | 194,230 | 0.8916 | 0.7416 | 0.9803 | 0.8444 | 0.8536 | 0.9902 |
| 60 | 40 | 145,675 | 0.9047 | 0.8176 | 0.9804 | 0.8916 | 0.8542 | 0.9849 |
| 50 | 50 | 116,540 | 0.9167 | 0.8697 | 0.9804 | 0.9217 | 0.8531 | 0.9775 |
| 40 | 60 | 97,115 | 0.9302 | 0.9102 | 0.9803 | 0.9440 | 0.8549 | 0.9667 |
| 30 | 70 | 83,240 | 0.9422 | 0.9396 | 0.9803 | 0.9596 | 0.8531 | 0.9490 |
| 20 | 80 | 72,835 | 0.9555 | 0.9645 | 0.9804 | 0.9724 | 0.8558 | 0.9160 |
| 10 | 90 | 64,740 | 0.9668 | 0.9827 | 0.9803 | 0.9815 | 0.8448 | 0.8269 |
| 0 | 100 | 58,270 | 0.9804 | 1.0000 | 0.9804 | 0.9901 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8520 | 0.0000 | 0.0000 | 0.0000 | 0.8520 | 1.0000 |
| 90 | 10 | 299,940 | 0.8662 | 0.4265 | 0.9808 | 0.5945 | 0.8534 | 0.9975 |
| 80 | 20 | 291,350 | 0.8787 | 0.6255 | 0.9804 | 0.7637 | 0.8533 | 0.9943 |
| 70 | 30 | 194,230 | 0.8916 | 0.7416 | 0.9803 | 0.8444 | 0.8536 | 0.9902 |
| 60 | 40 | 145,675 | 0.9047 | 0.8176 | 0.9804 | 0.8916 | 0.8542 | 0.9849 |
| 50 | 50 | 116,540 | 0.9167 | 0.8697 | 0.9804 | 0.9217 | 0.8531 | 0.9775 |
| 40 | 60 | 97,115 | 0.9302 | 0.9102 | 0.9803 | 0.9440 | 0.8549 | 0.9667 |
| 30 | 70 | 83,240 | 0.9422 | 0.9396 | 0.9803 | 0.9596 | 0.8531 | 0.9490 |
| 20 | 80 | 72,835 | 0.9555 | 0.9645 | 0.9804 | 0.9724 | 0.8558 | 0.9160 |
| 10 | 90 | 64,740 | 0.9668 | 0.9827 | 0.9803 | 0.9815 | 0.8448 | 0.8269 |
| 0 | 100 | 58,270 | 0.9804 | 1.0000 | 0.9804 | 0.9901 | 0.0000 | 0.0000 |


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
0.15       0.1606   0.1924   0.0674   0.9999   1.0000   0.1065   <--
0.20       0.1606   0.1924   0.0674   0.9999   1.0000   0.1065  
0.25       0.1606   0.1924   0.0674   0.9999   1.0000   0.1065  
0.30       0.1606   0.1924   0.0674   0.9999   1.0000   0.1065  
0.35       0.1606   0.1924   0.0674   0.9999   1.0000   0.1065  
0.40       0.1606   0.1924   0.0674   0.9999   1.0000   0.1065  
0.45       0.1606   0.1924   0.0674   0.9999   1.0000   0.1065  
0.50       0.1606   0.1924   0.0674   0.9999   1.0000   0.1065  
0.55       0.1606   0.1924   0.0674   0.9999   1.0000   0.1065  
0.60       0.1606   0.1924   0.0674   0.9999   1.0000   0.1065  
0.65       0.1606   0.1924   0.0674   0.9999   1.0000   0.1065  
0.70       0.1606   0.1924   0.0674   0.9999   1.0000   0.1065  
0.75       0.1606   0.1924   0.0674   0.9999   1.0000   0.1065  
0.80       0.1606   0.1924   0.0674   0.9999   1.0000   0.1065  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.1606, F1=0.1924, Normal Recall=0.0674, Normal Precision=0.9999, Attack Recall=1.0000, Attack Precision=0.1065

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
0.15       0.2538   0.3490   0.0672   0.9999   1.0000   0.2114   <--
0.20       0.2538   0.3490   0.0672   0.9999   1.0000   0.2114  
0.25       0.2538   0.3490   0.0672   0.9999   1.0000   0.2114  
0.30       0.2538   0.3490   0.0672   0.9999   1.0000   0.2114  
0.35       0.2538   0.3490   0.0672   0.9999   1.0000   0.2114  
0.40       0.2538   0.3490   0.0672   0.9999   1.0000   0.2114  
0.45       0.2538   0.3490   0.0672   0.9999   1.0000   0.2114  
0.50       0.2538   0.3490   0.0672   0.9999   1.0000   0.2114  
0.55       0.2538   0.3490   0.0672   0.9999   1.0000   0.2114  
0.60       0.2538   0.3490   0.0672   0.9999   1.0000   0.2114  
0.65       0.2538   0.3490   0.0672   0.9999   1.0000   0.2114  
0.70       0.2538   0.3490   0.0672   0.9999   1.0000   0.2114  
0.75       0.2538   0.3490   0.0672   0.9999   1.0000   0.2114  
0.80       0.2538   0.3490   0.0672   0.9999   1.0000   0.2114  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2538, F1=0.3490, Normal Recall=0.0672, Normal Precision=0.9999, Attack Recall=1.0000, Attack Precision=0.2114

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
0.15       0.3474   0.4790   0.0677   0.9998   1.0000   0.3149   <--
0.20       0.3474   0.4790   0.0677   0.9998   1.0000   0.3149  
0.25       0.3474   0.4790   0.0677   0.9998   1.0000   0.3149  
0.30       0.3474   0.4790   0.0677   0.9998   1.0000   0.3149  
0.35       0.3474   0.4790   0.0677   0.9998   1.0000   0.3149  
0.40       0.3474   0.4790   0.0677   0.9998   1.0000   0.3149  
0.45       0.3474   0.4790   0.0677   0.9998   1.0000   0.3149  
0.50       0.3474   0.4790   0.0677   0.9998   1.0000   0.3149  
0.55       0.3474   0.4790   0.0677   0.9998   1.0000   0.3149  
0.60       0.3474   0.4790   0.0677   0.9998   1.0000   0.3149  
0.65       0.3474   0.4790   0.0677   0.9998   1.0000   0.3149  
0.70       0.3474   0.4790   0.0677   0.9998   1.0000   0.3149  
0.75       0.3474   0.4790   0.0677   0.9998   1.0000   0.3149  
0.80       0.3474   0.4790   0.0677   0.9998   1.0000   0.3149  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3474, F1=0.4790, Normal Recall=0.0677, Normal Precision=0.9998, Attack Recall=1.0000, Attack Precision=0.3149

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
0.15       0.4406   0.5885   0.0677   0.9997   1.0000   0.4169   <--
0.20       0.4406   0.5885   0.0677   0.9997   1.0000   0.4169  
0.25       0.4406   0.5885   0.0677   0.9997   1.0000   0.4169  
0.30       0.4406   0.5885   0.0677   0.9997   1.0000   0.4169  
0.35       0.4406   0.5885   0.0677   0.9997   1.0000   0.4169  
0.40       0.4406   0.5885   0.0677   0.9997   1.0000   0.4169  
0.45       0.4406   0.5885   0.0677   0.9997   1.0000   0.4169  
0.50       0.4406   0.5885   0.0677   0.9997   1.0000   0.4169  
0.55       0.4406   0.5885   0.0677   0.9997   1.0000   0.4169  
0.60       0.4406   0.5885   0.0677   0.9997   1.0000   0.4169  
0.65       0.4406   0.5885   0.0677   0.9997   1.0000   0.4169  
0.70       0.4406   0.5885   0.0677   0.9997   1.0000   0.4169  
0.75       0.4406   0.5885   0.0677   0.9997   1.0000   0.4169  
0.80       0.4406   0.5885   0.0677   0.9997   1.0000   0.4169  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4406, F1=0.5885, Normal Recall=0.0677, Normal Precision=0.9997, Attack Recall=1.0000, Attack Precision=0.4169

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
0.15       0.5340   0.6821   0.0680   0.9995   1.0000   0.5176   <--
0.20       0.5340   0.6821   0.0680   0.9995   1.0000   0.5176  
0.25       0.5340   0.6821   0.0680   0.9995   1.0000   0.5176  
0.30       0.5340   0.6821   0.0680   0.9995   1.0000   0.5176  
0.35       0.5340   0.6821   0.0680   0.9995   1.0000   0.5176  
0.40       0.5340   0.6821   0.0680   0.9995   1.0000   0.5176  
0.45       0.5340   0.6821   0.0680   0.9995   1.0000   0.5176  
0.50       0.5340   0.6821   0.0680   0.9995   1.0000   0.5176  
0.55       0.5340   0.6821   0.0680   0.9995   1.0000   0.5176  
0.60       0.5340   0.6821   0.0680   0.9995   1.0000   0.5176  
0.65       0.5340   0.6821   0.0680   0.9995   1.0000   0.5176  
0.70       0.5340   0.6821   0.0680   0.9995   1.0000   0.5176  
0.75       0.5340   0.6821   0.0680   0.9995   1.0000   0.5176  
0.80       0.5340   0.6821   0.0680   0.9995   1.0000   0.5176  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5340, F1=0.6821, Normal Recall=0.0680, Normal Precision=0.9995, Attack Recall=1.0000, Attack Precision=0.5176

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
0.15       0.8665   0.5951   0.8538   0.9975   0.9810   0.4271   <--
0.20       0.8665   0.5951   0.8538   0.9975   0.9810   0.4271  
0.25       0.8665   0.5951   0.8538   0.9975   0.9810   0.4271  
0.30       0.8665   0.5951   0.8538   0.9975   0.9810   0.4271  
0.35       0.8665   0.5951   0.8538   0.9975   0.9810   0.4271  
0.40       0.8665   0.5951   0.8538   0.9975   0.9810   0.4271  
0.45       0.8665   0.5951   0.8538   0.9975   0.9810   0.4271  
0.50       0.8665   0.5951   0.8538   0.9975   0.9810   0.4271  
0.55       0.8665   0.5951   0.8538   0.9975   0.9810   0.4271  
0.60       0.8665   0.5951   0.8538   0.9975   0.9810   0.4271  
0.65       0.8665   0.5951   0.8538   0.9975   0.9810   0.4271  
0.70       0.8665   0.5951   0.8538   0.9975   0.9810   0.4271  
0.75       0.8665   0.5951   0.8538   0.9975   0.9810   0.4271  
0.80       0.8665   0.5951   0.8538   0.9975   0.9810   0.4271  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8665, F1=0.5951, Normal Recall=0.8538, Normal Precision=0.9975, Attack Recall=0.9810, Attack Precision=0.4271

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
0.15       0.8795   0.7650   0.8542   0.9944   0.9808   0.6270   <--
0.20       0.8795   0.7650   0.8542   0.9944   0.9808   0.6270  
0.25       0.8795   0.7650   0.8542   0.9944   0.9808   0.6270  
0.30       0.8795   0.7650   0.8542   0.9944   0.9808   0.6270  
0.35       0.8795   0.7650   0.8542   0.9944   0.9808   0.6270  
0.40       0.8795   0.7650   0.8542   0.9944   0.9808   0.6270  
0.45       0.8795   0.7650   0.8542   0.9944   0.9808   0.6270  
0.50       0.8795   0.7650   0.8542   0.9944   0.9808   0.6270  
0.55       0.8795   0.7650   0.8542   0.9944   0.9808   0.6270  
0.60       0.8795   0.7650   0.8542   0.9944   0.9808   0.6270  
0.65       0.8795   0.7650   0.8542   0.9944   0.9808   0.6270  
0.70       0.8795   0.7650   0.8542   0.9944   0.9808   0.6270  
0.75       0.8795   0.7650   0.8542   0.9944   0.9808   0.6270  
0.80       0.8795   0.7650   0.8542   0.9944   0.9808   0.6270  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8795, F1=0.7650, Normal Recall=0.8542, Normal Precision=0.9944, Attack Recall=0.9808, Attack Precision=0.6270

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
0.15       0.8912   0.8439   0.8528   0.9904   0.9808   0.7406   <--
0.20       0.8912   0.8439   0.8528   0.9904   0.9808   0.7406  
0.25       0.8912   0.8439   0.8528   0.9904   0.9808   0.7406  
0.30       0.8912   0.8439   0.8528   0.9904   0.9808   0.7406  
0.35       0.8912   0.8439   0.8528   0.9904   0.9808   0.7406  
0.40       0.8912   0.8439   0.8528   0.9904   0.9808   0.7406  
0.45       0.8912   0.8439   0.8528   0.9904   0.9808   0.7406  
0.50       0.8912   0.8439   0.8528   0.9904   0.9808   0.7406  
0.55       0.8912   0.8439   0.8528   0.9904   0.9808   0.7406  
0.60       0.8912   0.8439   0.8528   0.9904   0.9808   0.7406  
0.65       0.8912   0.8439   0.8528   0.9904   0.9808   0.7406  
0.70       0.8912   0.8439   0.8528   0.9904   0.9808   0.7406  
0.75       0.8912   0.8439   0.8528   0.9904   0.9808   0.7406  
0.80       0.8912   0.8439   0.8528   0.9904   0.9808   0.7406  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8912, F1=0.8439, Normal Recall=0.8528, Normal Precision=0.9904, Attack Recall=0.9808, Attack Precision=0.7406

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
0.15       0.9039   0.8908   0.8526   0.9852   0.9808   0.8160   <--
0.20       0.9039   0.8908   0.8526   0.9852   0.9808   0.8160  
0.25       0.9039   0.8908   0.8526   0.9852   0.9808   0.8160  
0.30       0.9039   0.8908   0.8526   0.9852   0.9808   0.8160  
0.35       0.9039   0.8908   0.8526   0.9852   0.9808   0.8160  
0.40       0.9039   0.8908   0.8526   0.9852   0.9808   0.8160  
0.45       0.9039   0.8908   0.8526   0.9852   0.9808   0.8160  
0.50       0.9039   0.8908   0.8526   0.9852   0.9808   0.8160  
0.55       0.9039   0.8908   0.8526   0.9852   0.9808   0.8160  
0.60       0.9039   0.8908   0.8526   0.9852   0.9808   0.8160  
0.65       0.9039   0.8908   0.8526   0.9852   0.9808   0.8160  
0.70       0.9039   0.8908   0.8526   0.9852   0.9808   0.8160  
0.75       0.9039   0.8908   0.8526   0.9852   0.9808   0.8160  
0.80       0.9039   0.8908   0.8526   0.9852   0.9808   0.8160  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9039, F1=0.8908, Normal Recall=0.8526, Normal Precision=0.9852, Attack Recall=0.9808, Attack Precision=0.8160

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
0.15       0.9169   0.9219   0.8530   0.9779   0.9808   0.8697   <--
0.20       0.9169   0.9219   0.8530   0.9779   0.9808   0.8697  
0.25       0.9169   0.9219   0.8530   0.9779   0.9808   0.8697  
0.30       0.9169   0.9219   0.8530   0.9779   0.9808   0.8697  
0.35       0.9169   0.9219   0.8530   0.9779   0.9808   0.8697  
0.40       0.9169   0.9219   0.8530   0.9779   0.9808   0.8697  
0.45       0.9169   0.9219   0.8530   0.9779   0.9808   0.8697  
0.50       0.9169   0.9219   0.8530   0.9779   0.9808   0.8697  
0.55       0.9169   0.9219   0.8530   0.9779   0.9808   0.8697  
0.60       0.9169   0.9219   0.8530   0.9779   0.9808   0.8697  
0.65       0.9169   0.9219   0.8530   0.9779   0.9808   0.8697  
0.70       0.9169   0.9219   0.8530   0.9779   0.9808   0.8697  
0.75       0.9169   0.9219   0.8530   0.9779   0.9808   0.8697  
0.80       0.9169   0.9219   0.8530   0.9779   0.9808   0.8697  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9169, F1=0.9219, Normal Recall=0.8530, Normal Precision=0.9779, Attack Recall=0.9808, Attack Precision=0.8697

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
0.15       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264   <--
0.20       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.25       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.30       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.35       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.40       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.45       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.50       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.55       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.60       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.65       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.70       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.75       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.80       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8661, F1=0.5943, Normal Recall=0.8534, Normal Precision=0.9974, Attack Recall=0.9804, Attack Precision=0.4264

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
0.15       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263   <--
0.20       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.25       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.30       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.35       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.40       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.45       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.50       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.55       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.60       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.65       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.70       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.75       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.80       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8791, F1=0.7643, Normal Recall=0.8538, Normal Precision=0.9943, Attack Recall=0.9804, Attack Precision=0.6263

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
0.15       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400   <--
0.20       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.25       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.30       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.35       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.40       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.45       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.50       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.55       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.60       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.65       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.70       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.75       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.80       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8908, F1=0.8434, Normal Recall=0.8524, Normal Precision=0.9902, Attack Recall=0.9803, Attack Precision=0.7400

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
0.15       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156   <--
0.20       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.25       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.30       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.35       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.40       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.45       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.50       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.55       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.60       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.65       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.70       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.75       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.80       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9035, F1=0.8904, Normal Recall=0.8522, Normal Precision=0.9849, Attack Recall=0.9804, Attack Precision=0.8156

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
0.15       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693   <--
0.20       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.25       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.30       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.35       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.40       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.45       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.50       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.55       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.60       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.65       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.70       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.75       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.80       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9165, F1=0.9215, Normal Recall=0.8526, Normal Precision=0.9775, Attack Recall=0.9804, Attack Precision=0.8693

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
0.15       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264   <--
0.20       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.25       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.30       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.35       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.40       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.45       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.50       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.55       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.60       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.65       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.70       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.75       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
0.80       0.8661   0.5943   0.8534   0.9974   0.9804   0.4264  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8661, F1=0.5943, Normal Recall=0.8534, Normal Precision=0.9974, Attack Recall=0.9804, Attack Precision=0.4264

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
0.15       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263   <--
0.20       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.25       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.30       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.35       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.40       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.45       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.50       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.55       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.60       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.65       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.70       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.75       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
0.80       0.8791   0.7643   0.8538   0.9943   0.9804   0.6263  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8791, F1=0.7643, Normal Recall=0.8538, Normal Precision=0.9943, Attack Recall=0.9804, Attack Precision=0.6263

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
0.15       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400   <--
0.20       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.25       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.30       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.35       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.40       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.45       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.50       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.55       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.60       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.65       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.70       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.75       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
0.80       0.8908   0.8434   0.8524   0.9902   0.9803   0.7400  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8908, F1=0.8434, Normal Recall=0.8524, Normal Precision=0.9902, Attack Recall=0.9803, Attack Precision=0.7400

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
0.15       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156   <--
0.20       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.25       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.30       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.35       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.40       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.45       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.50       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.55       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.60       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.65       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.70       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.75       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
0.80       0.9035   0.8904   0.8522   0.9849   0.9804   0.8156  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9035, F1=0.8904, Normal Recall=0.8522, Normal Precision=0.9849, Attack Recall=0.9804, Attack Precision=0.8156

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
0.15       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693   <--
0.20       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.25       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.30       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.35       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.40       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.45       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.50       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.55       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.60       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.65       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.70       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.75       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
0.80       0.9165   0.9215   0.8526   0.9775   0.9804   0.8693  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9165, F1=0.9215, Normal Recall=0.8526, Normal Precision=0.9775, Attack Recall=0.9804, Attack Precision=0.8693

```

