# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-24 15:07:49 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 1 |
| **Local epochs** | 2 |
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0003 | 0.1004 | 0.2003 | 0.3003 | 0.4003 | 0.5002 | 0.6002 | 0.7001 | 0.8000 | 0.9000 | 1.0000 |
| noQAT+PTQ | 0.0000 | 0.1000 | 0.2000 | 0.3000 | 0.4000 | 0.5000 | 0.6000 | 0.7000 | 0.8000 | 0.9000 | 1.0000 |
| saved_model_traditional_qat | 0.9516 | 0.9526 | 0.9530 | 0.9540 | 0.9548 | 0.9551 | 0.9562 | 0.9568 | 0.9571 | 0.9579 | 0.9586 |
| QAT+PTQ | 0.9564 | 0.9546 | 0.9516 | 0.9491 | 0.9466 | 0.9436 | 0.9410 | 0.9379 | 0.9352 | 0.9325 | 0.9296 |
| Compressed (QAT) | 0.9720 | 0.9679 | 0.9632 | 0.9588 | 0.9538 | 0.9494 | 0.9446 | 0.9395 | 0.9350 | 0.9303 | 0.9257 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1819 | 0.3334 | 0.4617 | 0.5716 | 0.6668 | 0.7501 | 0.8236 | 0.8889 | 0.9474 | 1.0000 |
| noQAT+PTQ | 0.0000 | 0.1818 | 0.3333 | 0.4615 | 0.5714 | 0.6667 | 0.7500 | 0.8235 | 0.8889 | 0.9474 | 1.0000 |
| saved_model_traditional_qat | 0.0000 | 0.8017 | 0.8909 | 0.9259 | 0.9444 | 0.9552 | 0.9633 | 0.9688 | 0.9728 | 0.9762 | 0.9789 |
| QAT+PTQ | 0.0000 | 0.8039 | 0.8848 | 0.9164 | 0.9330 | 0.9428 | 0.9497 | 0.9544 | 0.9582 | 0.9612 | 0.9635 |
| Compressed (QAT) | 0.0000 | 0.8523 | 0.9096 | 0.9310 | 0.9413 | 0.9481 | 0.9525 | 0.9554 | 0.9580 | 0.9599 | 0.9614 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0003 | 0.0004 | 0.0004 | 0.0005 | 0.0005 | 0.0004 | 0.0004 | 0.0004 | 0.0001 | 0.0002 | 0.0000 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| saved_model_traditional_qat | 0.9516 | 0.9519 | 0.9516 | 0.9520 | 0.9523 | 0.9516 | 0.9525 | 0.9527 | 0.9513 | 0.9513 | 0.0000 |
| QAT+PTQ | 0.9564 | 0.9573 | 0.9571 | 0.9575 | 0.9579 | 0.9575 | 0.9579 | 0.9570 | 0.9572 | 0.9581 | 0.0000 |
| Compressed (QAT) | 0.9720 | 0.9726 | 0.9726 | 0.9731 | 0.9726 | 0.9731 | 0.9730 | 0.9718 | 0.9723 | 0.9722 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0003 | 1.0000 |
| 90 | 10 | 299,940 | 0.1004 | 0.1000 | 1.0000 | 0.1819 | 0.0004 | 1.0000 |
| 80 | 20 | 291,350 | 0.2003 | 0.2001 | 1.0000 | 0.3334 | 0.0004 | 1.0000 |
| 70 | 30 | 194,230 | 0.3003 | 0.3001 | 1.0000 | 0.4617 | 0.0005 | 1.0000 |
| 60 | 40 | 145,675 | 0.4003 | 0.4001 | 1.0000 | 0.5716 | 0.0005 | 1.0000 |
| 50 | 50 | 116,540 | 0.5002 | 0.5001 | 1.0000 | 0.6668 | 0.0004 | 1.0000 |
| 40 | 60 | 97,115 | 0.6002 | 0.6001 | 1.0000 | 0.7501 | 0.0004 | 1.0000 |
| 30 | 70 | 83,240 | 0.7001 | 0.7001 | 1.0000 | 0.8236 | 0.0004 | 1.0000 |
| 20 | 80 | 72,835 | 0.8000 | 0.8000 | 1.0000 | 0.8889 | 0.0001 | 1.0000 |
| 10 | 90 | 64,740 | 0.9000 | 0.9000 | 1.0000 | 0.9474 | 0.0002 | 1.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### noQAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 299,940 | 0.1000 | 0.1000 | 1.0000 | 0.1818 | 0.0000 | 0.0000 |
| 80 | 20 | 291,350 | 0.2000 | 0.2000 | 1.0000 | 0.3333 | 0.0000 | 0.0000 |
| 70 | 30 | 194,230 | 0.3000 | 0.3000 | 1.0000 | 0.4615 | 0.0000 | 0.0000 |
| 60 | 40 | 145,675 | 0.4000 | 0.4000 | 1.0000 | 0.5714 | 0.0000 | 0.0000 |
| 50 | 50 | 116,540 | 0.5000 | 0.5000 | 1.0000 | 0.6667 | 0.0000 | 0.0000 |
| 40 | 60 | 97,115 | 0.6000 | 0.6000 | 1.0000 | 0.7500 | 0.0000 | 0.0000 |
| 30 | 70 | 83,240 | 0.7000 | 0.7000 | 1.0000 | 0.8235 | 0.0000 | 0.0000 |
| 20 | 80 | 72,835 | 0.8000 | 0.8000 | 1.0000 | 0.8889 | 0.0000 | 0.0000 |
| 10 | 90 | 64,740 | 0.9000 | 0.9000 | 1.0000 | 0.9474 | 0.0000 | 0.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### saved_model_traditional_qat

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9516 | 0.0000 | 0.0000 | 0.0000 | 0.9516 | 1.0000 |
| 90 | 10 | 299,940 | 0.9526 | 0.6888 | 0.9589 | 0.8017 | 0.9519 | 0.9952 |
| 80 | 20 | 291,350 | 0.9530 | 0.8321 | 0.9586 | 0.8909 | 0.9516 | 0.9892 |
| 70 | 30 | 194,230 | 0.9540 | 0.8954 | 0.9586 | 0.9259 | 0.9520 | 0.9817 |
| 60 | 40 | 145,675 | 0.9548 | 0.9306 | 0.9586 | 0.9444 | 0.9523 | 0.9718 |
| 50 | 50 | 116,540 | 0.9551 | 0.9519 | 0.9586 | 0.9552 | 0.9516 | 0.9583 |
| 40 | 60 | 97,115 | 0.9562 | 0.9680 | 0.9586 | 0.9633 | 0.9525 | 0.9388 |
| 30 | 70 | 83,240 | 0.9568 | 0.9793 | 0.9586 | 0.9688 | 0.9527 | 0.9079 |
| 20 | 80 | 72,835 | 0.9571 | 0.9874 | 0.9586 | 0.9728 | 0.9513 | 0.8517 |
| 10 | 90 | 64,740 | 0.9579 | 0.9944 | 0.9586 | 0.9762 | 0.9513 | 0.7185 |
| 0 | 100 | 58,270 | 0.9586 | 1.0000 | 0.9586 | 0.9789 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9564 | 0.0000 | 0.0000 | 0.0000 | 0.9564 | 1.0000 |
| 90 | 10 | 299,940 | 0.9546 | 0.7077 | 0.9303 | 0.8039 | 0.9573 | 0.9920 |
| 80 | 20 | 291,350 | 0.9516 | 0.8441 | 0.9296 | 0.8848 | 0.9571 | 0.9820 |
| 70 | 30 | 194,230 | 0.9491 | 0.9036 | 0.9296 | 0.9164 | 0.9575 | 0.9695 |
| 60 | 40 | 145,675 | 0.9466 | 0.9365 | 0.9296 | 0.9330 | 0.9579 | 0.9533 |
| 50 | 50 | 116,540 | 0.9436 | 0.9563 | 0.9296 | 0.9428 | 0.9575 | 0.9315 |
| 40 | 60 | 97,115 | 0.9410 | 0.9707 | 0.9296 | 0.9497 | 0.9579 | 0.9008 |
| 30 | 70 | 83,240 | 0.9379 | 0.9806 | 0.9296 | 0.9544 | 0.9570 | 0.8536 |
| 20 | 80 | 72,835 | 0.9352 | 0.9886 | 0.9297 | 0.9582 | 0.9572 | 0.7728 |
| 10 | 90 | 64,740 | 0.9325 | 0.9950 | 0.9297 | 0.9612 | 0.9581 | 0.6021 |
| 0 | 100 | 58,270 | 0.9296 | 1.0000 | 0.9296 | 0.9635 | 0.0000 | 0.0000 |

### Compressed (QAT)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9720 | 0.0000 | 0.0000 | 0.0000 | 0.9720 | 1.0000 |
| 90 | 10 | 299,940 | 0.9679 | 0.7897 | 0.9257 | 0.8523 | 0.9726 | 0.9916 |
| 80 | 20 | 291,350 | 0.9632 | 0.8940 | 0.9257 | 0.9096 | 0.9726 | 0.9813 |
| 70 | 30 | 194,230 | 0.9588 | 0.9364 | 0.9257 | 0.9310 | 0.9731 | 0.9683 |
| 60 | 40 | 145,675 | 0.9538 | 0.9575 | 0.9257 | 0.9413 | 0.9726 | 0.9515 |
| 50 | 50 | 116,540 | 0.9494 | 0.9717 | 0.9257 | 0.9481 | 0.9731 | 0.9290 |
| 40 | 60 | 97,115 | 0.9446 | 0.9809 | 0.9257 | 0.9525 | 0.9730 | 0.8972 |
| 30 | 70 | 83,240 | 0.9395 | 0.9871 | 0.9257 | 0.9554 | 0.9718 | 0.8486 |
| 20 | 80 | 72,835 | 0.9350 | 0.9926 | 0.9257 | 0.9580 | 0.9723 | 0.7659 |
| 10 | 90 | 64,740 | 0.9303 | 0.9967 | 0.9257 | 0.9599 | 0.9722 | 0.5924 |
| 0 | 100 | 58,270 | 0.9257 | 1.0000 | 0.9257 | 0.9614 | 0.0000 | 0.0000 |


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
0.15       0.1004   0.1819   0.0004   1.0000   1.0000   0.1000  
0.20       0.1004   0.1819   0.0004   1.0000   1.0000   0.1000  
0.25       0.1004   0.1819   0.0004   1.0000   1.0000   0.1000  
0.30       0.1004   0.1819   0.0004   1.0000   1.0000   0.1000  
0.35       0.1007   0.1819   0.0007   0.9951   1.0000   0.1001  
0.40       0.1049   0.1826   0.0054   0.9980   0.9999   0.1005  
0.45       0.1151   0.1842   0.0168   0.9950   0.9992   0.1015  
0.50       0.4921   0.2073   0.4729   0.9269   0.6643   0.1228  
0.55       0.8279   0.2967   0.8796   0.9255   0.3630   0.2509   <--
0.60       0.8720   0.1523   0.9561   0.9067   0.1149   0.2255  
0.65       0.8777   0.0119   0.9744   0.8983   0.0074   0.0310  
0.70       0.8877   0.0092   0.9858   0.8992   0.0052   0.0390  
0.75       0.8930   0.0068   0.9918   0.8996   0.0037   0.0475  
0.80       0.8981   0.0007   0.9978   0.8998   0.0003   0.0168  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.8279, F1=0.2967, Normal Recall=0.8796, Normal Precision=0.9255, Attack Recall=0.3630, Attack Precision=0.2509

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
0.15       0.2003   0.3334   0.0004   1.0000   1.0000   0.2001  
0.20       0.2003   0.3334   0.0004   1.0000   1.0000   0.2001  
0.25       0.2003   0.3334   0.0004   1.0000   1.0000   0.2001  
0.30       0.2004   0.3334   0.0004   1.0000   1.0000   0.2001  
0.35       0.2006   0.3335   0.0008   0.9944   1.0000   0.2001  
0.40       0.2042   0.3345   0.0053   0.9968   0.9999   0.2008  
0.45       0.2132   0.3369   0.0167   0.9891   0.9993   0.2026  
0.50       0.5121   0.3536   0.4734   0.8505   0.6671   0.2405  
0.55       0.7768   0.3955   0.8797   0.8472   0.3651   0.4315   <--
0.60       0.7881   0.1796   0.9562   0.8122   0.1159   0.3980  
0.65       0.7810   0.0128   0.9744   0.7970   0.0071   0.0649  
0.70       0.7897   0.0101   0.9858   0.7986   0.0054   0.0868  
0.75       0.7943   0.0076   0.9918   0.7993   0.0039   0.1074  
0.80       0.7984   0.0011   0.9978   0.7997   0.0005   0.0578  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.7768, F1=0.3955, Normal Recall=0.8797, Normal Precision=0.8472, Attack Recall=0.3651, Attack Precision=0.4315

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
0.15       0.3003   0.4616   0.0004   1.0000   1.0000   0.3001  
0.20       0.3003   0.4616   0.0004   1.0000   1.0000   0.3001  
0.25       0.3003   0.4616   0.0004   1.0000   1.0000   0.3001  
0.30       0.3003   0.4616   0.0004   1.0000   1.0000   0.3001  
0.35       0.3005   0.4617   0.0008   0.9904   1.0000   0.3002  
0.40       0.3038   0.4629   0.0054   0.9946   0.9999   0.3011  
0.45       0.3116   0.4655   0.0170   0.9817   0.9993   0.3034   <--
0.50       0.5308   0.4604   0.4724   0.7681   0.6671   0.3514  
0.55       0.7255   0.4438   0.8799   0.7638   0.3651   0.5658  
0.60       0.7040   0.1903   0.9561   0.7162   0.1159   0.5308  
0.65       0.6844   0.0133   0.9747   0.6961   0.0071   0.1074  
0.70       0.6917   0.0104   0.9858   0.6981   0.0054   0.1403  
0.75       0.6956   0.0077   0.9920   0.6991   0.0039   0.1740  
0.80       0.6988   0.0011   0.9980   0.6997   0.0005   0.1023  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.3116, F1=0.4655, Normal Recall=0.0170, Normal Precision=0.9817, Attack Recall=0.9993, Attack Precision=0.3034

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
0.15       0.4002   0.5715   0.0003   1.0000   1.0000   0.4001  
0.20       0.4002   0.5715   0.0003   1.0000   1.0000   0.4001  
0.25       0.4002   0.5715   0.0003   1.0000   1.0000   0.4001  
0.30       0.4002   0.5715   0.0003   1.0000   1.0000   0.4001  
0.35       0.4004   0.5716   0.0007   0.9844   1.0000   0.4002  
0.40       0.4032   0.5727   0.0054   0.9917   0.9999   0.4013  
0.45       0.4098   0.5753   0.0167   0.9715   0.9993   0.4039   <--
0.50       0.5503   0.5427   0.4724   0.6804   0.6671   0.4574  
0.55       0.6743   0.4728   0.8804   0.6753   0.3651   0.6705  
0.60       0.6200   0.1962   0.9561   0.6186   0.1159   0.6377  
0.65       0.5878   0.0136   0.9749   0.5956   0.0071   0.1588  
0.70       0.5938   0.0105   0.9860   0.5979   0.0054   0.2046  
0.75       0.5968   0.0077   0.9920   0.5990   0.0039   0.2473  
0.80       0.5990   0.0011   0.9980   0.5996   0.0005   0.1483  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.4098, F1=0.5753, Normal Recall=0.0167, Normal Precision=0.9715, Attack Recall=0.9993, Attack Precision=0.4039

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
0.15       0.5002   0.6667   0.0003   1.0000   1.0000   0.5001  
0.20       0.5002   0.6667   0.0003   1.0000   1.0000   0.5001  
0.25       0.5002   0.6667   0.0003   1.0000   1.0000   0.5001  
0.30       0.5002   0.6668   0.0004   1.0000   1.0000   0.5001  
0.35       0.5004   0.6668   0.0008   0.9792   1.0000   0.5002  
0.40       0.5027   0.6678   0.0054   0.9875   0.9999   0.5013  
0.45       0.5079   0.6700   0.0165   0.9573   0.9993   0.5040   <--
0.50       0.5690   0.6075   0.4709   0.5859   0.6671   0.5577  
0.55       0.6229   0.4919   0.8807   0.5811   0.3651   0.7538  
0.60       0.5361   0.2000   0.9563   0.5196   0.1159   0.7262  
0.65       0.4912   0.0138   0.9754   0.4955   0.0071   0.2239  
0.70       0.4959   0.0106   0.9865   0.4980   0.0054   0.2852  
0.75       0.4981   0.0078   0.9923   0.4991   0.0039   0.3383  
0.80       0.4993   0.0011   0.9980   0.4996   0.0005   0.2095  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.5079, F1=0.6700, Normal Recall=0.0165, Normal Precision=0.9573, Attack Recall=0.9993, Attack Precision=0.5040

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
0.15       0.1000   0.1818   0.0000   0.0000   1.0000   0.1000  
0.20       0.1000   0.1818   0.0000   0.0000   1.0000   0.1000  
0.25       0.1000   0.1818   0.0000   0.0000   1.0000   0.1000  
0.30       0.1000   0.1818   0.0000   0.0000   1.0000   0.1000  
0.35       0.1000   0.1818   0.0000   0.0000   1.0000   0.1000  
0.40       0.1009   0.1820   0.0010   1.0000   1.0000   0.1001  
0.45       0.1043   0.1825   0.0047   1.0000   1.0000   0.1004  
0.50       0.9477   0.6480   0.9995   0.9455   0.4814   0.9909   <--
0.55       0.9472   0.6419   0.9999   0.9447   0.4730   0.9985  
0.60       0.9466   0.6358   1.0000   0.9440   0.4662   0.9991  
0.65       0.9461   0.6313   1.0000   0.9435   0.4614   0.9994  
0.70       0.9454   0.6249   1.0000   0.9429   0.4545   0.9996  
0.75       0.9434   0.6050   1.0000   0.9408   0.4337   0.9996  
0.80       0.9425   0.5969   1.0000   0.9400   0.4255   0.9997  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.9477, F1=0.6480, Normal Recall=0.9995, Normal Precision=0.9455, Attack Recall=0.4814, Attack Precision=0.9909

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
0.15       0.2000   0.3333   0.0000   0.0000   1.0000   0.2000  
0.20       0.2000   0.3333   0.0000   0.0000   1.0000   0.2000  
0.25       0.2000   0.3333   0.0000   0.0000   1.0000   0.2000  
0.30       0.2000   0.3333   0.0000   0.0000   1.0000   0.2000  
0.35       0.2000   0.3333   0.0000   0.0000   1.0000   0.2000  
0.40       0.2008   0.3336   0.0010   1.0000   1.0000   0.2002  
0.45       0.2038   0.3344   0.0048   0.9991   1.0000   0.2008  
0.50       0.8958   0.6487   0.9995   0.8851   0.4810   0.9959   <--
0.55       0.8945   0.6419   0.9999   0.8835   0.4728   0.9993  
0.60       0.8932   0.6358   1.0000   0.8822   0.4661   0.9996  
0.65       0.8923   0.6314   1.0000   0.8813   0.4614   0.9997  
0.70       0.8910   0.6254   1.0000   0.8801   0.4550   0.9998  
0.75       0.8868   0.6056   1.0000   0.8761   0.4343   0.9998  
0.80       0.8852   0.5976   1.0000   0.8745   0.4261   0.9998  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.8958, F1=0.6487, Normal Recall=0.9995, Normal Precision=0.8851, Attack Recall=0.4810, Attack Precision=0.9959

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
0.15       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.20       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.25       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.30       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.35       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.40       0.3007   0.4618   0.0010   1.0000   1.0000   0.3002  
0.45       0.3033   0.4627   0.0048   0.9985   1.0000   0.3010  
0.50       0.8439   0.6490   0.9995   0.8180   0.4810   0.9975   <--
0.55       0.8418   0.6420   0.9999   0.8157   0.4728   0.9996  
0.60       0.8398   0.6358   1.0000   0.8138   0.4661   0.9998  
0.65       0.8384   0.6314   1.0000   0.8125   0.4614   0.9998  
0.70       0.8365   0.6254   1.0000   0.8106   0.4550   0.9998  
0.75       0.8303   0.6056   1.0000   0.8049   0.4343   0.9998  
0.80       0.8278   0.5976   1.0000   0.8026   0.4261   0.9998  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.8439, F1=0.6490, Normal Recall=0.9995, Normal Precision=0.8180, Attack Recall=0.4810, Attack Precision=0.9975

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
0.15       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.20       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.25       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.30       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.35       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.40       0.4007   0.5717   0.0011   1.0000   1.0000   0.4003  
0.45       0.4029   0.5726   0.0048   0.9976   1.0000   0.4012  
0.50       0.7921   0.6493   0.9996   0.7429   0.4810   0.9986   <--
0.55       0.7891   0.6420   1.0000   0.7399   0.4728   0.9999  
0.60       0.7864   0.6359   1.0000   0.7375   0.4661   0.9999  
0.65       0.7846   0.6315   1.0000   0.7358   0.4614   0.9999  
0.70       0.7820   0.6254   1.0000   0.7335   0.4550   0.9999  
0.75       0.7737   0.6056   1.0000   0.7261   0.4343   0.9999  
0.80       0.7704   0.5976   1.0000   0.7233   0.4261   1.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.7921, F1=0.6493, Normal Recall=0.9996, Normal Precision=0.7429, Attack Recall=0.4810, Attack Precision=0.9986

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
0.15       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.20       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.25       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.30       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.35       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.40       0.5005   0.6669   0.0010   1.0000   1.0000   0.5003  
0.45       0.5025   0.6678   0.0049   0.9965   1.0000   0.5012   <--
0.50       0.7403   0.6494   0.9996   0.6582   0.4810   0.9991  
0.55       0.7364   0.6420   0.9999   0.6548   0.4728   0.9999  
0.60       0.7331   0.6359   1.0000   0.6519   0.4661   1.0000  
0.65       0.7307   0.6315   1.0000   0.6499   0.4614   1.0000  
0.70       0.7275   0.6254   1.0000   0.6472   0.4550   1.0000  
0.75       0.7171   0.6056   1.0000   0.6387   0.4343   1.0000  
0.80       0.7131   0.5976   1.0000   0.6354   0.4261   1.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.5025, F1=0.6678, Normal Recall=0.0049, Normal Precision=0.9965, Attack Recall=1.0000, Attack Precision=0.5012

```


## Threshold Tuning (saved_model_traditional_qat)

Model: `models/tflite/saved_model_traditional_qat.tflite`

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
Loading model: models/tflite/saved_model_traditional_qat.tflite

Test set: Normal 90% : Attack 10%  (n=299,940, N=269,946, A=29,994)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9369   0.7567   0.9319   0.9978   0.9819   0.6155  
0.20       0.9442   0.7775   0.9409   0.9970   0.9742   0.6468  
0.25       0.9516   0.7989   0.9505   0.9955   0.9617   0.6832  
0.30       0.9527   0.8025   0.9519   0.9954   0.9605   0.6892  
0.35       0.9648   0.8418   0.9680   0.9927   0.9363   0.7646  
0.40       0.9705   0.8638   0.9745   0.9926   0.9346   0.8030  
0.45       0.9708   0.8643   0.9753   0.9921   0.9303   0.8070   <--
0.50       0.9570   0.7839   0.9766   0.9756   0.7806   0.7872  
0.55       0.9578   0.7828   0.9798   0.9735   0.7599   0.8071  
0.60       0.9611   0.7902   0.9864   0.9708   0.7334   0.8566  
0.65       0.9648   0.7991   0.9942   0.9676   0.7000   0.9308  
0.70       0.9640   0.7935   0.9944   0.9666   0.6909   0.9317  
0.75       0.9626   0.7796   0.9960   0.9637   0.6621   0.9479  
0.80       0.9615   0.7676   0.9976   0.9611   0.6362   0.9673  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9708, F1=0.8643, Normal Recall=0.9753, Normal Precision=0.9921, Attack Recall=0.9303, Attack Precision=0.8070

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
Loading model: models/tflite/saved_model_traditional_qat.tflite

Test set: Normal 80% : Attack 20%  (n=291,350, N=233,080, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9417   0.8706   0.9318   0.9950   0.9811   0.7825  
0.20       0.9473   0.8808   0.9410   0.9928   0.9727   0.8048  
0.25       0.9524   0.8897   0.9505   0.9895   0.9598   0.8291  
0.30       0.9533   0.8914   0.9519   0.9892   0.9586   0.8329  
0.35       0.9613   0.9061   0.9680   0.9834   0.9344   0.8794  
0.40       0.9662   0.9169   0.9746   0.9830   0.9327   0.9017   <--
0.45       0.9660   0.9161   0.9753   0.9820   0.9286   0.9039  
0.50       0.9365   0.8302   0.9766   0.9458   0.7762   0.8923  
0.55       0.9352   0.8238   0.9798   0.9416   0.7569   0.9036  
0.60       0.9350   0.8178   0.9864   0.9358   0.7294   0.9307  
0.65       0.9346   0.8098   0.9942   0.9290   0.6962   0.9678  
0.70       0.9330   0.8041   0.9944   0.9272   0.6876   0.9683  
0.75       0.9284   0.7862   0.9960   0.9210   0.6582   0.9760  
0.80       0.9245   0.7702   0.9976   0.9156   0.6323   0.9850  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9662, F1=0.9169, Normal Recall=0.9746, Normal Precision=0.9830, Attack Recall=0.9327, Attack Precision=0.9017

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
Loading model: models/tflite/saved_model_traditional_qat.tflite

Test set: Normal 70% : Attack 30%  (n=194,230, N=135,961, A=58,269)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9467   0.9170   0.9320   0.9914   0.9811   0.8608  
0.20       0.9507   0.9221   0.9412   0.9877   0.9727   0.8765  
0.25       0.9534   0.9251   0.9506   0.9822   0.9598   0.8927  
0.30       0.9540   0.9259   0.9520   0.9817   0.9586   0.8953  
0.35       0.9576   0.9298   0.9676   0.9718   0.9344   0.9251  
0.40       0.9618   0.9360   0.9742   0.9712   0.9327   0.9394   <--
0.45       0.9610   0.9345   0.9749   0.9696   0.9286   0.9406  
0.50       0.9163   0.8476   0.9763   0.9106   0.7762   0.9334  
0.55       0.9127   0.8388   0.9795   0.9039   0.7569   0.9405  
0.60       0.9091   0.8280   0.9861   0.8948   0.7294   0.9574  
0.65       0.9047   0.8142   0.9941   0.8842   0.6962   0.9806  
0.70       0.9022   0.8084   0.9942   0.8813   0.6876   0.9808  
0.75       0.8947   0.7894   0.9960   0.8718   0.6583   0.9859  
0.80       0.8880   0.7721   0.9976   0.8636   0.6324   0.9911  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9618, F1=0.9360, Normal Recall=0.9742, Normal Precision=0.9712, Attack Recall=0.9327, Attack Precision=0.9394

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
Loading model: models/tflite/saved_model_traditional_qat.tflite

Test set: Normal 60% : Attack 40%  (n=145,675, N=87,405, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9519   0.9422   0.9324   0.9867   0.9811   0.9064  
0.20       0.9539   0.9441   0.9414   0.9811   0.9727   0.9171  
0.25       0.9543   0.9438   0.9506   0.9726   0.9598   0.9283  
0.30       0.9545   0.9440   0.9519   0.9718   0.9586   0.9299  
0.35       0.9543   0.9424   0.9676   0.9568   0.9344   0.9505  
0.40       0.9576   0.9462   0.9742   0.9560   0.9327   0.9601   <--
0.45       0.9563   0.9445   0.9748   0.9534   0.9286   0.9609  
0.50       0.8962   0.8568   0.9763   0.8674   0.7762   0.9562  
0.55       0.8904   0.8467   0.9794   0.8580   0.7569   0.9607  
0.60       0.8834   0.8334   0.9860   0.8453   0.7294   0.9721  
0.65       0.8748   0.8165   0.9939   0.8307   0.6962   0.9870  
0.70       0.8715   0.8106   0.9940   0.8268   0.6876   0.9871  
0.75       0.8608   0.7909   0.9958   0.8138   0.6582   0.9905  
0.80       0.8514   0.7730   0.9975   0.8027   0.6323   0.9941  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9576, F1=0.9462, Normal Recall=0.9742, Normal Precision=0.9560, Attack Recall=0.9327, Attack Precision=0.9601

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
Loading model: models/tflite/saved_model_traditional_qat.tflite

Test set: Normal 50% : Attack 50%  (n=116,540, N=58,270, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9570   0.9580   0.9330   0.9801   0.9811   0.9360   <--
0.20       0.9573   0.9580   0.9419   0.9719   0.9727   0.9437  
0.25       0.9554   0.9556   0.9511   0.9595   0.9598   0.9515  
0.30       0.9554   0.9556   0.9523   0.9583   0.9586   0.9526  
0.35       0.9511   0.9502   0.9677   0.9366   0.9344   0.9666  
0.40       0.9536   0.9526   0.9745   0.9354   0.9327   0.9734  
0.45       0.9519   0.9507   0.9752   0.9318   0.9286   0.9740  
0.50       0.8764   0.8626   0.9766   0.8136   0.7762   0.9707  
0.55       0.8683   0.8518   0.9797   0.8012   0.7569   0.9738  
0.60       0.8577   0.8367   0.9860   0.7846   0.7294   0.9812  
0.65       0.8450   0.8179   0.9938   0.7659   0.6962   0.9912  
0.70       0.8408   0.8120   0.9939   0.7609   0.6876   0.9913  
0.75       0.8270   0.7919   0.9958   0.7445   0.6582   0.9936  
0.80       0.8149   0.7735   0.9974   0.7307   0.6323   0.9959  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9570, F1=0.9580, Normal Recall=0.9330, Normal Precision=0.9801, Attack Recall=0.9811, Attack Precision=0.9360

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
0.15       0.9247   0.7169   0.9215   0.9944   0.9534   0.5744  
0.20       0.9338   0.7421   0.9317   0.9944   0.9528   0.6077  
0.25       0.9414   0.7630   0.9411   0.9934   0.9438   0.6404  
0.30       0.9546   0.8040   0.9573   0.9920   0.9305   0.7078  
0.35       0.9665   0.8442   0.9732   0.9895   0.9067   0.7898   <--
0.40       0.9665   0.8442   0.9732   0.9895   0.9067   0.7898  
0.45       0.9653   0.8360   0.9744   0.9869   0.8839   0.7930  
0.50       0.9653   0.8360   0.9744   0.9869   0.8839   0.7930  
0.55       0.9640   0.8266   0.9757   0.9841   0.8584   0.7971  
0.60       0.9630   0.8181   0.9775   0.9813   0.8327   0.8041  
0.65       0.9630   0.8181   0.9775   0.9813   0.8327   0.8041  
0.70       0.9565   0.7406   0.9939   0.9593   0.6205   0.9182  
0.75       0.9471   0.6410   0.9998   0.9446   0.4727   0.9954  
0.80       0.9469   0.6390   0.9998   0.9444   0.4702   0.9968  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9665, F1=0.8442, Normal Recall=0.9732, Normal Precision=0.9895, Attack Recall=0.9067, Attack Precision=0.7898

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
0.15       0.9279   0.8409   0.9217   0.9873   0.9527   0.7526  
0.20       0.9359   0.8559   0.9318   0.9873   0.9520   0.7774  
0.25       0.9415   0.8658   0.9412   0.9850   0.9428   0.8004  
0.30       0.9519   0.8854   0.9574   0.9820   0.9296   0.8452  
0.35       0.9597   0.8999   0.9732   0.9763   0.9057   0.8942   <--
0.40       0.9597   0.8999   0.9732   0.9763   0.9057   0.8942  
0.45       0.9561   0.8894   0.9744   0.9708   0.8830   0.8959  
0.50       0.9561   0.8894   0.9744   0.9708   0.8830   0.8959  
0.55       0.9521   0.8776   0.9757   0.9648   0.8577   0.8983  
0.60       0.9483   0.8655   0.9775   0.9587   0.8316   0.9023  
0.65       0.9483   0.8655   0.9775   0.9587   0.8316   0.9023  
0.70       0.9196   0.7559   0.9939   0.9133   0.6224   0.9624  
0.75       0.8944   0.6418   0.9997   0.8836   0.4730   0.9978  
0.80       0.8939   0.6394   0.9998   0.8830   0.4702   0.9984  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9597, F1=0.8999, Normal Recall=0.9732, Normal Precision=0.9763, Attack Recall=0.9057, Attack Precision=0.8942

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
0.15       0.9310   0.8923   0.9217   0.9785   0.9527   0.8391  
0.20       0.9378   0.9018   0.9317   0.9784   0.9520   0.8567  
0.25       0.9416   0.9065   0.9411   0.9746   0.9428   0.8729  
0.30       0.9488   0.9160   0.9571   0.9695   0.9296   0.9027  
0.35       0.9527   0.9199   0.9728   0.9601   0.9057   0.9346   <--
0.40       0.9527   0.9199   0.9728   0.9601   0.9057   0.9346  
0.45       0.9467   0.9086   0.9740   0.9510   0.8830   0.9357  
0.50       0.9467   0.9086   0.9740   0.9510   0.8830   0.9357  
0.55       0.9401   0.8957   0.9754   0.9412   0.8577   0.9372  
0.60       0.9335   0.8824   0.9772   0.9312   0.8316   0.9398  
0.65       0.9335   0.8824   0.9772   0.9312   0.8316   0.9398  
0.70       0.8824   0.7605   0.9938   0.8600   0.6224   0.9773  
0.75       0.8417   0.6420   0.9998   0.8157   0.4730   0.9989  
0.80       0.8409   0.6395   0.9998   0.8149   0.4703   0.9991  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9527, F1=0.9199, Normal Recall=0.9728, Normal Precision=0.9601, Attack Recall=0.9057, Attack Precision=0.9346

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
0.15       0.9340   0.9203   0.9214   0.9669   0.9527   0.8899  
0.20       0.9397   0.9266   0.9315   0.9668   0.9520   0.9026  
0.25       0.9417   0.9283   0.9410   0.9611   0.9428   0.9142  
0.30       0.9459   0.9322   0.9567   0.9533   0.9296   0.9347   <--
0.35       0.9458   0.9304   0.9726   0.9393   0.9057   0.9565  
0.40       0.9458   0.9304   0.9726   0.9393   0.9057   0.9565  
0.45       0.9374   0.9186   0.9737   0.9258   0.8830   0.9572  
0.50       0.9374   0.9186   0.9737   0.9258   0.8830   0.9572  
0.55       0.9282   0.9053   0.9752   0.9114   0.8577   0.9584  
0.60       0.9189   0.8913   0.9771   0.8969   0.8316   0.9603  
0.65       0.9189   0.8913   0.9771   0.8969   0.8316   0.9603  
0.70       0.8452   0.7629   0.9937   0.7979   0.6224   0.9851  
0.75       0.7891   0.6421   0.9997   0.7400   0.4730   0.9992  
0.80       0.7880   0.6395   0.9998   0.7390   0.4702   0.9993  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9459, F1=0.9322, Normal Recall=0.9567, Normal Precision=0.9533, Attack Recall=0.9296, Attack Precision=0.9347

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
0.15       0.9371   0.9381   0.9215   0.9512   0.9527   0.9238  
0.20       0.9417   0.9423   0.9314   0.9510   0.9520   0.9328  
0.25       0.9419   0.9419   0.9410   0.9427   0.9428   0.9411  
0.30       0.9433   0.9425   0.9569   0.9315   0.9296   0.9557   <--
0.35       0.9392   0.9371   0.9727   0.9116   0.9057   0.9708  
0.40       0.9392   0.9371   0.9727   0.9116   0.9057   0.9708  
0.45       0.9285   0.9251   0.9740   0.8927   0.8830   0.9714  
0.50       0.9285   0.9251   0.9740   0.8927   0.8830   0.9714  
0.55       0.9166   0.9114   0.9755   0.8727   0.8577   0.9722  
0.60       0.9046   0.8970   0.9776   0.8530   0.8316   0.9737  
0.65       0.9046   0.8970   0.9776   0.8530   0.8316   0.9737  
0.70       0.8081   0.7644   0.9938   0.7247   0.6224   0.9901  
0.75       0.7364   0.6421   0.9998   0.6548   0.4730   0.9995  
0.80       0.7350   0.6396   0.9998   0.6536   0.4702   0.9995  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9433, F1=0.9425, Normal Recall=0.9569, Normal Precision=0.9315, Attack Recall=0.9296, Attack Precision=0.9557

```


## Threshold Tuning (saved_model_pruned_qat)

Model: `models/tflite/saved_model_pruned_qat.tflite`

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
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 90% : Attack 10%  (n=299,940, N=269,946, A=29,994)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9639   0.8374   0.9679   0.9918   0.9284   0.7626  
0.20       0.9661   0.8454   0.9703   0.9918   0.9281   0.7763  
0.25       0.9674   0.8507   0.9719   0.9918   0.9276   0.7855  
0.30       0.9681   0.8530   0.9726   0.9917   0.9270   0.7900  
0.35       0.9682   0.8527   0.9733   0.9911   0.9217   0.7934  
0.40       0.9683   0.8528   0.9737   0.9909   0.9193   0.7953  
0.45       0.9689   0.8547   0.9747   0.9905   0.9159   0.8011  
0.50       0.9692   0.8558   0.9753   0.9904   0.9145   0.8042  
0.55       0.9692   0.8555   0.9756   0.9900   0.9114   0.8060  
0.60       0.9698   0.8572   0.9770   0.9894   0.9056   0.8136  
0.65       0.9747   0.8612   0.9958   0.9765   0.7847   0.9543  
0.70       0.9761   0.8674   0.9978   0.9762   0.7810   0.9753   <--
0.75       0.9756   0.8636   0.9980   0.9754   0.7737   0.9771  
0.80       0.9742   0.8545   0.9981   0.9738   0.7586   0.9781  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.7
  At threshold 0.7: Accuracy=0.9761, F1=0.8674, Normal Recall=0.9978, Normal Precision=0.9762, Attack Recall=0.7810, Attack Precision=0.9753

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
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 80% : Attack 20%  (n=291,350, N=233,080, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9599   0.9025   0.9681   0.9816   0.9273   0.8790  
0.20       0.9617   0.9064   0.9704   0.9815   0.9270   0.8867  
0.25       0.9628   0.9088   0.9719   0.9814   0.9263   0.8920  
0.30       0.9633   0.9098   0.9727   0.9813   0.9257   0.8945   <--
0.35       0.9628   0.9083   0.9734   0.9800   0.9205   0.8964  
0.40       0.9627   0.9077   0.9738   0.9794   0.9183   0.8975  
0.45       0.9628   0.9078   0.9748   0.9786   0.9149   0.9007  
0.50       0.9629   0.9079   0.9753   0.9783   0.9135   0.9023  
0.55       0.9626   0.9068   0.9757   0.9775   0.9102   0.9034  
0.60       0.9625   0.9062   0.9771   0.9761   0.9045   0.9079  
0.65       0.9533   0.8703   0.9959   0.9483   0.7830   0.9795  
0.70       0.9542   0.8720   0.9979   0.9477   0.7796   0.9893  
0.75       0.9531   0.8682   0.9981   0.9462   0.7730   0.9902  
0.80       0.9501   0.8585   0.9982   0.9428   0.7575   0.9905  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9633, F1=0.9098, Normal Recall=0.9727, Normal Precision=0.9813, Attack Recall=0.9257, Attack Precision=0.8945

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
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 70% : Attack 30%  (n=194,230, N=135,961, A=58,269)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9557   0.9263   0.9679   0.9688   0.9273   0.9253  
0.20       0.9571   0.9284   0.9700   0.9687   0.9270   0.9298  
0.25       0.9580   0.9298   0.9716   0.9685   0.9263   0.9333  
0.30       0.9584   0.9303   0.9724   0.9683   0.9257   0.9349   <--
0.35       0.9573   0.9282   0.9731   0.9662   0.9205   0.9361  
0.40       0.9569   0.9275   0.9735   0.9653   0.9183   0.9369  
0.45       0.9566   0.9268   0.9745   0.9639   0.9149   0.9389  
0.50       0.9565   0.9264   0.9749   0.9634   0.9135   0.9397  
0.55       0.9558   0.9251   0.9753   0.9620   0.9102   0.9404  
0.60       0.9550   0.9235   0.9767   0.9598   0.9045   0.9433  
0.65       0.9321   0.8737   0.9960   0.9146   0.7830   0.9881  
0.70       0.9324   0.8737   0.9979   0.9135   0.7796   0.9937  
0.75       0.9305   0.8697   0.9980   0.9112   0.7730   0.9941  
0.80       0.9259   0.8599   0.9981   0.9057   0.7575   0.9943  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9584, F1=0.9303, Normal Recall=0.9724, Normal Precision=0.9683, Attack Recall=0.9257, Attack Precision=0.9349

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
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 60% : Attack 40%  (n=145,675, N=87,405, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9516   0.9388   0.9679   0.9523   0.9273   0.9506  
0.20       0.9528   0.9401   0.9699   0.9522   0.9270   0.9536  
0.25       0.9535   0.9410   0.9716   0.9519   0.9263   0.9561  
0.30       0.9537   0.9411   0.9724   0.9515   0.9257   0.9571   <--
0.35       0.9521   0.9389   0.9731   0.9483   0.9205   0.9580  
0.40       0.9514   0.9380   0.9735   0.9470   0.9183   0.9585  
0.45       0.9507   0.9369   0.9745   0.9450   0.9149   0.9599  
0.50       0.9503   0.9364   0.9749   0.9442   0.9135   0.9604  
0.55       0.9492   0.9348   0.9752   0.9422   0.9102   0.9608  
0.60       0.9478   0.9327   0.9766   0.9388   0.9045   0.9626  
0.65       0.9107   0.8752   0.9959   0.8732   0.7830   0.9921  
0.70       0.9105   0.8745   0.9977   0.8717   0.7796   0.9957  
0.75       0.9079   0.8704   0.9979   0.8683   0.7730   0.9959  
0.80       0.9018   0.8606   0.9980   0.8606   0.7575   0.9960  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9537, F1=0.9411, Normal Recall=0.9724, Normal Precision=0.9515, Attack Recall=0.9257, Attack Precision=0.9571

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
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 50% : Attack 50%  (n=116,540, N=58,270, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9476   0.9466   0.9680   0.9301   0.9273   0.9667  
0.20       0.9487   0.9475   0.9704   0.9300   0.9270   0.9690  
0.25       0.9492   0.9480   0.9721   0.9295   0.9263   0.9707   <--
0.30       0.9492   0.9480   0.9728   0.9290   0.9257   0.9714  
0.35       0.9470   0.9455   0.9734   0.9245   0.9205   0.9719  
0.40       0.9460   0.9445   0.9738   0.9226   0.9183   0.9722  
0.45       0.9449   0.9431   0.9748   0.9197   0.9149   0.9732  
0.50       0.9443   0.9426   0.9752   0.9185   0.9135   0.9736  
0.55       0.9428   0.9409   0.9754   0.9157   0.9102   0.9737  
0.60       0.9406   0.9384   0.9768   0.9109   0.9045   0.9750  
0.65       0.8894   0.8763   0.9959   0.8211   0.7830   0.9948  
0.70       0.8887   0.8751   0.9978   0.8191   0.7796   0.9971  
0.75       0.8855   0.8709   0.9979   0.8147   0.7730   0.9973  
0.80       0.8778   0.8611   0.9980   0.8045   0.7575   0.9974  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.9492, F1=0.9480, Normal Recall=0.9721, Normal Precision=0.9295, Attack Recall=0.9263, Attack Precision=0.9707

```

