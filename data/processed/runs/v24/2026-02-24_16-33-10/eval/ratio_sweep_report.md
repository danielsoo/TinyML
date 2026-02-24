# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-24 16:37:00 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0075 | 0.1064 | 0.2053 | 0.3044 | 0.4035 | 0.5024 | 0.6014 | 0.7003 | 0.7993 | 0.8981 | 0.9973 |
| noQAT+PTQ | 0.0000 | 0.1000 | 0.2000 | 0.3000 | 0.4000 | 0.5000 | 0.6000 | 0.7000 | 0.8000 | 0.9000 | 1.0000 |
| saved_model_traditional_qat | 0.9556 | 0.9560 | 0.9556 | 0.9557 | 0.9555 | 0.9553 | 0.9552 | 0.9553 | 0.9549 | 0.9548 | 0.9545 |
| QAT+PTQ | 0.9388 | 0.9396 | 0.9399 | 0.9410 | 0.9416 | 0.9417 | 0.9426 | 0.9433 | 0.9436 | 0.9442 | 0.9445 |
| Compressed (QAT) | 0.9719 | 0.9679 | 0.9633 | 0.9591 | 0.9542 | 0.9499 | 0.9453 | 0.9404 | 0.9361 | 0.9314 | 0.9270 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1825 | 0.3342 | 0.4624 | 0.5722 | 0.6671 | 0.7501 | 0.8233 | 0.8883 | 0.9463 | 0.9986 |
| noQAT+PTQ | 0.0000 | 0.1818 | 0.3333 | 0.4615 | 0.5714 | 0.6667 | 0.7500 | 0.8235 | 0.8889 | 0.9474 | 1.0000 |
| saved_model_traditional_qat | 0.0000 | 0.8130 | 0.8959 | 0.9283 | 0.9449 | 0.9552 | 0.9624 | 0.9677 | 0.9713 | 0.9744 | 0.9767 |
| QAT+PTQ | 0.0000 | 0.7577 | 0.8627 | 0.9056 | 0.9283 | 0.9418 | 0.9518 | 0.9589 | 0.9640 | 0.9682 | 0.9715 |
| Compressed (QAT) | 0.0000 | 0.8525 | 0.9099 | 0.9315 | 0.9418 | 0.9487 | 0.9531 | 0.9561 | 0.9587 | 0.9605 | 0.9621 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0075 | 0.0074 | 0.0073 | 0.0074 | 0.0076 | 0.0075 | 0.0075 | 0.0074 | 0.0075 | 0.0051 | 0.0000 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| saved_model_traditional_qat | 0.9556 | 0.9561 | 0.9559 | 0.9562 | 0.9561 | 0.9560 | 0.9563 | 0.9572 | 0.9561 | 0.9569 | 0.0000 |
| QAT+PTQ | 0.9388 | 0.9391 | 0.9387 | 0.9394 | 0.9397 | 0.9388 | 0.9398 | 0.9406 | 0.9398 | 0.9410 | 0.0000 |
| Compressed (QAT) | 0.9719 | 0.9725 | 0.9724 | 0.9729 | 0.9724 | 0.9727 | 0.9728 | 0.9717 | 0.9725 | 0.9716 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0075 | 0.0000 | 0.0000 | 0.0000 | 0.0075 | 1.0000 |
| 90 | 10 | 299,940 | 0.1064 | 0.1004 | 0.9974 | 0.1825 | 0.0074 | 0.9627 |
| 80 | 20 | 291,350 | 0.2053 | 0.2007 | 0.9973 | 0.3342 | 0.0073 | 0.9152 |
| 70 | 30 | 194,230 | 0.3044 | 0.3010 | 0.9973 | 0.4624 | 0.0074 | 0.8645 |
| 60 | 40 | 145,675 | 0.4035 | 0.4012 | 0.9973 | 0.5722 | 0.0076 | 0.8087 |
| 50 | 50 | 116,540 | 0.5024 | 0.5012 | 0.9973 | 0.6671 | 0.0075 | 0.7336 |
| 40 | 60 | 97,115 | 0.6014 | 0.6012 | 0.9973 | 0.7501 | 0.0075 | 0.6489 |
| 30 | 70 | 83,240 | 0.7003 | 0.7010 | 0.9973 | 0.8233 | 0.0074 | 0.5380 |
| 20 | 80 | 72,835 | 0.7993 | 0.8008 | 0.9973 | 0.8883 | 0.0075 | 0.4082 |
| 10 | 90 | 64,740 | 0.8981 | 0.9002 | 0.9973 | 0.9463 | 0.0051 | 0.1728 |
| 0 | 100 | 58,270 | 0.9973 | 1.0000 | 0.9973 | 0.9986 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9556 | 0.0000 | 0.0000 | 0.0000 | 0.9556 | 1.0000 |
| 90 | 10 | 299,940 | 0.9560 | 0.7076 | 0.9552 | 0.8130 | 0.9561 | 0.9948 |
| 80 | 20 | 291,350 | 0.9556 | 0.8441 | 0.9545 | 0.8959 | 0.9559 | 0.9883 |
| 70 | 30 | 194,230 | 0.9557 | 0.9034 | 0.9545 | 0.9283 | 0.9562 | 0.9800 |
| 60 | 40 | 145,675 | 0.9555 | 0.9355 | 0.9545 | 0.9449 | 0.9561 | 0.9693 |
| 50 | 50 | 116,540 | 0.9553 | 0.9559 | 0.9545 | 0.9552 | 0.9560 | 0.9546 |
| 40 | 60 | 97,115 | 0.9552 | 0.9704 | 0.9545 | 0.9624 | 0.9563 | 0.9334 |
| 30 | 70 | 83,240 | 0.9553 | 0.9811 | 0.9545 | 0.9677 | 0.9572 | 0.9002 |
| 20 | 80 | 72,835 | 0.9549 | 0.9886 | 0.9545 | 0.9713 | 0.9561 | 0.8402 |
| 10 | 90 | 64,740 | 0.9548 | 0.9950 | 0.9545 | 0.9744 | 0.9569 | 0.7005 |
| 0 | 100 | 58,270 | 0.9545 | 1.0000 | 0.9545 | 0.9767 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9388 | 0.0000 | 0.0000 | 0.0000 | 0.9388 | 1.0000 |
| 90 | 10 | 299,940 | 0.9396 | 0.6327 | 0.9442 | 0.7577 | 0.9391 | 0.9934 |
| 80 | 20 | 291,350 | 0.9399 | 0.7940 | 0.9445 | 0.8627 | 0.9387 | 0.9854 |
| 70 | 30 | 194,230 | 0.9410 | 0.8699 | 0.9445 | 0.9056 | 0.9394 | 0.9753 |
| 60 | 40 | 145,675 | 0.9416 | 0.9126 | 0.9445 | 0.9283 | 0.9397 | 0.9621 |
| 50 | 50 | 116,540 | 0.9417 | 0.9391 | 0.9445 | 0.9418 | 0.9388 | 0.9442 |
| 40 | 60 | 97,115 | 0.9426 | 0.9592 | 0.9445 | 0.9518 | 0.9398 | 0.9186 |
| 30 | 70 | 83,240 | 0.9433 | 0.9737 | 0.9445 | 0.9589 | 0.9406 | 0.8790 |
| 20 | 80 | 72,835 | 0.9436 | 0.9843 | 0.9445 | 0.9640 | 0.9398 | 0.8089 |
| 10 | 90 | 64,740 | 0.9442 | 0.9931 | 0.9445 | 0.9682 | 0.9410 | 0.6533 |
| 0 | 100 | 58,270 | 0.9445 | 1.0000 | 0.9445 | 0.9715 | 0.0000 | 0.0000 |

### Compressed (QAT)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9719 | 0.0000 | 0.0000 | 0.0000 | 0.9719 | 1.0000 |
| 90 | 10 | 299,940 | 0.9679 | 0.7890 | 0.9270 | 0.8525 | 0.9725 | 0.9917 |
| 80 | 20 | 291,350 | 0.9633 | 0.8935 | 0.9270 | 0.9099 | 0.9724 | 0.9816 |
| 70 | 30 | 194,230 | 0.9591 | 0.9361 | 0.9270 | 0.9315 | 0.9729 | 0.9688 |
| 60 | 40 | 145,675 | 0.9542 | 0.9572 | 0.9270 | 0.9418 | 0.9724 | 0.9523 |
| 50 | 50 | 116,540 | 0.9499 | 0.9714 | 0.9270 | 0.9487 | 0.9727 | 0.9302 |
| 40 | 60 | 97,115 | 0.9453 | 0.9808 | 0.9270 | 0.9531 | 0.9728 | 0.8988 |
| 30 | 70 | 83,240 | 0.9404 | 0.9871 | 0.9270 | 0.9561 | 0.9717 | 0.8508 |
| 20 | 80 | 72,835 | 0.9361 | 0.9926 | 0.9270 | 0.9587 | 0.9725 | 0.7691 |
| 10 | 90 | 64,740 | 0.9314 | 0.9966 | 0.9270 | 0.9605 | 0.9716 | 0.5965 |
| 0 | 100 | 58,270 | 0.9270 | 1.0000 | 0.9270 | 0.9621 | 0.0000 | 0.0000 |


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
0.15       0.1002   0.1818   0.0003   0.8706   0.9996   0.1000  
0.20       0.1004   0.1818   0.0005   0.8726   0.9993   0.1000  
0.25       0.1008   0.1818   0.0010   0.8861   0.9988   0.1000  
0.30       0.1064   0.1825   0.0074   0.9627   0.9974   0.1004   <--
0.35       0.1130   0.1806   0.0170   0.8705   0.9772   0.0995  
0.40       0.1217   0.1584   0.0435   0.6924   0.8262   0.0876  
0.45       0.2784   0.1704   0.2270   0.8874   0.7408   0.0962  
0.50       0.8233   0.0644   0.9080   0.8969   0.0608   0.0684  
0.55       0.8989   0.0000   0.9988   0.8999   0.0000   0.0000  
0.60       0.9000   0.0000   0.9999   0.9000   0.0000   0.0000  
0.65       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.70       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.75       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.80       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.1064, F1=0.1825, Normal Recall=0.0074, Normal Precision=0.9627, Attack Recall=0.9974, Attack Precision=0.1004

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
0.15       0.2001   0.3333   0.0003   0.6809   0.9995   0.2000  
0.20       0.2002   0.3332   0.0005   0.7059   0.9991   0.1999  
0.25       0.2006   0.3332   0.0010   0.7634   0.9987   0.2000  
0.30       0.2053   0.3342   0.0073   0.9152   0.9973   0.2007   <--
0.35       0.2088   0.3305   0.0169   0.7424   0.9765   0.1989  
0.40       0.1997   0.2919   0.0434   0.4978   0.8248   0.1773  
0.45       0.3299   0.3064   0.2273   0.7777   0.7402   0.1932  
0.50       0.7381   0.0824   0.9080   0.7942   0.0588   0.1377  
0.55       0.7990   0.0000   0.9988   0.7998   0.0000   0.0000  
0.60       0.8000   0.0000   0.9999   0.8000   0.0000   0.0000  
0.65       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.70       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.75       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.80       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.2053, F1=0.3342, Normal Recall=0.0073, Normal Precision=0.9152, Attack Recall=0.9973, Attack Precision=0.2007

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
0.15       0.3000   0.4614   0.0003   0.5522   0.9995   0.2999  
0.20       0.3001   0.4614   0.0005   0.5763   0.9991   0.2999  
0.25       0.3003   0.4613   0.0010   0.6512   0.9987   0.2999  
0.30       0.3044   0.4624   0.0075   0.8658   0.9973   0.3010   <--
0.35       0.3048   0.4573   0.0169   0.6261   0.9765   0.2986  
0.40       0.2778   0.4066   0.0434   0.3661   0.8248   0.2698  
0.45       0.3822   0.4182   0.2287   0.6725   0.7402   0.2914  
0.50       0.6534   0.0923   0.9082   0.6924   0.0588   0.2153  
0.55       0.6991   0.0000   0.9988   0.6997   0.0000   0.0000  
0.60       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.65       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.70       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.75       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.80       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.3044, F1=0.4624, Normal Recall=0.0075, Normal Precision=0.8658, Attack Recall=0.9973, Attack Precision=0.3010

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
0.15       0.4000   0.5713   0.0003   0.4545   0.9995   0.3999  
0.20       0.3999   0.5712   0.0005   0.4565   0.9991   0.3999  
0.25       0.4001   0.5712   0.0011   0.5614   0.9987   0.4000  
0.30       0.4035   0.5722   0.0076   0.8087   0.9973   0.4012   <--
0.35       0.4007   0.5659   0.0168   0.5174   0.9765   0.3984  
0.40       0.3560   0.5061   0.0435   0.2713   0.8248   0.3650  
0.45       0.4325   0.5106   0.2273   0.5676   0.7402   0.3897  
0.50       0.5680   0.0982   0.9074   0.5912   0.0588   0.2974  
0.55       0.5992   0.0000   0.9987   0.5997   0.0000   0.0000  
0.60       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.65       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.70       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.75       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.80       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.4035, F1=0.5722, Normal Recall=0.0076, Normal Precision=0.8087, Attack Recall=0.9973, Attack Precision=0.4012

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
0.15       0.4999   0.6665   0.0002   0.3182   0.9995   0.4999  
0.20       0.4998   0.6664   0.0004   0.3421   0.9991   0.4999  
0.25       0.4999   0.6663   0.0010   0.4444   0.9987   0.4999  
0.30       0.5022   0.6671   0.0072   0.7252   0.9973   0.5011   <--
0.35       0.4966   0.6598   0.0167   0.4155   0.9765   0.4983  
0.40       0.4342   0.5931   0.0435   0.1990   0.8248   0.4630  
0.45       0.4830   0.5888   0.2258   0.4650   0.7402   0.4888  
0.50       0.4831   0.1021   0.9074   0.4908   0.0588   0.3882  
0.55       0.4994   0.0000   0.9987   0.4997   0.0000   0.0000  
0.60       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.65       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.70       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.75       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.80       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.5022, F1=0.6671, Normal Recall=0.0072, Normal Precision=0.7252, Attack Recall=0.9973, Attack Precision=0.5011

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
0.40       0.1000   0.1818   0.0000   0.0000   1.0000   0.1000  
0.45       0.1031   0.1823   0.0034   1.0000   1.0000   0.1003  
0.50       0.9488   0.6578   0.9996   0.9465   0.4918   0.9929   <--
0.55       0.9469   0.6388   0.9999   0.9444   0.4698   0.9979  
0.60       0.9465   0.6356   0.9999   0.9440   0.4662   0.9983  
0.65       0.9460   0.6304   0.9999   0.9434   0.4605   0.9990  
0.70       0.9442   0.6136   1.0000   0.9417   0.4428   0.9991  
0.75       0.9438   0.6089   1.0000   0.9412   0.4378   0.9994  
0.80       0.9431   0.6030   1.0000   0.9406   0.4317   0.9995  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.9488, F1=0.6578, Normal Recall=0.9996, Normal Precision=0.9465, Attack Recall=0.4918, Attack Precision=0.9929

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
0.40       0.2000   0.3333   0.0000   0.0000   1.0000   0.2000  
0.45       0.2027   0.3341   0.0034   1.0000   1.0000   0.2005  
0.50       0.8980   0.6585   0.9996   0.8872   0.4916   0.9967   <--
0.55       0.8938   0.6389   0.9999   0.8829   0.4697   0.9990  
0.60       0.8932   0.6358   0.9999   0.8823   0.4662   0.9992  
0.65       0.8920   0.6303   0.9999   0.8811   0.4603   0.9995  
0.70       0.8886   0.6142   1.0000   0.8778   0.4432   0.9996  
0.75       0.8877   0.6096   1.0000   0.8769   0.4385   0.9997  
0.80       0.8864   0.6032   1.0000   0.8756   0.4319   0.9998  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.8980, F1=0.6585, Normal Recall=0.9996, Normal Precision=0.8872, Attack Recall=0.4916, Attack Precision=0.9967

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
0.40       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.45       0.3023   0.4624   0.0033   1.0000   1.0000   0.3007  
0.50       0.8472   0.6588   0.9996   0.8210   0.4916   0.9981   <--
0.55       0.8408   0.6390   0.9999   0.8148   0.4697   0.9994  
0.60       0.8398   0.6359   0.9999   0.8138   0.4662   0.9996  
0.65       0.8381   0.6304   0.9999   0.8121   0.4603   0.9997  
0.70       0.8329   0.6142   1.0000   0.8074   0.4433   0.9998  
0.75       0.8315   0.6096   1.0000   0.8060   0.4385   0.9998  
0.80       0.8295   0.6032   1.0000   0.8042   0.4319   0.9998  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.8472, F1=0.6588, Normal Recall=0.9996, Normal Precision=0.8210, Attack Recall=0.4916, Attack Precision=0.9981

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
0.40       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.45       0.4021   0.5723   0.0035   1.0000   1.0000   0.4008  
0.50       0.7964   0.6589   0.9996   0.7468   0.4916   0.9988   <--
0.55       0.7878   0.6391   0.9999   0.7388   0.4697   0.9997  
0.60       0.7865   0.6359   0.9999   0.7375   0.4662   0.9998  
0.65       0.7841   0.6304   1.0000   0.7354   0.4603   0.9999  
0.70       0.7773   0.6142   1.0000   0.7293   0.4432   0.9999  
0.75       0.7754   0.6096   1.0000   0.7276   0.4385   0.9999  
0.80       0.7727   0.6032   1.0000   0.7253   0.4319   0.9999  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.7964, F1=0.6589, Normal Recall=0.9996, Normal Precision=0.7468, Attack Recall=0.4916, Attack Precision=0.9988

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
0.40       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.45       0.5018   0.6675   0.0036   1.0000   1.0000   0.5009   <--
0.50       0.7456   0.6590   0.9996   0.6629   0.4916   0.9992  
0.55       0.7348   0.6391   0.9999   0.6534   0.4697   0.9998  
0.60       0.7331   0.6359   0.9999   0.6520   0.4662   0.9999  
0.65       0.7301   0.6304   1.0000   0.6495   0.4603   1.0000  
0.70       0.7216   0.6142   1.0000   0.6424   0.4432   1.0000  
0.75       0.7192   0.6096   1.0000   0.6404   0.4385   1.0000  
0.80       0.7159   0.6032   1.0000   0.6377   0.4319   1.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.5018, F1=0.6675, Normal Recall=0.0036, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5009

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
0.15       0.9433   0.7741   0.9402   0.9966   0.9715   0.6434  
0.20       0.9467   0.7847   0.9440   0.9966   0.9711   0.6583  
0.25       0.9475   0.7863   0.9453   0.9961   0.9667   0.6627  
0.30       0.9562   0.8137   0.9561   0.9950   0.9567   0.7079  
0.35       0.9704   0.8634   0.9744   0.9926   0.9345   0.8023  
0.40       0.9709   0.8650   0.9751   0.9924   0.9327   0.8064  
0.45       0.9713   0.8658   0.9762   0.9917   0.9269   0.8123   <--
0.50       0.9575   0.7846   0.9780   0.9749   0.7732   0.7963  
0.55       0.9594   0.7789   0.9867   0.9688   0.7142   0.8565  
0.60       0.9633   0.7836   0.9964   0.9640   0.6650   0.9538  
0.65       0.9633   0.7835   0.9966   0.9639   0.6636   0.9561  
0.70       0.9616   0.7687   0.9976   0.9612   0.6379   0.9671  
0.75       0.9619   0.7696   0.9982   0.9610   0.6358   0.9747  
0.80       0.9625   0.7723   0.9988   0.9610   0.6356   0.9838  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9713, F1=0.8658, Normal Recall=0.9762, Normal Precision=0.9917, Attack Recall=0.9269, Attack Precision=0.8123

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
0.15       0.9461   0.8780   0.9402   0.9920   0.9696   0.8022  
0.20       0.9491   0.8839   0.9441   0.9919   0.9691   0.8125  
0.25       0.9493   0.8838   0.9454   0.9908   0.9647   0.8155  
0.30       0.9559   0.8964   0.9562   0.9883   0.9545   0.8449  
0.35       0.9661   0.9167   0.9744   0.9830   0.9326   0.9012  
0.40       0.9663   0.9170   0.9752   0.9826   0.9309   0.9035   <--
0.45       0.9661   0.9160   0.9763   0.9812   0.9253   0.9069  
0.50       0.9364   0.8287   0.9781   0.9444   0.7695   0.8976  
0.55       0.9312   0.8049   0.9868   0.9314   0.7092   0.9305  
0.60       0.9293   0.7891   0.9964   0.9216   0.6611   0.9786  
0.65       0.9292   0.7885   0.9966   0.9214   0.6598   0.9797  
0.70       0.9249   0.7714   0.9976   0.9160   0.6340   0.9849  
0.75       0.9249   0.7710   0.9982   0.9156   0.6319   0.9885  
0.80       0.9254   0.7722   0.9989   0.9156   0.6317   0.9929  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9663, F1=0.9170, Normal Recall=0.9752, Normal Precision=0.9826, Attack Recall=0.9309, Attack Precision=0.9035

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
0.15       0.9491   0.9195   0.9403   0.9863   0.9696   0.8744  
0.20       0.9517   0.9232   0.9442   0.9862   0.9691   0.8815  
0.25       0.9513   0.9224   0.9456   0.9843   0.9647   0.8837  
0.30       0.9557   0.9283   0.9562   0.9800   0.9545   0.9034  
0.35       0.9617   0.9359   0.9741   0.9712   0.9326   0.9392   <--
0.40       0.9617   0.9358   0.9749   0.9705   0.9309   0.9407  
0.45       0.9608   0.9340   0.9760   0.9682   0.9253   0.9429  
0.50       0.9153   0.8451   0.9778   0.9083   0.7696   0.9370  
0.55       0.9034   0.8149   0.9866   0.8879   0.7092   0.9576  
0.60       0.8958   0.7920   0.9964   0.8728   0.6611   0.9876  
0.65       0.8956   0.7913   0.9966   0.8724   0.6598   0.9882  
0.70       0.8885   0.7734   0.9976   0.8641   0.6340   0.9913  
0.75       0.8883   0.7724   0.9981   0.8635   0.6319   0.9932  
0.80       0.8887   0.7731   0.9989   0.8636   0.6317   0.9959  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9617, F1=0.9359, Normal Recall=0.9741, Normal Precision=0.9712, Attack Recall=0.9326, Attack Precision=0.9392

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
0.15       0.9522   0.9420   0.9406   0.9789   0.9696   0.9159  
0.20       0.9542   0.9442   0.9443   0.9786   0.9691   0.9206  
0.25       0.9533   0.9429   0.9456   0.9757   0.9647   0.9220  
0.30       0.9554   0.9448   0.9560   0.9693   0.9545   0.9353  
0.35       0.9575   0.9461   0.9740   0.9559   0.9326   0.9599   <--
0.40       0.9572   0.9456   0.9748   0.9548   0.9309   0.9609  
0.45       0.9557   0.9435   0.9760   0.9514   0.9253   0.9625  
0.50       0.8945   0.8537   0.9778   0.8642   0.7695   0.9584  
0.55       0.8756   0.8202   0.9865   0.8358   0.7092   0.9723  
0.60       0.8622   0.7933   0.9963   0.8151   0.6611   0.9916  
0.65       0.8618   0.7924   0.9964   0.8146   0.6598   0.9919  
0.70       0.8521   0.7742   0.9975   0.8035   0.6340   0.9941  
0.75       0.8516   0.7730   0.9980   0.8027   0.6319   0.9953  
0.80       0.8520   0.7735   0.9988   0.8027   0.6317   0.9972  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9575, F1=0.9461, Normal Recall=0.9740, Normal Precision=0.9559, Attack Recall=0.9326, Attack Precision=0.9599

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
0.15       0.9553   0.9559   0.9409   0.9687   0.9696   0.9426  
0.20       0.9568   0.9573   0.9444   0.9683   0.9691   0.9458   <--
0.25       0.9552   0.9556   0.9457   0.9640   0.9647   0.9467  
0.30       0.9553   0.9553   0.9561   0.9546   0.9545   0.9561  
0.35       0.9535   0.9525   0.9744   0.9353   0.9326   0.9733  
0.40       0.9529   0.9519   0.9750   0.9338   0.9309   0.9739  
0.45       0.9508   0.9495   0.9763   0.9289   0.9253   0.9750  
0.50       0.8737   0.8591   0.9779   0.8093   0.7695   0.9721  
0.55       0.8479   0.8234   0.9865   0.7723   0.7092   0.9814  
0.60       0.8286   0.7941   0.9961   0.7461   0.6611   0.9942  
0.65       0.8280   0.7932   0.9963   0.7454   0.6598   0.9944  
0.70       0.8157   0.7748   0.9974   0.7316   0.6340   0.9959  
0.75       0.8149   0.7735   0.9979   0.7305   0.6319   0.9967  
0.80       0.8152   0.7737   0.9987   0.7306   0.6317   0.9980  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9568, F1=0.9573, Normal Recall=0.9444, Normal Precision=0.9683, Attack Recall=0.9691, Attack Precision=0.9458

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
0.15       0.9113   0.6831   0.9064   0.9946   0.9558   0.5315  
0.20       0.9205   0.7056   0.9170   0.9943   0.9524   0.5604  
0.25       0.9309   0.7333   0.9289   0.9940   0.9495   0.5973  
0.30       0.9397   0.7582   0.9391   0.9936   0.9454   0.6330  
0.35       0.9518   0.7940   0.9542   0.9919   0.9296   0.6929  
0.40       0.9518   0.7940   0.9542   0.9919   0.9296   0.6929  
0.45       0.9604   0.8189   0.9676   0.9881   0.8954   0.7544  
0.50       0.9604   0.8189   0.9676   0.9881   0.8954   0.7544  
0.55       0.9623   0.8238   0.9714   0.9865   0.8807   0.7737   <--
0.60       0.9617   0.8175   0.9733   0.9840   0.8575   0.7811  
0.65       0.9617   0.8175   0.9733   0.9840   0.8575   0.7811  
0.70       0.9621   0.8168   0.9752   0.9826   0.8447   0.7907  
0.75       0.9645   0.7919   0.9967   0.9650   0.6750   0.9578  
0.80       0.9475   0.6505   0.9986   0.9461   0.4882   0.9746  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9623, F1=0.8238, Normal Recall=0.9714, Normal Precision=0.9865, Attack Recall=0.8807, Attack Precision=0.7737

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
0.15       0.9162   0.8201   0.9066   0.9877   0.9548   0.7187  
0.20       0.9241   0.8338   0.9172   0.9870   0.9517   0.7419  
0.25       0.9329   0.8499   0.9289   0.9865   0.9490   0.7695  
0.30       0.9402   0.8633   0.9391   0.9854   0.9445   0.7950  
0.35       0.9492   0.8797   0.9542   0.9817   0.9290   0.8354  
0.40       0.9492   0.8797   0.9542   0.9817   0.9290   0.8354  
0.45       0.9529   0.8837   0.9676   0.9734   0.8941   0.8735   <--
0.50       0.9529   0.8837   0.9676   0.9734   0.8941   0.8735  
0.55       0.9531   0.8823   0.9714   0.9699   0.8796   0.8850  
0.60       0.9499   0.8723   0.9733   0.9643   0.8560   0.8891  
0.65       0.9499   0.8723   0.9733   0.9643   0.8560   0.8891  
0.70       0.9489   0.8686   0.9752   0.9616   0.8441   0.8947  
0.75       0.9318   0.7976   0.9967   0.9240   0.6720   0.9809  
0.80       0.8965   0.6537   0.9986   0.8865   0.4883   0.9885  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9529, F1=0.8837, Normal Recall=0.9676, Normal Precision=0.9734, Attack Recall=0.8941, Attack Precision=0.8735

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
0.15       0.9215   0.8794   0.9072   0.9791   0.9548   0.8151  
0.20       0.9279   0.8879   0.9177   0.9779   0.9517   0.8321  
0.25       0.9351   0.8977   0.9292   0.9770   0.9490   0.8517  
0.30       0.9409   0.9055   0.9393   0.9753   0.9445   0.8697  
0.35       0.9465   0.9124   0.9540   0.9691   0.9290   0.8964   <--
0.40       0.9465   0.9124   0.9540   0.9691   0.9290   0.8964  
0.45       0.9453   0.9075   0.9673   0.9552   0.8941   0.9213  
0.50       0.9453   0.9075   0.9673   0.9552   0.8941   0.9213  
0.55       0.9436   0.9035   0.9711   0.9495   0.8796   0.9287  
0.60       0.9379   0.8921   0.9730   0.9404   0.8560   0.9313  
0.65       0.9379   0.8921   0.9730   0.9404   0.8560   0.9313  
0.70       0.9356   0.8871   0.9748   0.9358   0.8441   0.9348  
0.75       0.8993   0.8002   0.9967   0.8764   0.6720   0.9888  
0.80       0.8455   0.6548   0.9986   0.8200   0.4884   0.9935  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9465, F1=0.9124, Normal Recall=0.9540, Normal Precision=0.9691, Attack Recall=0.9290, Attack Precision=0.8964

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
0.15       0.9259   0.9116   0.9067   0.9678   0.9548   0.8721  
0.20       0.9310   0.9169   0.9173   0.9661   0.9517   0.8847  
0.25       0.9369   0.9233   0.9288   0.9647   0.9490   0.8989  
0.30       0.9413   0.9279   0.9392   0.9621   0.9445   0.9120  
0.35       0.9439   0.9298   0.9538   0.9527   0.9290   0.9306   <--
0.40       0.9439   0.9298   0.9538   0.9527   0.9290   0.9306  
0.45       0.9380   0.9203   0.9673   0.9320   0.8941   0.9479  
0.50       0.9380   0.9203   0.9673   0.9320   0.8941   0.9479  
0.55       0.9344   0.9147   0.9709   0.9236   0.8796   0.9527  
0.60       0.9261   0.9026   0.9729   0.9102   0.8560   0.9546  
0.65       0.9261   0.9026   0.9729   0.9102   0.8560   0.9546  
0.70       0.9224   0.8969   0.9746   0.9036   0.8441   0.9569  
0.75       0.8668   0.8014   0.9966   0.8201   0.6720   0.9925  
0.80       0.7945   0.6553   0.9986   0.7454   0.4883   0.9956  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9439, F1=0.9298, Normal Recall=0.9538, Normal Precision=0.9527, Attack Recall=0.9290, Attack Precision=0.9306

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
0.15       0.9305   0.9322   0.9063   0.9525   0.9548   0.9106  
0.20       0.9344   0.9355   0.9171   0.9499   0.9517   0.9198  
0.25       0.9389   0.9395   0.9287   0.9480   0.9490   0.9301  
0.30       0.9418   0.9419   0.9391   0.9442   0.9445   0.9394   <--
0.35       0.9415   0.9407   0.9540   0.9307   0.9290   0.9528  
0.40       0.9415   0.9407   0.9540   0.9307   0.9290   0.9528  
0.45       0.9308   0.9281   0.9674   0.9014   0.8941   0.9648  
0.50       0.9308   0.9281   0.9674   0.9014   0.8941   0.9648  
0.55       0.9253   0.9218   0.9711   0.8897   0.8796   0.9682  
0.60       0.9145   0.9092   0.9730   0.8711   0.8560   0.9694  
0.65       0.9145   0.9092   0.9730   0.8711   0.8560   0.9694  
0.70       0.9094   0.9031   0.9747   0.8621   0.8441   0.9709  
0.75       0.8342   0.8021   0.9964   0.7523   0.6720   0.9947  
0.80       0.7434   0.6555   0.9985   0.6612   0.4883   0.9968  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9418, F1=0.9419, Normal Recall=0.9391, Normal Precision=0.9442, Attack Recall=0.9445, Attack Precision=0.9394

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
0.15       0.9453   0.7777   0.9441   0.9949   0.9566   0.6552  
0.20       0.9469   0.7816   0.9465   0.9942   0.9504   0.6637  
0.25       0.9644   0.8393   0.9684   0.9919   0.9288   0.7655  
0.30       0.9680   0.8531   0.9725   0.9919   0.9282   0.7892  
0.35       0.9685   0.8548   0.9730   0.9918   0.9278   0.7926  
0.40       0.9690   0.8566   0.9737   0.9916   0.9262   0.7967  
0.45       0.9697   0.8592   0.9746   0.9916   0.9255   0.8017  
0.50       0.9696   0.8582   0.9749   0.9911   0.9215   0.8031  
0.55       0.9682   0.8480   0.9772   0.9874   0.8874   0.8121  
0.60       0.9725   0.8613   0.9857   0.9838   0.8537   0.8690  
0.65       0.9753   0.8654   0.9954   0.9775   0.7940   0.9509   <--
0.70       0.9756   0.8649   0.9971   0.9763   0.7822   0.9672  
0.75       0.9758   0.8646   0.9982   0.9754   0.7735   0.9799  
0.80       0.9750   0.8598   0.9983   0.9746   0.7654   0.9809  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.65
  At threshold 0.65: Accuracy=0.9753, F1=0.8654, Normal Recall=0.9954, Normal Precision=0.9775, Attack Recall=0.7940, Attack Precision=0.9509

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
0.15       0.9464   0.8770   0.9442   0.9883   0.9552   0.8105  
0.20       0.9470   0.8776   0.9466   0.9867   0.9489   0.8162  
0.25       0.9601   0.9030   0.9683   0.9816   0.9274   0.8798  
0.30       0.9634   0.9101   0.9725   0.9816   0.9270   0.8939  
0.35       0.9638   0.9109   0.9731   0.9815   0.9266   0.8958  
0.40       0.9640   0.9113   0.9738   0.9811   0.9249   0.8982  
0.45       0.9646   0.9126   0.9746   0.9810   0.9244   0.9011   <--
0.50       0.9641   0.9111   0.9749   0.9801   0.9206   0.9018  
0.55       0.9590   0.8962   0.9772   0.9716   0.8859   0.9068  
0.60       0.9591   0.8929   0.9857   0.9640   0.8527   0.9370  
0.65       0.9549   0.8754   0.9955   0.9505   0.7925   0.9776  
0.70       0.9538   0.8712   0.9971   0.9479   0.7809   0.9852  
0.75       0.9532   0.8684   0.9983   0.9461   0.7727   0.9912  
0.80       0.9516   0.8633   0.9984   0.9443   0.7644   0.9917  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9646, F1=0.9126, Normal Recall=0.9746, Normal Precision=0.9810, Attack Recall=0.9244, Attack Precision=0.9011

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
0.15       0.9477   0.9164   0.9445   0.9801   0.9552   0.8806  
0.20       0.9475   0.9155   0.9469   0.9774   0.9489   0.8844  
0.25       0.9558   0.9264   0.9680   0.9689   0.9274   0.9254  
0.30       0.9586   0.9308   0.9722   0.9688   0.9270   0.9346  
0.35       0.9590   0.9312   0.9728   0.9687   0.9266   0.9359  
0.40       0.9590   0.9312   0.9736   0.9680   0.9249   0.9376  
0.45       0.9594   0.9318   0.9744   0.9678   0.9244   0.9394   <--
0.50       0.9585   0.9301   0.9747   0.9663   0.9206   0.9398  
0.55       0.9498   0.9136   0.9771   0.9523   0.8859   0.9432  
0.60       0.9458   0.9042   0.9857   0.9398   0.8527   0.9624  
0.65       0.9346   0.8790   0.9954   0.9180   0.7925   0.9867  
0.70       0.9322   0.8735   0.9970   0.9139   0.7809   0.9912  
0.75       0.9306   0.8697   0.9982   0.9111   0.7727   0.9947  
0.80       0.9282   0.8646   0.9984   0.9081   0.7644   0.9950  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9594, F1=0.9318, Normal Recall=0.9744, Normal Precision=0.9678, Attack Recall=0.9244, Attack Precision=0.9394

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
0.15       0.9487   0.9371   0.9444   0.9694   0.9552   0.9197  
0.20       0.9476   0.9354   0.9468   0.9653   0.9489   0.9224  
0.25       0.9518   0.9390   0.9680   0.9524   0.9274   0.9509  
0.30       0.9542   0.9418   0.9723   0.9523   0.9270   0.9571  
0.35       0.9544   0.9420   0.9729   0.9521   0.9266   0.9579  
0.40       0.9542   0.9417   0.9738   0.9511   0.9249   0.9593  
0.45       0.9545   0.9420   0.9746   0.9508   0.9244   0.9604   <--
0.50       0.9531   0.9402   0.9748   0.9485   0.9206   0.9605  
0.55       0.9406   0.9227   0.9771   0.9278   0.8859   0.9627  
0.60       0.9324   0.9099   0.9856   0.9094   0.8527   0.9752  
0.65       0.9142   0.8808   0.9953   0.8780   0.7925   0.9911  
0.70       0.9105   0.8746   0.9969   0.8722   0.7809   0.9940  
0.75       0.9079   0.8704   0.9981   0.8682   0.7727   0.9964  
0.80       0.9047   0.8652   0.9983   0.8640   0.7644   0.9966  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9545, F1=0.9420, Normal Recall=0.9746, Normal Precision=0.9508, Attack Recall=0.9244, Attack Precision=0.9604

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
0.15       0.9500   0.9502   0.9447   0.9548   0.9552   0.9453   <--
0.20       0.9479   0.9480   0.9469   0.9488   0.9489   0.9470  
0.25       0.9478   0.9467   0.9683   0.9303   0.9274   0.9669  
0.30       0.9498   0.9487   0.9727   0.9302   0.9270   0.9714  
0.35       0.9499   0.9487   0.9732   0.9299   0.9266   0.9719  
0.40       0.9495   0.9482   0.9741   0.9284   0.9249   0.9728  
0.45       0.9496   0.9483   0.9749   0.9280   0.9244   0.9735  
0.50       0.9479   0.9464   0.9751   0.9247   0.9206   0.9737  
0.55       0.9317   0.9284   0.9775   0.8954   0.8859   0.9752  
0.60       0.9193   0.9135   0.9859   0.8700   0.8527   0.9837  
0.65       0.8939   0.8819   0.9952   0.8275   0.7925   0.9940  
0.70       0.8889   0.8754   0.9969   0.8198   0.7809   0.9960  
0.75       0.8854   0.8709   0.9982   0.8145   0.7727   0.9976  
0.80       0.8813   0.8656   0.9983   0.8091   0.7644   0.9978  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9500, F1=0.9502, Normal Recall=0.9447, Normal Precision=0.9548, Attack Recall=0.9552, Attack Precision=0.9453

```

