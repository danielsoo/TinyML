# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-24 21:24:16 |

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
| Original (TFLite) | 0.0005 | 0.1005 | 0.2005 | 0.3005 | 0.4004 | 0.5002 | 0.6002 | 0.7002 | 0.8000 | 0.9000 | 0.9999 |
| noQAT+PTQ | 0.0014 | 0.1012 | 0.2011 | 0.3010 | 0.4009 | 0.5006 | 0.6006 | 0.7003 | 0.8002 | 0.9000 | 1.0000 |
| saved_model_traditional_qat | 0.9716 | 0.9684 | 0.9645 | 0.9609 | 0.9569 | 0.9531 | 0.9492 | 0.9450 | 0.9414 | 0.9376 | 0.9338 |
| QAT+PTQ | 0.9698 | 0.9658 | 0.9608 | 0.9564 | 0.9512 | 0.9461 | 0.9415 | 0.9360 | 0.9314 | 0.9263 | 0.9215 |
| Compressed (QAT) | 0.9641 | 0.9623 | 0.9595 | 0.9573 | 0.9543 | 0.9517 | 0.9494 | 0.9462 | 0.9436 | 0.9411 | 0.9385 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1819 | 0.3334 | 0.4617 | 0.5716 | 0.6667 | 0.7501 | 0.8236 | 0.8889 | 0.9474 | 1.0000 |
| noQAT+PTQ | 0.0000 | 0.1820 | 0.3336 | 0.4619 | 0.5718 | 0.6669 | 0.7503 | 0.8237 | 0.8890 | 0.9474 | 1.0000 |
| saved_model_traditional_qat | 0.0000 | 0.8555 | 0.9132 | 0.9348 | 0.9454 | 0.9522 | 0.9566 | 0.9596 | 0.9623 | 0.9642 | 0.9658 |
| QAT+PTQ | 0.0000 | 0.8434 | 0.9039 | 0.9268 | 0.9379 | 0.9447 | 0.9498 | 0.9527 | 0.9556 | 0.9575 | 0.9592 |
| Compressed (QAT) | 0.0000 | 0.8327 | 0.9026 | 0.9295 | 0.9427 | 0.9510 | 0.9570 | 0.9607 | 0.9638 | 0.9663 | 0.9683 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0005 | 0.0006 | 0.0006 | 0.0007 | 0.0006 | 0.0005 | 0.0006 | 0.0007 | 0.0004 | 0.0005 | 0.0000 |
| noQAT+PTQ | 0.0014 | 0.0014 | 0.0014 | 0.0014 | 0.0015 | 0.0013 | 0.0017 | 0.0012 | 0.0012 | 0.0005 | 0.0000 |
| saved_model_traditional_qat | 0.9716 | 0.9722 | 0.9722 | 0.9725 | 0.9722 | 0.9724 | 0.9722 | 0.9709 | 0.9717 | 0.9716 | 0.0000 |
| QAT+PTQ | 0.9698 | 0.9707 | 0.9706 | 0.9713 | 0.9709 | 0.9706 | 0.9715 | 0.9697 | 0.9710 | 0.9694 | 0.0000 |
| Compressed (QAT) | 0.9641 | 0.9648 | 0.9648 | 0.9654 | 0.9649 | 0.9649 | 0.9657 | 0.9642 | 0.9640 | 0.9642 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0005 | 0.0000 | 0.0000 | 0.0000 | 0.0005 | 1.0000 |
| 90 | 10 | 299,940 | 0.1005 | 0.1000 | 0.9999 | 0.1819 | 0.0006 | 0.9750 |
| 80 | 20 | 291,350 | 0.2005 | 0.2001 | 0.9999 | 0.3334 | 0.0006 | 0.9714 |
| 70 | 30 | 194,230 | 0.3005 | 0.3001 | 0.9999 | 0.4617 | 0.0007 | 0.9583 |
| 60 | 40 | 145,675 | 0.4004 | 0.4001 | 0.9999 | 0.5716 | 0.0006 | 0.9322 |
| 50 | 50 | 116,540 | 0.5002 | 0.5001 | 0.9999 | 0.6667 | 0.0005 | 0.8788 |
| 40 | 60 | 97,115 | 0.6002 | 0.6001 | 0.9999 | 0.7501 | 0.0006 | 0.8621 |
| 30 | 70 | 83,240 | 0.7002 | 0.7001 | 0.9999 | 0.8236 | 0.0007 | 0.8182 |
| 20 | 80 | 72,835 | 0.8000 | 0.8001 | 0.9999 | 0.8889 | 0.0004 | 0.6000 |
| 10 | 90 | 64,740 | 0.9000 | 0.9000 | 0.9999 | 0.9474 | 0.0005 | 0.4286 |
| 0 | 100 | 58,270 | 0.9999 | 1.0000 | 0.9999 | 1.0000 | 0.0000 | 0.0000 |

### noQAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0014 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 |
| 90 | 10 | 299,940 | 0.1012 | 0.1001 | 1.0000 | 0.1820 | 0.0014 | 0.9973 |
| 80 | 20 | 291,350 | 0.2011 | 0.2002 | 1.0000 | 0.3336 | 0.0014 | 0.9939 |
| 70 | 30 | 194,230 | 0.3010 | 0.3003 | 1.0000 | 0.4619 | 0.0014 | 0.9898 |
| 60 | 40 | 145,675 | 0.4009 | 0.4003 | 1.0000 | 0.5718 | 0.0015 | 0.9846 |
| 50 | 50 | 116,540 | 0.5006 | 0.5003 | 1.0000 | 0.6669 | 0.0013 | 0.9744 |
| 40 | 60 | 97,115 | 0.6006 | 0.6004 | 1.0000 | 0.7503 | 0.0017 | 0.9701 |
| 30 | 70 | 83,240 | 0.7003 | 0.7002 | 1.0000 | 0.8237 | 0.0012 | 0.9375 |
| 20 | 80 | 72,835 | 0.8002 | 0.8002 | 1.0000 | 0.8890 | 0.0012 | 0.9000 |
| 10 | 90 | 64,740 | 0.9000 | 0.9000 | 1.0000 | 0.9474 | 0.0005 | 0.6000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### saved_model_traditional_qat

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9716 | 0.0000 | 0.0000 | 0.0000 | 0.9716 | 1.0000 |
| 90 | 10 | 299,940 | 0.9684 | 0.7887 | 0.9347 | 0.8555 | 0.9722 | 0.9926 |
| 80 | 20 | 291,350 | 0.9645 | 0.8934 | 0.9338 | 0.9132 | 0.9722 | 0.9833 |
| 70 | 30 | 194,230 | 0.9609 | 0.9358 | 0.9338 | 0.9348 | 0.9725 | 0.9717 |
| 60 | 40 | 145,675 | 0.9569 | 0.9573 | 0.9338 | 0.9454 | 0.9722 | 0.9566 |
| 50 | 50 | 116,540 | 0.9531 | 0.9713 | 0.9338 | 0.9522 | 0.9724 | 0.9363 |
| 40 | 60 | 97,115 | 0.9492 | 0.9805 | 0.9338 | 0.9566 | 0.9722 | 0.9074 |
| 30 | 70 | 83,240 | 0.9450 | 0.9868 | 0.9338 | 0.9596 | 0.9709 | 0.8628 |
| 20 | 80 | 72,835 | 0.9414 | 0.9925 | 0.9339 | 0.9623 | 0.9717 | 0.7860 |
| 10 | 90 | 64,740 | 0.9376 | 0.9966 | 0.9338 | 0.9642 | 0.9716 | 0.6200 |
| 0 | 100 | 58,270 | 0.9338 | 1.0000 | 0.9338 | 0.9658 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9698 | 0.0000 | 0.0000 | 0.0000 | 0.9698 | 1.0000 |
| 90 | 10 | 299,940 | 0.9658 | 0.7773 | 0.9219 | 0.8434 | 0.9707 | 0.9911 |
| 80 | 20 | 291,350 | 0.9608 | 0.8869 | 0.9215 | 0.9039 | 0.9706 | 0.9802 |
| 70 | 30 | 194,230 | 0.9564 | 0.9322 | 0.9215 | 0.9268 | 0.9713 | 0.9665 |
| 60 | 40 | 145,675 | 0.9512 | 0.9548 | 0.9215 | 0.9379 | 0.9709 | 0.9489 |
| 50 | 50 | 116,540 | 0.9461 | 0.9691 | 0.9215 | 0.9447 | 0.9706 | 0.9252 |
| 40 | 60 | 97,115 | 0.9415 | 0.9798 | 0.9215 | 0.9498 | 0.9715 | 0.8919 |
| 30 | 70 | 83,240 | 0.9360 | 0.9861 | 0.9215 | 0.9527 | 0.9697 | 0.8412 |
| 20 | 80 | 72,835 | 0.9314 | 0.9922 | 0.9216 | 0.9556 | 0.9710 | 0.7558 |
| 10 | 90 | 64,740 | 0.9263 | 0.9963 | 0.9215 | 0.9575 | 0.9694 | 0.5786 |
| 0 | 100 | 58,270 | 0.9215 | 1.0000 | 0.9215 | 0.9592 | 0.0000 | 0.0000 |

### Compressed (QAT)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9641 | 0.0000 | 0.0000 | 0.0000 | 0.9641 | 1.0000 |
| 90 | 10 | 299,940 | 0.9623 | 0.7479 | 0.9391 | 0.8327 | 0.9648 | 0.9930 |
| 80 | 20 | 291,350 | 0.9595 | 0.8694 | 0.9385 | 0.9026 | 0.9648 | 0.9843 |
| 70 | 30 | 194,230 | 0.9573 | 0.9207 | 0.9385 | 0.9295 | 0.9654 | 0.9734 |
| 60 | 40 | 145,675 | 0.9543 | 0.9469 | 0.9385 | 0.9427 | 0.9649 | 0.9592 |
| 50 | 50 | 116,540 | 0.9517 | 0.9639 | 0.9385 | 0.9510 | 0.9649 | 0.9401 |
| 40 | 60 | 97,115 | 0.9494 | 0.9762 | 0.9385 | 0.9570 | 0.9657 | 0.9128 |
| 30 | 70 | 83,240 | 0.9462 | 0.9839 | 0.9385 | 0.9607 | 0.9642 | 0.8704 |
| 20 | 80 | 72,835 | 0.9436 | 0.9905 | 0.9385 | 0.9638 | 0.9640 | 0.7967 |
| 10 | 90 | 64,740 | 0.9411 | 0.9958 | 0.9385 | 0.9663 | 0.9642 | 0.6353 |
| 0 | 100 | 58,270 | 0.9385 | 1.0000 | 0.9385 | 0.9683 | 0.0000 | 0.0000 |


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
0.15       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.20       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.25       0.1002   0.1819   0.0002   1.0000   1.0000   0.1000  
0.30       0.1005   0.1819   0.0006   1.0000   1.0000   0.1001   <--
0.35       0.1010   0.1819   0.0012   0.9424   0.9993   0.1000  
0.40       0.1018   0.1817   0.0024   0.8721   0.9969   0.0999  
0.45       0.1086   0.1772   0.0141   0.7586   0.9597   0.0976  
0.50       0.2320   0.1586   0.1774   0.8526   0.7239   0.0891  
0.55       0.7795   0.0692   0.8570   0.8936   0.0819   0.0598  
0.60       0.8871   0.0044   0.9854   0.8989   0.0025   0.0186  
0.65       0.8988   0.0025   0.9985   0.9000   0.0013   0.0856  
0.70       0.8994   0.0000   0.9994   0.8999   0.0000   0.0000  
0.75       0.8996   0.0000   0.9995   0.9000   0.0000   0.0000  
0.80       0.8996   0.0000   0.9996   0.9000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.1005, F1=0.1819, Normal Recall=0.0006, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1001

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
0.15       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.20       0.2001   0.3334   0.0002   1.0000   1.0000   0.2000  
0.25       0.2002   0.3334   0.0002   0.9655   1.0000   0.2000  
0.30       0.2005   0.3334   0.0006   0.9716   0.9999   0.2001   <--
0.35       0.2008   0.3333   0.0012   0.8388   0.9991   0.2000  
0.40       0.2013   0.3330   0.0023   0.7524   0.9969   0.1999  
0.45       0.2030   0.3249   0.0139   0.5765   0.9592   0.1956  
0.50       0.2864   0.2880   0.1776   0.7184   0.7216   0.1799  
0.55       0.7027   0.1014   0.8574   0.7892   0.0839   0.1282  
0.60       0.7887   0.0044   0.9852   0.7980   0.0023   0.0380  
0.65       0.7990   0.0025   0.9985   0.8000   0.0012   0.1667  
0.70       0.7995   0.0001   0.9993   0.7999   0.0000   0.0125  
0.75       0.7996   0.0001   0.9995   0.7999   0.0000   0.0172  
0.80       0.7996   0.0001   0.9995   0.7999   0.0000   0.0187  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.2005, F1=0.3334, Normal Recall=0.0006, Normal Precision=0.9716, Attack Recall=0.9999, Attack Precision=0.2001

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
0.15       0.3001   0.4616   0.0001   1.0000   1.0000   0.3000  
0.20       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.25       0.3001   0.4616   0.0002   0.9394   1.0000   0.3000  
0.30       0.3004   0.4617   0.0006   0.9506   0.9999   0.3001   <--
0.35       0.3006   0.4615   0.0012   0.7523   0.9991   0.3001  
0.40       0.3008   0.4610   0.0024   0.6455   0.9969   0.2999  
0.45       0.2976   0.4504   0.0141   0.4466   0.9592   0.2943  
0.50       0.3407   0.3964   0.1774   0.5980   0.7216   0.2732  
0.55       0.6256   0.1185   0.8577   0.6860   0.0839   0.2018  
0.60       0.6905   0.0045   0.9854   0.6974   0.0023   0.0643  
0.65       0.6994   0.0025   0.9986   0.7000   0.0012   0.2717  
0.70       0.6996   0.0001   0.9994   0.6999   0.0000   0.0233  
0.75       0.6997   0.0001   0.9996   0.6999   0.0000   0.0323  
0.80       0.6997   0.0001   0.9996   0.6999   0.0000   0.0357  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.3004, F1=0.4617, Normal Recall=0.0006, Normal Precision=0.9506, Attack Recall=0.9999, Attack Precision=0.3001

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
0.15       0.4000   0.5714   0.0001   1.0000   1.0000   0.4000  
0.20       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.25       0.4001   0.5715   0.0002   0.9000   1.0000   0.4000  
0.30       0.4003   0.5715   0.0005   0.9200   0.9999   0.4001   <--
0.35       0.4003   0.5713   0.0012   0.6582   0.9991   0.4001  
0.40       0.4002   0.5707   0.0023   0.5339   0.9969   0.3998  
0.45       0.3920   0.5579   0.0138   0.3368   0.9592   0.3934  
0.50       0.3954   0.4884   0.1779   0.4894   0.7216   0.3692  
0.55       0.5488   0.1295   0.8588   0.5844   0.0839   0.2837  
0.60       0.5923   0.0046   0.9856   0.5971   0.0023   0.0978  
0.65       0.5996   0.0025   0.9985   0.5999   0.0012   0.3564  
0.70       0.5996   0.0001   0.9994   0.5999   0.0000   0.0364  
0.75       0.5998   0.0001   0.9996   0.5999   0.0000   0.0556  
0.80       0.5998   0.0001   0.9997   0.5999   0.0000   0.0625  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.4003, F1=0.5715, Normal Recall=0.0005, Normal Precision=0.9200, Attack Recall=0.9999, Attack Precision=0.4001

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
0.15       0.5000   0.6667   0.0001   1.0000   1.0000   0.5000  
0.20       0.5001   0.6667   0.0001   1.0000   1.0000   0.5000  
0.25       0.5001   0.6667   0.0002   0.8462   1.0000   0.5000  
0.30       0.5002   0.6668   0.0005   0.8824   0.9999   0.5001   <--
0.35       0.5001   0.6665   0.0012   0.5645   0.9991   0.5001  
0.40       0.4997   0.6659   0.0025   0.4475   0.9969   0.4999  
0.45       0.4866   0.6513   0.0139   0.2545   0.9592   0.4931  
0.50       0.4497   0.5673   0.1777   0.3896   0.7216   0.4674  
0.55       0.4711   0.1369   0.8584   0.4837   0.0839   0.3720  
0.60       0.4940   0.0046   0.9857   0.4970   0.0023   0.1405  
0.65       0.4999   0.0025   0.9986   0.5000   0.0012   0.4737  
0.70       0.4997   0.0001   0.9993   0.4998   0.0000   0.0488  
0.75       0.4998   0.0001   0.9996   0.4999   0.0000   0.0833  
0.80       0.4998   0.0001   0.9997   0.4999   0.0000   0.0909  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.5002, F1=0.6668, Normal Recall=0.0005, Normal Precision=0.8824, Attack Recall=0.9999, Attack Precision=0.5001

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
0.20       0.1006   0.1819   0.0006   0.9943   1.0000   0.1001  
0.25       0.1008   0.1819   0.0009   0.9959   1.0000   0.1001  
0.30       0.1012   0.1820   0.0014   0.9973   1.0000   0.1001  
0.35       0.1021   0.1822   0.0023   0.9984   1.0000   0.1002  
0.40       0.1055   0.1827   0.0061   0.9982   0.9999   0.1005  
0.45       0.1309   0.1871   0.0344   0.9997   0.9999   0.1032  
0.50       0.9466   0.6404   0.9989   0.9449   0.4759   0.9788   <--
0.55       0.9465   0.6362   0.9996   0.9442   0.4681   0.9925  
0.60       0.9463   0.6337   0.9999   0.9438   0.4643   0.9978  
0.65       0.9459   0.6292   0.9999   0.9433   0.4593   0.9983  
0.70       0.9445   0.6165   0.9999   0.9420   0.4459   0.9985  
0.75       0.9435   0.6067   0.9999   0.9410   0.4357   0.9987  
0.80       0.9427   0.5989   0.9999   0.9402   0.4277   0.9988  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.9466, F1=0.6404, Normal Recall=0.9989, Normal Precision=0.9449, Attack Recall=0.4759, Attack Precision=0.9788

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
0.20       0.2005   0.3335   0.0006   0.9933   1.0000   0.2001  
0.25       0.2007   0.3335   0.0009   0.9906   1.0000   0.2001  
0.30       0.2011   0.3336   0.0014   0.9938   1.0000   0.2002  
0.35       0.2018   0.3338   0.0023   0.9963   1.0000   0.2004  
0.40       0.2048   0.3347   0.0060   0.9972   0.9999   0.2010  
0.45       0.2275   0.3411   0.0344   0.9993   0.9999   0.2056  
0.50       0.8942   0.6425   0.9989   0.8839   0.4754   0.9905   <--
0.55       0.8933   0.6372   0.9996   0.8826   0.4683   0.9967  
0.60       0.8927   0.6338   0.9999   0.8818   0.4641   0.9989  
0.65       0.8918   0.6294   0.9999   0.8809   0.4594   0.9992  
0.70       0.8891   0.6166   0.9999   0.8783   0.4458   0.9993  
0.75       0.8872   0.6072   0.9999   0.8764   0.4361   0.9994  
0.80       0.8856   0.5995   0.9999   0.8749   0.4282   0.9994  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.8942, F1=0.6425, Normal Recall=0.9989, Normal Precision=0.8839, Attack Recall=0.4754, Attack Precision=0.9905

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
0.20       0.3005   0.4617   0.0007   0.9895   1.0000   0.3001  
0.25       0.3007   0.4618   0.0009   0.9847   1.0000   0.3002  
0.30       0.3010   0.4619   0.0014   0.9895   1.0000   0.3003  
0.35       0.3016   0.4621   0.0023   0.9937   1.0000   0.3005  
0.40       0.3040   0.4629   0.0057   0.9949   0.9999   0.3012  
0.45       0.3239   0.4702   0.0342   0.9987   0.9999   0.3073  
0.50       0.8418   0.6432   0.9988   0.8163   0.4754   0.9940   <--
0.55       0.8402   0.6374   0.9996   0.8143   0.4683   0.9978  
0.60       0.8391   0.6339   0.9999   0.8132   0.4641   0.9994  
0.65       0.8377   0.6295   0.9999   0.8119   0.4594   0.9995  
0.70       0.8337   0.6166   0.9999   0.8081   0.4458   0.9996  
0.75       0.8308   0.6072   0.9999   0.8053   0.4361   0.9996  
0.80       0.8284   0.5996   0.9999   0.8032   0.4282   0.9996  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.8418, F1=0.6432, Normal Recall=0.9988, Normal Precision=0.8163, Attack Recall=0.4754, Attack Precision=0.9940

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
0.20       0.4004   0.5716   0.0007   0.9841   1.0000   0.4002  
0.25       0.4006   0.5717   0.0010   0.9773   1.0000   0.4002  
0.30       0.4008   0.5718   0.0014   0.9837   1.0000   0.4003  
0.35       0.4014   0.5720   0.0023   0.9901   1.0000   0.4005  
0.40       0.4034   0.5728   0.0058   0.9921   0.9999   0.4014  
0.45       0.4202   0.5798   0.0338   0.9980   0.9999   0.4082  
0.50       0.7895   0.6437   0.9988   0.7407   0.4754   0.9963   <--
0.55       0.7871   0.6376   0.9996   0.7382   0.4683   0.9988  
0.60       0.7856   0.6339   0.9999   0.7368   0.4641   0.9997  
0.65       0.7837   0.6295   1.0000   0.7351   0.4594   0.9999  
0.70       0.7783   0.6167   1.0000   0.7302   0.4458   0.9999  
0.75       0.7744   0.6073   1.0000   0.7268   0.4361   0.9999  
0.80       0.7713   0.5996   1.0000   0.7240   0.4282   0.9999  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.7895, F1=0.6437, Normal Recall=0.9988, Normal Precision=0.7407, Attack Recall=0.4754, Attack Precision=0.9963

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
0.20       0.5003   0.6668   0.0006   0.9730   1.0000   0.5002  
0.25       0.5005   0.6669   0.0010   0.9655   1.0000   0.5002  
0.30       0.5007   0.6669   0.0013   0.9750   1.0000   0.5003  
0.35       0.5011   0.6671   0.0022   0.9847   1.0000   0.5005  
0.40       0.5030   0.6680   0.0060   0.9887   0.9999   0.5015  
0.45       0.5172   0.6744   0.0344   0.9970   0.9999   0.5087   <--
0.50       0.7371   0.6440   0.9989   0.6557   0.4754   0.9976  
0.55       0.7340   0.6377   0.9997   0.6528   0.4683   0.9993  
0.60       0.7320   0.6339   0.9999   0.6511   0.4641   0.9998  
0.65       0.7297   0.6296   1.0000   0.6491   0.4594   0.9999  
0.70       0.7229   0.6167   1.0000   0.6434   0.4458   0.9999  
0.75       0.7180   0.6073   1.0000   0.6394   0.4361   0.9999  
0.80       0.7141   0.5996   1.0000   0.6362   0.4282   1.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.5172, F1=0.6744, Normal Recall=0.0344, Normal Precision=0.9970, Attack Recall=0.9999, Attack Precision=0.5087

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
0.15       0.9395   0.7637   0.9354   0.9973   0.9771   0.6268  
0.20       0.9520   0.7997   0.9515   0.9950   0.9574   0.6866  
0.25       0.9601   0.8262   0.9616   0.9939   0.9473   0.7325  
0.30       0.9685   0.8561   0.9722   0.9927   0.9357   0.7889  
0.35       0.9691   0.8580   0.9732   0.9924   0.9327   0.7944  
0.40       0.9701   0.8619   0.9743   0.9924   0.9325   0.8013   <--
0.45       0.9699   0.8597   0.9750   0.9913   0.9234   0.8042  
0.50       0.9570   0.7847   0.9763   0.9759   0.7833   0.7862  
0.55       0.9603   0.7919   0.9831   0.9731   0.7551   0.8325  
0.60       0.9617   0.7926   0.9871   0.9708   0.7325   0.8634  
0.65       0.9662   0.8072   0.9951   0.9683   0.7068   0.9408  
0.70       0.9657   0.8031   0.9952   0.9676   0.6999   0.9421  
0.75       0.9651   0.7986   0.9956   0.9667   0.6913   0.9454  
0.80       0.9648   0.7911   0.9981   0.9641   0.6655   0.9749  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9701, F1=0.8619, Normal Recall=0.9743, Normal Precision=0.9924, Attack Recall=0.9325, Attack Precision=0.8013

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
0.15       0.9435   0.8735   0.9354   0.9936   0.9759   0.7906  
0.20       0.9524   0.8892   0.9515   0.9885   0.9558   0.8313  
0.25       0.9583   0.9007   0.9615   0.9860   0.9454   0.8600  
0.30       0.9645   0.9132   0.9722   0.9833   0.9338   0.8935  
0.35       0.9648   0.9136   0.9732   0.9826   0.9312   0.8967  
0.40       0.9657   0.9156   0.9743   0.9826   0.9311   0.9006   <--
0.45       0.9644   0.9119   0.9750   0.9803   0.9218   0.9022  
0.50       0.9370   0.8318   0.9764   0.9465   0.7794   0.8918  
0.55       0.9367   0.8259   0.9832   0.9404   0.7507   0.9178  
0.60       0.9356   0.8190   0.9872   0.9358   0.7290   0.9344  
0.65       0.9368   0.8166   0.9951   0.9307   0.7035   0.9730  
0.70       0.9356   0.8124   0.9953   0.9293   0.6970   0.9736  
0.75       0.9341   0.8068   0.9956   0.9274   0.6881   0.9750  
0.80       0.9309   0.7931   0.9981   0.9220   0.6622   0.9885  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9657, F1=0.9156, Normal Recall=0.9743, Normal Precision=0.9826, Attack Recall=0.9311, Attack Precision=0.9006

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
0.15       0.9476   0.9179   0.9355   0.9891   0.9759   0.8664  
0.20       0.9529   0.9241   0.9516   0.9805   0.9558   0.8944  
0.25       0.9566   0.9290   0.9614   0.9763   0.9454   0.9131  
0.30       0.9605   0.9342   0.9719   0.9717   0.9338   0.9345  
0.35       0.9604   0.9339   0.9730   0.9706   0.9312   0.9366  
0.40       0.9611   0.9350   0.9740   0.9706   0.9311   0.9389   <--
0.45       0.9589   0.9308   0.9748   0.9668   0.9218   0.9400  
0.50       0.9171   0.8495   0.9762   0.9117   0.7794   0.9334  
0.55       0.9133   0.8385   0.9830   0.9020   0.7507   0.9497  
0.60       0.9096   0.8287   0.9870   0.8947   0.7290   0.9599  
0.65       0.9076   0.8204   0.9951   0.8868   0.7035   0.9841  
0.70       0.9058   0.8161   0.9952   0.8846   0.6970   0.9843  
0.75       0.9033   0.8102   0.9956   0.8816   0.6881   0.9852  
0.80       0.8973   0.7945   0.9980   0.8733   0.6622   0.9930  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9611, F1=0.9350, Normal Recall=0.9740, Normal Precision=0.9706, Attack Recall=0.9311, Attack Precision=0.9389

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
0.15       0.9517   0.9418   0.9356   0.9831   0.9759   0.9099  
0.20       0.9532   0.9423   0.9514   0.9700   0.9558   0.9292  
0.25       0.9550   0.9438   0.9613   0.9635   0.9454   0.9422  
0.30       0.9566   0.9451   0.9718   0.9566   0.9338   0.9566  
0.35       0.9561   0.9444   0.9728   0.9550   0.9312   0.9580  
0.40       0.9568   0.9451   0.9739   0.9549   0.9311   0.9597   <--
0.45       0.9535   0.9407   0.9747   0.9492   0.9218   0.9605  
0.50       0.8975   0.8588   0.9762   0.8691   0.7794   0.9561  
0.55       0.8901   0.8453   0.9831   0.8554   0.7507   0.9673  
0.60       0.8838   0.8339   0.9870   0.8453   0.7290   0.9740  
0.65       0.8784   0.8223   0.9950   0.8343   0.7035   0.9895  
0.70       0.8759   0.8180   0.9952   0.8313   0.6970   0.9898  
0.75       0.8725   0.8120   0.9955   0.8272   0.6881   0.9903  
0.80       0.8637   0.7954   0.9980   0.8159   0.6622   0.9956  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9568, F1=0.9451, Normal Recall=0.9739, Normal Precision=0.9549, Attack Recall=0.9311, Attack Precision=0.9597

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
0.15       0.9560   0.9568   0.9360   0.9749   0.9759   0.9385   <--
0.20       0.9538   0.9539   0.9518   0.9557   0.9558   0.9520  
0.25       0.9535   0.9531   0.9616   0.9463   0.9454   0.9609  
0.30       0.9531   0.9521   0.9723   0.9363   0.9338   0.9712  
0.35       0.9522   0.9512   0.9732   0.9340   0.9312   0.9720  
0.40       0.9527   0.9516   0.9743   0.9339   0.9311   0.9732  
0.45       0.9484   0.9470   0.9751   0.9258   0.9218   0.9737  
0.50       0.8779   0.8646   0.9764   0.8157   0.7794   0.9706  
0.55       0.8668   0.8493   0.9830   0.7977   0.7507   0.9778  
0.60       0.8580   0.8370   0.9871   0.7846   0.7290   0.9826  
0.65       0.8492   0.8235   0.9950   0.7704   0.7035   0.9930  
0.70       0.8461   0.8191   0.9952   0.7666   0.6970   0.9931  
0.75       0.8417   0.8130   0.9954   0.7614   0.6881   0.9934  
0.80       0.8301   0.7958   0.9979   0.7471   0.6622   0.9969  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9560, F1=0.9568, Normal Recall=0.9360, Normal Precision=0.9749, Attack Recall=0.9759, Attack Precision=0.9385

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
0.15       0.9443   0.7770   0.9415   0.9965   0.9699   0.6480  
0.20       0.9647   0.8398   0.9689   0.9917   0.9266   0.7679  
0.25       0.9658   0.8435   0.9707   0.9912   0.9220   0.7773   <--
0.30       0.9658   0.8435   0.9707   0.9912   0.9220   0.7773  
0.35       0.9660   0.8432   0.9719   0.9901   0.9129   0.7833  
0.40       0.9659   0.8403   0.9737   0.9883   0.8962   0.7910  
0.45       0.9661   0.8403   0.9745   0.9877   0.8910   0.7950  
0.50       0.9661   0.8403   0.9745   0.9877   0.8910   0.7950  
0.55       0.9669   0.8414   0.9767   0.9863   0.8782   0.8074  
0.60       0.9664   0.8288   0.9832   0.9795   0.8145   0.8437  
0.65       0.9707   0.8339   0.9969   0.9713   0.7353   0.9629  
0.70       0.9705   0.8311   0.9977   0.9703   0.7255   0.9726  
0.75       0.9705   0.8311   0.9977   0.9703   0.7255   0.9726  
0.80       0.9707   0.8310   0.9984   0.9699   0.7210   0.9807  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.9658, F1=0.8435, Normal Recall=0.9707, Normal Precision=0.9912, Attack Recall=0.9220, Attack Precision=0.7773

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
0.15       0.9471   0.8799   0.9416   0.9918   0.9690   0.8058  
0.20       0.9604   0.9035   0.9689   0.9814   0.9265   0.8816  
0.25       0.9608   0.9038   0.9706   0.9802   0.9215   0.8868   <--
0.30       0.9608   0.9038   0.9706   0.9802   0.9215   0.8868  
0.35       0.9600   0.9011   0.9719   0.9780   0.9124   0.8902  
0.40       0.9579   0.8947   0.9736   0.9737   0.8949   0.8945  
0.45       0.9576   0.8937   0.9744   0.9727   0.8906   0.8968  
0.50       0.9576   0.8937   0.9744   0.9727   0.8906   0.8968  
0.55       0.9569   0.8905   0.9767   0.9696   0.8774   0.9040  
0.60       0.9491   0.8645   0.9832   0.9545   0.8127   0.9234  
0.65       0.9442   0.8402   0.9969   0.9374   0.7336   0.9831  
0.70       0.9430   0.8356   0.9977   0.9353   0.7241   0.9876  
0.75       0.9430   0.8356   0.9977   0.9353   0.7241   0.9876  
0.80       0.9427   0.8340   0.9985   0.9344   0.7196   0.9916  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.9608, F1=0.9038, Normal Recall=0.9706, Normal Precision=0.9802, Attack Recall=0.9215, Attack Precision=0.8868

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
0.15       0.9499   0.9206   0.9417   0.9861   0.9690   0.8768  
0.20       0.9559   0.9266   0.9686   0.9685   0.9265   0.9266   <--
0.25       0.9556   0.9257   0.9702   0.9665   0.9215   0.9298  
0.30       0.9556   0.9257   0.9702   0.9665   0.9215   0.9298  
0.35       0.9537   0.9220   0.9714   0.9628   0.9124   0.9319  
0.40       0.9497   0.9143   0.9732   0.9557   0.8948   0.9347  
0.45       0.9490   0.9128   0.9740   0.9541   0.8906   0.9363  
0.50       0.9490   0.9128   0.9740   0.9541   0.8906   0.9363  
0.55       0.9467   0.9081   0.9764   0.9490   0.8774   0.9410  
0.60       0.9320   0.8776   0.9831   0.9245   0.8127   0.9539  
0.65       0.9179   0.8427   0.9968   0.8972   0.7336   0.9900  
0.70       0.9156   0.8374   0.9977   0.8941   0.7241   0.9927  
0.75       0.9156   0.8374   0.9977   0.8941   0.7241   0.9927  
0.80       0.9148   0.8352   0.9985   0.8926   0.7196   0.9951  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9559, F1=0.9266, Normal Recall=0.9686, Normal Precision=0.9685, Attack Recall=0.9265, Attack Precision=0.9266

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
0.15       0.9524   0.9422   0.9414   0.9785   0.9690   0.9168   <--
0.20       0.9516   0.9387   0.9684   0.9518   0.9265   0.9513  
0.25       0.9506   0.9372   0.9700   0.9488   0.9215   0.9535  
0.30       0.9506   0.9372   0.9700   0.9488   0.9215   0.9535  
0.35       0.9477   0.9331   0.9712   0.9433   0.9124   0.9548  
0.40       0.9417   0.9247   0.9730   0.9328   0.8949   0.9567  
0.45       0.9404   0.9229   0.9737   0.9303   0.8906   0.9576  
0.50       0.9404   0.9229   0.9737   0.9303   0.8906   0.9576  
0.55       0.9366   0.9172   0.9761   0.9228   0.8774   0.9608  
0.60       0.9148   0.8841   0.9828   0.8873   0.8127   0.9693  
0.65       0.8914   0.8439   0.9966   0.8488   0.7336   0.9931  
0.70       0.8881   0.8382   0.9975   0.8443   0.7241   0.9948  
0.75       0.8881   0.8382   0.9975   0.8443   0.7241   0.9948  
0.80       0.8868   0.8357   0.9983   0.8423   0.7196   0.9966  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9524, F1=0.9422, Normal Recall=0.9414, Normal Precision=0.9785, Attack Recall=0.9690, Attack Precision=0.9168

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
0.15       0.9553   0.9559   0.9416   0.9681   0.9690   0.9431   <--
0.20       0.9476   0.9465   0.9688   0.9295   0.9265   0.9674  
0.25       0.9460   0.9446   0.9704   0.9252   0.9215   0.9689  
0.30       0.9460   0.9446   0.9704   0.9252   0.9215   0.9689  
0.35       0.9420   0.9402   0.9716   0.9173   0.9124   0.9698  
0.40       0.9340   0.9314   0.9732   0.9025   0.8949   0.9710  
0.45       0.9322   0.9293   0.9739   0.8990   0.8906   0.9715  
0.50       0.9322   0.9293   0.9739   0.8990   0.8906   0.9715  
0.55       0.9269   0.9231   0.9764   0.8885   0.8774   0.9738  
0.60       0.8980   0.8884   0.9832   0.8400   0.8127   0.9797  
0.65       0.8651   0.8447   0.9966   0.7891   0.7336   0.9954  
0.70       0.8608   0.8388   0.9974   0.7833   0.7241   0.9965  
0.75       0.8608   0.8388   0.9974   0.7833   0.7241   0.9965  
0.80       0.8590   0.8361   0.9983   0.7807   0.7196   0.9977  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9553, F1=0.9559, Normal Recall=0.9416, Normal Precision=0.9681, Attack Recall=0.9690, Attack Precision=0.9431

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
0.15       0.9449   0.7777   0.9427   0.9958   0.9642   0.6517  
0.20       0.9517   0.7988   0.9509   0.9952   0.9589   0.6846  
0.25       0.9522   0.8003   0.9515   0.9952   0.9585   0.6869  
0.30       0.9623   0.8328   0.9648   0.9931   0.9393   0.7480  
0.35       0.9637   0.8379   0.9664   0.9930   0.9390   0.7565  
0.40       0.9674   0.8519   0.9707   0.9929   0.9378   0.7804  
0.45       0.9696   0.8593   0.9743   0.9918   0.9277   0.8003  
0.50       0.9691   0.8521   0.9777   0.9878   0.8911   0.8163  
0.55       0.9752   0.8631   0.9967   0.9763   0.7818   0.9634  
0.60       0.9756   0.8646   0.9973   0.9760   0.7797   0.9703  
0.65       0.9761   0.8672   0.9980   0.9760   0.7793   0.9774   <--
0.70       0.9761   0.8669   0.9980   0.9759   0.7786   0.9778  
0.75       0.9754   0.8621   0.9983   0.9749   0.7690   0.9808  
0.80       0.9755   0.8618   0.9988   0.9746   0.7653   0.9862  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.65
  At threshold 0.65: Accuracy=0.9761, F1=0.8672, Normal Recall=0.9980, Normal Precision=0.9760, Attack Recall=0.7793, Attack Precision=0.9774

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
0.15       0.9469   0.8789   0.9429   0.9903   0.9632   0.8083  
0.20       0.9524   0.8896   0.9511   0.9890   0.9579   0.8304  
0.25       0.9528   0.8903   0.9516   0.9890   0.9575   0.8319  
0.30       0.9596   0.9029   0.9649   0.9843   0.9385   0.8699  
0.35       0.9608   0.9054   0.9665   0.9842   0.9381   0.8749  
0.40       0.9640   0.9123   0.9708   0.9840   0.9367   0.8891  
0.45       0.9649   0.9135   0.9744   0.9816   0.9270   0.9004   <--
0.50       0.9602   0.8994   0.9778   0.9726   0.8898   0.9091  
0.55       0.9536   0.8706   0.9967   0.9479   0.7810   0.9834  
0.60       0.9536   0.8704   0.9974   0.9475   0.7787   0.9866  
0.65       0.9541   0.8715   0.9980   0.9474   0.7783   0.9900  
0.70       0.9539   0.8708   0.9981   0.9471   0.7771   0.9902  
0.75       0.9522   0.8653   0.9984   0.9450   0.7675   0.9916  
0.80       0.9518   0.8638   0.9988   0.9442   0.7639   0.9939  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9649, F1=0.9135, Normal Recall=0.9744, Normal Precision=0.9816, Attack Recall=0.9270, Attack Precision=0.9004

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
0.15       0.9490   0.9189   0.9429   0.9835   0.9632   0.8785  
0.20       0.9530   0.9244   0.9509   0.9814   0.9579   0.8932  
0.25       0.9533   0.9248   0.9514   0.9812   0.9575   0.8942  
0.30       0.9566   0.9284   0.9643   0.9734   0.9385   0.9185  
0.35       0.9576   0.9299   0.9659   0.9733   0.9381   0.9219  
0.40       0.9603   0.9339   0.9704   0.9728   0.9367   0.9312   <--
0.45       0.9599   0.9327   0.9740   0.9689   0.9270   0.9385  
0.50       0.9512   0.9162   0.9775   0.9539   0.8898   0.9443  
0.55       0.9319   0.8731   0.9966   0.9139   0.7810   0.9899  
0.60       0.9317   0.8724   0.9972   0.9132   0.7787   0.9917  
0.65       0.9320   0.8729   0.9979   0.9130   0.7783   0.9939  
0.70       0.9317   0.8723   0.9980   0.9126   0.7771   0.9940  
0.75       0.9291   0.8666   0.9983   0.9093   0.7675   0.9949  
0.80       0.9283   0.8647   0.9988   0.9080   0.7639   0.9963  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9603, F1=0.9339, Normal Recall=0.9704, Normal Precision=0.9728, Attack Recall=0.9367, Attack Precision=0.9312

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
0.15       0.9510   0.9402   0.9429   0.9746   0.9632   0.9184  
0.20       0.9537   0.9430   0.9509   0.9713   0.9579   0.9286  
0.25       0.9539   0.9432   0.9515   0.9711   0.9575   0.9294  
0.30       0.9542   0.9425   0.9647   0.9592   0.9385   0.9465  
0.35       0.9550   0.9435   0.9663   0.9590   0.9381   0.9489  
0.40       0.9570   0.9458   0.9706   0.9583   0.9367   0.9550   <--
0.45       0.9552   0.9430   0.9740   0.9524   0.9270   0.9597  
0.50       0.9424   0.9251   0.9774   0.9301   0.8898   0.9633  
0.55       0.9102   0.8744   0.9964   0.8722   0.7810   0.9931  
0.60       0.9097   0.8734   0.9970   0.8711   0.7787   0.9942  
0.65       0.9100   0.8737   0.9978   0.8710   0.7783   0.9958  
0.70       0.9096   0.8730   0.9978   0.8704   0.7771   0.9959  
0.75       0.9059   0.8672   0.9982   0.8656   0.7675   0.9965  
0.80       0.9047   0.8651   0.9986   0.8638   0.7639   0.9973  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9570, F1=0.9458, Normal Recall=0.9706, Normal Precision=0.9583, Attack Recall=0.9367, Attack Precision=0.9550

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
0.15       0.9531   0.9536   0.9430   0.9624   0.9632   0.9442  
0.20       0.9544   0.9546   0.9510   0.9576   0.9579   0.9513  
0.25       0.9545   0.9547   0.9516   0.9572   0.9575   0.9519   <--
0.30       0.9515   0.9509   0.9645   0.9401   0.9385   0.9636  
0.35       0.9521   0.9515   0.9662   0.9398   0.9381   0.9652  
0.40       0.9537   0.9529   0.9707   0.9388   0.9367   0.9697  
0.45       0.9505   0.9494   0.9741   0.9303   0.9270   0.9729  
0.50       0.9338   0.9307   0.9778   0.8987   0.8898   0.9757  
0.55       0.8887   0.8752   0.9963   0.8198   0.7810   0.9953  
0.60       0.8878   0.8741   0.9969   0.8184   0.7787   0.9961  
0.65       0.8880   0.8742   0.9978   0.8182   0.7783   0.9972  
0.70       0.8875   0.8735   0.9979   0.8174   0.7771   0.9973  
0.75       0.8829   0.8676   0.9982   0.8111   0.7675   0.9977  
0.80       0.8813   0.8655   0.9986   0.8088   0.7639   0.9982  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.9545, F1=0.9547, Normal Recall=0.9516, Normal Precision=0.9572, Attack Recall=0.9575, Attack Precision=0.9519

```

