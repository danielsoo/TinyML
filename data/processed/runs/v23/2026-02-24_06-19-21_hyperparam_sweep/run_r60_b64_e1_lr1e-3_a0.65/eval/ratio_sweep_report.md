# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-24 14:33:54 |

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
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0017 | 0.1015 | 0.2013 | 0.3012 | 0.4012 | 0.5007 | 0.6007 | 0.7005 | 0.8002 | 0.9001 | 0.9999 |
| noQAT+PTQ | 0.0521 | 0.1469 | 0.2418 | 0.3362 | 0.4319 | 0.5260 | 0.6211 | 0.7160 | 0.8101 | 0.9056 | 1.0000 |
| saved_model_traditional_qat | 0.9659 | 0.9638 | 0.9617 | 0.9598 | 0.9573 | 0.9552 | 0.9534 | 0.9509 | 0.9489 | 0.9466 | 0.9447 |
| QAT+PTQ | 0.0005 | 0.1005 | 0.2005 | 0.3005 | 0.4004 | 0.5003 | 0.6003 | 0.7001 | 0.8000 | 0.8999 | 0.9999 |
| Compressed (QAT) | 0.9732 | 0.9670 | 0.9601 | 0.9535 | 0.9464 | 0.9398 | 0.9329 | 0.9258 | 0.9190 | 0.9122 | 0.9054 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1821 | 0.3337 | 0.4620 | 0.5719 | 0.6670 | 0.7503 | 0.8237 | 0.8890 | 0.9474 | 1.0000 |
| noQAT+PTQ | 0.0000 | 0.1899 | 0.3454 | 0.4748 | 0.5847 | 0.6784 | 0.7600 | 0.8314 | 0.8939 | 0.9502 | 1.0000 |
| saved_model_traditional_qat | 0.0000 | 0.8394 | 0.9079 | 0.9337 | 0.9466 | 0.9547 | 0.9605 | 0.9642 | 0.9673 | 0.9695 | 0.9715 |
| QAT+PTQ | 0.0000 | 0.1819 | 0.3334 | 0.4617 | 0.5716 | 0.6668 | 0.7501 | 0.8236 | 0.8889 | 0.9473 | 1.0000 |
| Compressed (QAT) | 0.0000 | 0.8460 | 0.9008 | 0.9212 | 0.9311 | 0.9376 | 0.9418 | 0.9447 | 0.9471 | 0.9489 | 0.9504 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0017 | 0.0017 | 0.0017 | 0.0018 | 0.0020 | 0.0015 | 0.0018 | 0.0017 | 0.0012 | 0.0014 | 0.0000 |
| noQAT+PTQ | 0.0521 | 0.0521 | 0.0522 | 0.0518 | 0.0531 | 0.0519 | 0.0527 | 0.0535 | 0.0505 | 0.0562 | 0.0000 |
| saved_model_traditional_qat | 0.9659 | 0.9659 | 0.9659 | 0.9662 | 0.9658 | 0.9658 | 0.9664 | 0.9656 | 0.9655 | 0.9637 | 0.0000 |
| QAT+PTQ | 0.0005 | 0.0006 | 0.0006 | 0.0007 | 0.0007 | 0.0006 | 0.0008 | 0.0006 | 0.0003 | 0.0002 | 0.0000 |
| Compressed (QAT) | 0.9732 | 0.9738 | 0.9738 | 0.9741 | 0.9738 | 0.9741 | 0.9740 | 0.9733 | 0.9732 | 0.9731 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0017 | 0.0000 | 0.0000 | 0.0000 | 0.0017 | 1.0000 |
| 90 | 10 | 299,940 | 0.1015 | 0.1001 | 0.9999 | 0.1821 | 0.0017 | 0.9957 |
| 80 | 20 | 291,350 | 0.2013 | 0.2003 | 0.9999 | 0.3337 | 0.0017 | 0.9923 |
| 70 | 30 | 194,230 | 0.3012 | 0.3004 | 0.9999 | 0.4620 | 0.0018 | 0.9879 |
| 60 | 40 | 145,675 | 0.4012 | 0.4005 | 0.9999 | 0.5719 | 0.0020 | 0.9833 |
| 50 | 50 | 116,540 | 0.5007 | 0.5004 | 0.9999 | 0.6670 | 0.0015 | 0.9674 |
| 40 | 60 | 97,115 | 0.6007 | 0.6004 | 0.9999 | 0.7503 | 0.0018 | 0.9589 |
| 30 | 70 | 83,240 | 0.7005 | 0.7003 | 0.9999 | 0.8237 | 0.0017 | 0.9333 |
| 20 | 80 | 72,835 | 0.8002 | 0.8002 | 0.9999 | 0.8890 | 0.0012 | 0.8500 |
| 10 | 90 | 64,740 | 0.9001 | 0.9001 | 0.9999 | 0.9474 | 0.0014 | 0.7500 |
| 0 | 100 | 58,270 | 0.9999 | 1.0000 | 0.9999 | 1.0000 | 0.0000 | 0.0000 |

### noQAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0521 | 0.0000 | 0.0000 | 0.0000 | 0.0521 | 1.0000 |
| 90 | 10 | 299,940 | 0.1469 | 0.1049 | 1.0000 | 0.1899 | 0.0521 | 1.0000 |
| 80 | 20 | 291,350 | 0.2418 | 0.2087 | 1.0000 | 0.3454 | 0.0522 | 1.0000 |
| 70 | 30 | 194,230 | 0.3362 | 0.3113 | 1.0000 | 0.4748 | 0.0518 | 1.0000 |
| 60 | 40 | 145,675 | 0.4319 | 0.4132 | 1.0000 | 0.5847 | 0.0531 | 1.0000 |
| 50 | 50 | 116,540 | 0.5260 | 0.5133 | 1.0000 | 0.6784 | 0.0519 | 1.0000 |
| 40 | 60 | 97,115 | 0.6211 | 0.6129 | 1.0000 | 0.7600 | 0.0527 | 1.0000 |
| 30 | 70 | 83,240 | 0.7160 | 0.7114 | 1.0000 | 0.8314 | 0.0535 | 1.0000 |
| 20 | 80 | 72,835 | 0.8101 | 0.8082 | 1.0000 | 0.8939 | 0.0505 | 1.0000 |
| 10 | 90 | 64,740 | 0.9056 | 0.9051 | 1.0000 | 0.9502 | 0.0562 | 1.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### saved_model_traditional_qat

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9659 | 0.0000 | 0.0000 | 0.0000 | 0.9659 | 1.0000 |
| 90 | 10 | 299,940 | 0.9638 | 0.7548 | 0.9454 | 0.8394 | 0.9659 | 0.9938 |
| 80 | 20 | 291,350 | 0.9617 | 0.8739 | 0.9447 | 0.9079 | 0.9659 | 0.9859 |
| 70 | 30 | 194,230 | 0.9598 | 0.9230 | 0.9447 | 0.9337 | 0.9662 | 0.9760 |
| 60 | 40 | 145,675 | 0.9573 | 0.9485 | 0.9447 | 0.9466 | 0.9658 | 0.9632 |
| 50 | 50 | 116,540 | 0.9552 | 0.9650 | 0.9447 | 0.9547 | 0.9658 | 0.9458 |
| 40 | 60 | 97,115 | 0.9534 | 0.9768 | 0.9447 | 0.9605 | 0.9664 | 0.9209 |
| 30 | 70 | 83,240 | 0.9509 | 0.9846 | 0.9447 | 0.9642 | 0.9656 | 0.8821 |
| 20 | 80 | 72,835 | 0.9489 | 0.9910 | 0.9447 | 0.9673 | 0.9655 | 0.8136 |
| 10 | 90 | 64,740 | 0.9466 | 0.9957 | 0.9447 | 0.9695 | 0.9637 | 0.6593 |
| 0 | 100 | 58,270 | 0.9447 | 1.0000 | 0.9447 | 0.9715 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0005 | 0.0000 | 0.0000 | 0.0000 | 0.0005 | 1.0000 |
| 90 | 10 | 299,940 | 0.1005 | 0.1001 | 1.0000 | 0.1819 | 0.0006 | 0.9939 |
| 80 | 20 | 291,350 | 0.2005 | 0.2001 | 0.9999 | 0.3334 | 0.0006 | 0.9650 |
| 70 | 30 | 194,230 | 0.3005 | 0.3001 | 0.9999 | 0.4617 | 0.0007 | 0.9510 |
| 60 | 40 | 145,675 | 0.4004 | 0.4001 | 0.9999 | 0.5716 | 0.0007 | 0.9231 |
| 50 | 50 | 116,540 | 0.5003 | 0.5001 | 0.9999 | 0.6668 | 0.0006 | 0.8810 |
| 40 | 60 | 97,115 | 0.6003 | 0.6002 | 0.9999 | 0.7501 | 0.0008 | 0.8649 |
| 30 | 70 | 83,240 | 0.7001 | 0.7001 | 0.9999 | 0.8236 | 0.0006 | 0.7368 |
| 20 | 80 | 72,835 | 0.8000 | 0.8000 | 0.9999 | 0.8889 | 0.0003 | 0.4444 |
| 10 | 90 | 64,740 | 0.8999 | 0.9000 | 0.9999 | 0.9473 | 0.0002 | 0.1667 |
| 0 | 100 | 58,270 | 0.9999 | 1.0000 | 0.9999 | 1.0000 | 0.0000 | 0.0000 |

### Compressed (QAT)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9732 | 0.0000 | 0.0000 | 0.0000 | 0.9732 | 1.0000 |
| 90 | 10 | 299,940 | 0.9670 | 0.7935 | 0.9060 | 0.8460 | 0.9738 | 0.9894 |
| 80 | 20 | 291,350 | 0.9601 | 0.8962 | 0.9054 | 0.9008 | 0.9738 | 0.9763 |
| 70 | 30 | 194,230 | 0.9535 | 0.9375 | 0.9054 | 0.9212 | 0.9741 | 0.9601 |
| 60 | 40 | 145,675 | 0.9464 | 0.9583 | 0.9054 | 0.9311 | 0.9738 | 0.9392 |
| 50 | 50 | 116,540 | 0.9398 | 0.9722 | 0.9054 | 0.9376 | 0.9741 | 0.9115 |
| 40 | 60 | 97,115 | 0.9329 | 0.9812 | 0.9054 | 0.9418 | 0.9740 | 0.8729 |
| 30 | 70 | 83,240 | 0.9258 | 0.9875 | 0.9054 | 0.9447 | 0.9733 | 0.8152 |
| 20 | 80 | 72,835 | 0.9190 | 0.9927 | 0.9055 | 0.9471 | 0.9732 | 0.7202 |
| 10 | 90 | 64,740 | 0.9122 | 0.9967 | 0.9055 | 0.9489 | 0.9731 | 0.5335 |
| 0 | 100 | 58,270 | 0.9054 | 1.0000 | 0.9054 | 0.9504 | 0.0000 | 0.0000 |


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
0.15       0.1004   0.1819   0.0005   1.0000   1.0000   0.1000  
0.20       0.1008   0.1819   0.0009   1.0000   1.0000   0.1001  
0.25       0.1012   0.1820   0.0014   1.0000   1.0000   0.1001  
0.30       0.1015   0.1821   0.0017   0.9978   1.0000   0.1002  
0.35       0.1020   0.1820   0.0024   0.9422   0.9987   0.1001  
0.40       0.1100   0.1832   0.0113   0.9832   0.9983   0.1009  
0.45       0.1513   0.1883   0.0587   0.9715   0.9845   0.1041  
0.50       0.7145   0.1927   0.7560   0.9117   0.3409   0.1344   <--
0.55       0.8574   0.0667   0.9470   0.8998   0.0510   0.0966  
0.60       0.8809   0.0065   0.9784   0.8984   0.0039   0.0195  
0.65       0.8908   0.0053   0.9894   0.8993   0.0029   0.0296  
0.70       0.8971   0.0053   0.9965   0.8999   0.0027   0.0803  
0.75       0.8991   0.0009   0.9990   0.9000   0.0005   0.0484  
0.80       0.8999   0.0001   0.9998   0.9000   0.0000   0.0238  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.7145, F1=0.1927, Normal Recall=0.7560, Normal Precision=0.9117, Attack Recall=0.3409, Attack Precision=0.1344

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
0.15       0.2004   0.3334   0.0005   0.9910   1.0000   0.2001  
0.20       0.2007   0.3335   0.0009   0.9953   1.0000   0.2001  
0.25       0.2011   0.3336   0.0014   0.9939   1.0000   0.2002  
0.30       0.2014   0.3337   0.0017   0.9927   0.9999   0.2003  
0.35       0.2017   0.3335   0.0025   0.8832   0.9987   0.2002  
0.40       0.2087   0.3354   0.0114   0.9605   0.9981   0.2015  
0.45       0.2439   0.3423   0.0590   0.9354   0.9837   0.2072   <--
0.50       0.6735   0.2948   0.7565   0.8212   0.3413   0.2595  
0.55       0.7679   0.0807   0.9472   0.7997   0.0509   0.1942  
0.60       0.7836   0.0073   0.9785   0.7971   0.0040   0.0442  
0.65       0.7923   0.0060   0.9896   0.7988   0.0031   0.0701  
0.70       0.7978   0.0058   0.9966   0.7999   0.0030   0.1771  
0.75       0.7993   0.0016   0.9989   0.8000   0.0008   0.1575  
0.80       0.7999   0.0000   0.9998   0.8000   0.0000   0.0278  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.2439, F1=0.3423, Normal Recall=0.0590, Normal Precision=0.9354, Attack Recall=0.9837, Attack Precision=0.2072

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
0.15       0.3003   0.4616   0.0004   0.9833   1.0000   0.3001  
0.20       0.3006   0.4617   0.0009   0.9917   1.0000   0.3002  
0.25       0.3009   0.4619   0.0013   0.9890   1.0000   0.3003  
0.30       0.3012   0.4619   0.0017   0.9873   0.9999   0.3003  
0.35       0.3013   0.4617   0.0024   0.8089   0.9987   0.3002  
0.40       0.3073   0.4637   0.0113   0.9335   0.9981   0.3020  
0.45       0.3364   0.4707   0.0590   0.8941   0.9837   0.3094   <--
0.50       0.6314   0.3571   0.7557   0.7280   0.3413   0.3745  
0.55       0.6782   0.0867   0.9470   0.6995   0.0509   0.2918  
0.60       0.6861   0.0076   0.9784   0.6962   0.0040   0.0734  
0.65       0.6935   0.0061   0.9894   0.6984   0.0031   0.1126  
0.70       0.6984   0.0059   0.9964   0.6999   0.0030   0.2637  
0.75       0.6995   0.0016   0.9989   0.6999   0.0008   0.2421  
0.80       0.6999   0.0000   0.9998   0.7000   0.0000   0.0455  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.3364, F1=0.4707, Normal Recall=0.0590, Normal Precision=0.8941, Attack Recall=0.9837, Attack Precision=0.3094

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
0.15       0.4002   0.5715   0.0004   0.9730   1.0000   0.4001  
0.20       0.4005   0.5716   0.0008   0.9863   1.0000   0.4002  
0.25       0.4007   0.5717   0.0012   0.9817   1.0000   0.4003  
0.30       0.4010   0.5718   0.0017   0.9797   0.9999   0.4004  
0.35       0.4009   0.5715   0.0024   0.7308   0.9987   0.4003  
0.40       0.4061   0.5735   0.0114   0.9016   0.9981   0.4023  
0.45       0.4289   0.5795   0.0590   0.8444   0.9837   0.4107   <--
0.50       0.5895   0.3995   0.7550   0.6323   0.3413   0.4815  
0.55       0.5888   0.0901   0.9475   0.5996   0.0509   0.3925  
0.60       0.5888   0.0077   0.9786   0.5958   0.0040   0.1105  
0.65       0.5952   0.0062   0.9898   0.5983   0.0031   0.1707  
0.70       0.5992   0.0059   0.9967   0.5999   0.0030   0.3737  
0.75       0.5997   0.0016   0.9990   0.6000   0.0008   0.3485  
0.80       0.5999   0.0000   0.9998   0.6000   0.0000   0.0625  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.4289, F1=0.5795, Normal Recall=0.0590, Normal Precision=0.8444, Attack Recall=0.9837, Attack Precision=0.4107

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
0.15       0.5002   0.6668   0.0005   0.9655   1.0000   0.5001  
0.20       0.5004   0.6668   0.0008   0.9796   1.0000   0.5002  
0.25       0.5006   0.6669   0.0011   0.9710   1.0000   0.5003  
0.30       0.5008   0.6670   0.0016   0.9697   0.9999   0.5004  
0.35       0.5005   0.6666   0.0024   0.6435   0.9987   0.5003  
0.40       0.5048   0.6684   0.0115   0.8603   0.9981   0.5024  
0.45       0.5214   0.6727   0.0591   0.7837   0.9837   0.5111   <--
0.50       0.5479   0.4302   0.7544   0.5339   0.3413   0.5815  
0.55       0.4996   0.0923   0.9483   0.4998   0.0509   0.4960  
0.60       0.4916   0.0078   0.9792   0.4958   0.0040   0.1609  
0.65       0.4965   0.0062   0.9899   0.4982   0.0031   0.2367  
0.70       0.4998   0.0059   0.9965   0.4999   0.0030   0.4613  
0.75       0.4999   0.0016   0.9991   0.5000   0.0008   0.4554  
0.80       0.4999   0.0000   0.9999   0.5000   0.0000   0.1250  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.5214, F1=0.6727, Normal Recall=0.0591, Normal Precision=0.7837, Attack Recall=0.9837, Attack Precision=0.5111

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
0.15       0.1129   0.1840   0.0143   1.0000   1.0000   0.1013  
0.20       0.1187   0.1850   0.0208   1.0000   1.0000   0.1019  
0.25       0.1314   0.1872   0.0348   1.0000   1.0000   0.1032  
0.30       0.1469   0.1899   0.0521   1.0000   1.0000   0.1049  
0.35       0.1772   0.1955   0.0858   1.0000   1.0000   0.1084  
0.40       0.2735   0.2159   0.1928   1.0000   1.0000   0.1210  
0.45       0.5748   0.3199   0.5276   1.0000   0.9998   0.1904  
0.50       0.7845   0.4806   0.7608   0.9996   0.9974   0.3166  
0.55       0.9485   0.7618   0.9624   0.9800   0.8235   0.7087   <--
0.60       0.9503   0.7017   0.9911   0.9554   0.5840   0.8789  
0.65       0.9471   0.6485   0.9982   0.9461   0.4878   0.9671  
0.70       0.9475   0.6466   0.9993   0.9454   0.4805   0.9879  
0.75       0.9465   0.6357   0.9997   0.9441   0.4671   0.9945  
0.80       0.9413   0.5858   0.9998   0.9389   0.4148   0.9967  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9485, F1=0.7618, Normal Recall=0.9624, Normal Precision=0.9800, Attack Recall=0.8235, Attack Precision=0.7087

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
0.15       0.2113   0.3365   0.0142   1.0000   1.0000   0.2023  
0.20       0.2166   0.3380   0.0207   1.0000   1.0000   0.2034  
0.25       0.2278   0.3412   0.0347   1.0000   1.0000   0.2057  
0.30       0.2415   0.3453   0.0518   1.0000   1.0000   0.2087  
0.35       0.2684   0.3535   0.0855   0.9999   1.0000   0.2147  
0.40       0.3541   0.3824   0.1927   1.0000   1.0000   0.2364  
0.45       0.6221   0.5141   0.5277   0.9999   0.9997   0.3460  
0.50       0.8083   0.6754   0.7610   0.9991   0.9972   0.5106  
0.55       0.9345   0.8342   0.9623   0.9561   0.8233   0.8453   <--
0.60       0.9099   0.7220   0.9912   0.9052   0.5849   0.9431  
0.65       0.8961   0.6524   0.9982   0.8863   0.4876   0.9852  
0.70       0.8955   0.6478   0.9993   0.8850   0.4803   0.9945  
0.75       0.8931   0.6359   0.9997   0.8823   0.4667   0.9975  
0.80       0.8826   0.5850   0.9998   0.8721   0.4137   0.9985  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9345, F1=0.8342, Normal Recall=0.9623, Normal Precision=0.9561, Attack Recall=0.8233, Attack Precision=0.8453

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
0.15       0.3100   0.4651   0.0143   1.0000   1.0000   0.3030  
0.20       0.3149   0.4669   0.0213   1.0000   1.0000   0.3045  
0.25       0.3248   0.4705   0.0354   1.0000   1.0000   0.3076  
0.30       0.3369   0.4750   0.0528   1.0000   1.0000   0.3115  
0.35       0.3605   0.4841   0.0865   0.9999   1.0000   0.3193  
0.40       0.4358   0.5154   0.1940   0.9999   1.0000   0.3471  
0.45       0.6702   0.6452   0.5290   0.9998   0.9997   0.4763  
0.50       0.8314   0.7802   0.7604   0.9984   0.9972   0.6408  
0.55       0.9203   0.8611   0.9619   0.9270   0.8233   0.9025   <--
0.60       0.8691   0.7283   0.9909   0.8478   0.5849   0.9648  
0.65       0.8450   0.6537   0.9981   0.8197   0.4876   0.9911  
0.70       0.8436   0.6483   0.9993   0.8178   0.4803   0.9967  
0.75       0.8398   0.6361   0.9997   0.8139   0.4667   0.9986  
0.80       0.8240   0.5851   0.9999   0.7992   0.4137   0.9993  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9203, F1=0.8611, Normal Recall=0.9619, Normal Precision=0.9270, Attack Recall=0.8233, Attack Precision=0.9025

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
0.15       0.4083   0.5748   0.0138   1.0000   1.0000   0.4033  
0.20       0.4125   0.5766   0.0208   1.0000   1.0000   0.4051  
0.25       0.4211   0.5802   0.0351   1.0000   1.0000   0.4086  
0.30       0.4314   0.5845   0.0523   1.0000   1.0000   0.4129  
0.35       0.4518   0.5934   0.0863   0.9999   1.0000   0.4218  
0.40       0.5165   0.6233   0.1942   0.9999   1.0000   0.4527  
0.45       0.7172   0.7388   0.5288   0.9996   0.9997   0.5858  
0.50       0.8550   0.8462   0.7603   0.9975   0.9972   0.7350  
0.55       0.9065   0.8757   0.9619   0.8909   0.8233   0.9351   <--
0.60       0.8287   0.7320   0.9912   0.7817   0.5849   0.9780  
0.65       0.7940   0.6544   0.9982   0.7451   0.4876   0.9946  
0.70       0.7918   0.6485   0.9994   0.7426   0.4803   0.9981  
0.75       0.7865   0.6362   0.9997   0.7376   0.4667   0.9990  
0.80       0.7654   0.5852   0.9998   0.7189   0.4137   0.9994  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9065, F1=0.8757, Normal Recall=0.9619, Normal Precision=0.8909, Attack Recall=0.8233, Attack Precision=0.9351

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
0.15       0.5069   0.6697   0.0137   1.0000   1.0000   0.5035  
0.20       0.5103   0.6713   0.0205   1.0000   1.0000   0.5052  
0.25       0.5173   0.6745   0.0346   1.0000   1.0000   0.5088  
0.30       0.5259   0.6784   0.0517   1.0000   1.0000   0.5133  
0.35       0.5430   0.6863   0.0860   0.9998   1.0000   0.5225  
0.40       0.5965   0.7125   0.1930   0.9998   1.0000   0.5534  
0.45       0.7643   0.8092   0.5289   0.9994   0.9997   0.6797  
0.50       0.8788   0.8917   0.7605   0.9963   0.9972   0.8063   <--
0.55       0.8926   0.8846   0.9618   0.8448   0.8233   0.9556  
0.60       0.7880   0.7340   0.9911   0.7048   0.5849   0.9851  
0.65       0.7429   0.6548   0.9982   0.6608   0.4876   0.9964  
0.70       0.7399   0.6487   0.9994   0.6579   0.4803   0.9987  
0.75       0.7332   0.6362   0.9997   0.6521   0.4667   0.9993  
0.80       0.7068   0.5852   0.9998   0.6304   0.4137   0.9996  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.8788, F1=0.8917, Normal Recall=0.7605, Normal Precision=0.9963, Attack Recall=0.9972, Attack Precision=0.8063

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
0.15       0.9372   0.7561   0.9333   0.9968   0.9727   0.6184  
0.20       0.9406   0.7659   0.9371   0.9967   0.9717   0.6320  
0.25       0.9632   0.8377   0.9647   0.9942   0.9495   0.7495  
0.30       0.9639   0.8399   0.9659   0.9939   0.9464   0.7549  
0.35       0.9663   0.8476   0.9696   0.9928   0.9369   0.7738  
0.40       0.9685   0.8547   0.9730   0.9918   0.9276   0.7924  
0.45       0.9690   0.8568   0.9737   0.9918   0.9271   0.7963  
0.50       0.9693   0.8580   0.9740   0.9917   0.9270   0.7987  
0.55       0.9698   0.8598   0.9746   0.9917   0.9266   0.8020  
0.60       0.9705   0.8515   0.9843   0.9830   0.8464   0.8566  
0.65       0.9733   0.8574   0.9924   0.9783   0.8016   0.9215  
0.70       0.9743   0.8605   0.9947   0.9772   0.7910   0.9433  
0.75       0.9748   0.8609   0.9967   0.9759   0.7785   0.9628  
0.80       0.9751   0.8610   0.9976   0.9753   0.7723   0.9727   <--
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.8
  At threshold 0.8: Accuracy=0.9751, F1=0.8610, Normal Recall=0.9976, Normal Precision=0.9753, Attack Recall=0.7723, Attack Precision=0.9727

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
0.15       0.9409   0.8680   0.9335   0.9923   0.9709   0.7848  
0.20       0.9438   0.8735   0.9373   0.9920   0.9699   0.7946  
0.25       0.9615   0.9078   0.9649   0.9867   0.9478   0.8710  
0.30       0.9618   0.9081   0.9661   0.9859   0.9447   0.8743  
0.35       0.9629   0.9097   0.9697   0.9836   0.9353   0.8854  
0.40       0.9638   0.9109   0.9731   0.9815   0.9266   0.8958  
0.45       0.9641   0.9118   0.9737   0.9814   0.9260   0.8979  
0.50       0.9644   0.9124   0.9741   0.9813   0.9259   0.8993  
0.55       0.9648   0.9132   0.9746   0.9812   0.9255   0.9011   <--
0.60       0.9564   0.8857   0.9843   0.9621   0.8449   0.9306  
0.65       0.9540   0.8742   0.9924   0.9521   0.8001   0.9635  
0.70       0.9537   0.8721   0.9947   0.9498   0.7896   0.9740  
0.75       0.9529   0.8684   0.9967   0.9471   0.7775   0.9833  
0.80       0.9524   0.8663   0.9976   0.9458   0.7715   0.9878  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9648, F1=0.9132, Normal Recall=0.9746, Normal Precision=0.9812, Attack Recall=0.9255, Attack Precision=0.9011

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
0.15       0.9449   0.9136   0.9338   0.9868   0.9709   0.8628  
0.20       0.9474   0.9171   0.9377   0.9864   0.9699   0.8697  
0.25       0.9597   0.9338   0.9648   0.9773   0.9478   0.9202   <--
0.30       0.9595   0.9333   0.9659   0.9760   0.9447   0.9222  
0.35       0.9593   0.9324   0.9696   0.9722   0.9353   0.9294  
0.40       0.9589   0.9312   0.9728   0.9687   0.9266   0.9359  
0.45       0.9592   0.9316   0.9735   0.9685   0.9260   0.9373  
0.50       0.9595   0.9320   0.9738   0.9684   0.9259   0.9381  
0.55       0.9597   0.9323   0.9743   0.9683   0.9255   0.9392  
0.60       0.9425   0.8981   0.9843   0.9367   0.8449   0.9584  
0.65       0.9347   0.8803   0.9924   0.9205   0.8001   0.9784  
0.70       0.9332   0.8764   0.9947   0.9169   0.7896   0.9847  
0.75       0.9309   0.8710   0.9966   0.9127   0.7775   0.9900  
0.80       0.9297   0.8682   0.9975   0.9106   0.7715   0.9926  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.9597, F1=0.9338, Normal Recall=0.9648, Normal Precision=0.9773, Attack Recall=0.9478, Attack Precision=0.9202

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
0.15       0.9486   0.9380   0.9338   0.9796   0.9709   0.9072  
0.20       0.9507   0.9403   0.9379   0.9791   0.9699   0.9124  
0.25       0.9582   0.9477   0.9651   0.9652   0.9478   0.9476   <--
0.30       0.9576   0.9469   0.9662   0.9632   0.9447   0.9491  
0.35       0.9560   0.9444   0.9698   0.9574   0.9353   0.9537  
0.40       0.9544   0.9420   0.9729   0.9521   0.9266   0.9579  
0.45       0.9545   0.9421   0.9735   0.9518   0.9260   0.9588  
0.50       0.9546   0.9423   0.9738   0.9517   0.9259   0.9593  
0.55       0.9548   0.9424   0.9743   0.9515   0.9255   0.9600  
0.60       0.9285   0.9044   0.9843   0.9049   0.8449   0.9728  
0.65       0.9155   0.8833   0.9924   0.8816   0.8001   0.9859  
0.70       0.9126   0.8784   0.9946   0.8764   0.7896   0.9898  
0.75       0.9089   0.8722   0.9965   0.8704   0.7775   0.9933  
0.80       0.9070   0.8690   0.9973   0.8675   0.7715   0.9948  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.9582, F1=0.9477, Normal Recall=0.9651, Normal Precision=0.9652, Attack Recall=0.9478, Attack Precision=0.9476

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
0.15       0.9525   0.9534   0.9341   0.9698   0.9709   0.9365  
0.20       0.9539   0.9547   0.9380   0.9689   0.9699   0.9399  
0.25       0.9566   0.9562   0.9653   0.9487   0.9478   0.9647   <--
0.30       0.9556   0.9551   0.9664   0.9458   0.9447   0.9657  
0.35       0.9528   0.9519   0.9702   0.9375   0.9353   0.9691  
0.40       0.9499   0.9487   0.9732   0.9298   0.9266   0.9719  
0.45       0.9499   0.9487   0.9738   0.9294   0.9260   0.9725  
0.50       0.9500   0.9487   0.9741   0.9293   0.9259   0.9728  
0.55       0.9500   0.9488   0.9746   0.9290   0.9255   0.9733  
0.60       0.9148   0.9084   0.9847   0.8639   0.8449   0.9822  
0.65       0.8963   0.8852   0.9924   0.8323   0.8001   0.9906  
0.70       0.8921   0.8798   0.9947   0.8254   0.7896   0.9933  
0.75       0.8870   0.8731   0.9965   0.8175   0.7775   0.9955  
0.80       0.8844   0.8697   0.9973   0.8136   0.7715   0.9966  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.9566, F1=0.9562, Normal Recall=0.9653, Normal Precision=0.9487, Attack Recall=0.9478, Attack Precision=0.9647

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
0.15       0.1004   0.1819   0.0005   1.0000   1.0000   0.1000  
0.20       0.1004   0.1819   0.0005   1.0000   1.0000   0.1000  
0.25       0.1005   0.1819   0.0005   1.0000   1.0000   0.1000  
0.30       0.1005   0.1819   0.0006   0.9939   1.0000   0.1001  
0.35       0.1029   0.1822   0.0032   0.9865   0.9996   0.1003  
0.40       0.1069   0.1825   0.0080   0.9562   0.9967   0.1004  
0.45       0.1171   0.1839   0.0195   0.9722   0.9950   0.1013   <--
0.50       0.7403   0.0547   0.8142   0.8879   0.0751   0.0430  
0.55       0.8982   0.0071   0.9976   0.9001   0.0036   0.1457  
0.60       0.8996   0.0000   0.9995   0.9000   0.0000   0.0000  
0.65       0.8997   0.0000   0.9996   0.9000   0.0000   0.0000  
0.70       0.8999   0.0000   0.9999   0.9000   0.0000   0.0000  
0.75       0.8999   0.0000   0.9999   0.9000   0.0000   0.0000  
0.80       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.1171, F1=0.1839, Normal Recall=0.0195, Normal Precision=0.9722, Attack Recall=0.9950, Attack Precision=0.1013

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
0.15       0.2004   0.3334   0.0005   0.9908   1.0000   0.2001  
0.20       0.2004   0.3334   0.0005   0.9913   1.0000   0.2001  
0.25       0.2004   0.3334   0.0006   0.9701   0.9999   0.2001  
0.30       0.2005   0.3334   0.0006   0.9660   0.9999   0.2001  
0.35       0.2025   0.3339   0.0032   0.9580   0.9994   0.2004  
0.40       0.2057   0.3342   0.0080   0.9081   0.9968   0.2008  
0.45       0.2146   0.3363   0.0195   0.9372   0.9948   0.2023   <--
0.50       0.6663   0.0824   0.8142   0.7788   0.0749   0.0915  
0.55       0.7988   0.0075   0.9976   0.8002   0.0038   0.2824  
0.60       0.7996   0.0001   0.9995   0.7999   0.0001   0.0238  
0.65       0.7997   0.0000   0.9996   0.7999   0.0000   0.0105  
0.70       0.7999   0.0000   0.9999   0.8000   0.0000   0.0357  
0.75       0.7999   0.0000   0.9999   0.8000   0.0000   0.0000  
0.80       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.2146, F1=0.3363, Normal Recall=0.0195, Normal Precision=0.9372, Attack Recall=0.9948, Attack Precision=0.2023

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
0.15       0.3003   0.4616   0.0004   0.9831   1.0000   0.3001  
0.20       0.3003   0.4616   0.0004   0.9836   1.0000   0.3001  
0.25       0.3003   0.4616   0.0005   0.9429   0.9999   0.3001  
0.30       0.3004   0.4616   0.0006   0.9383   0.9999   0.3001  
0.35       0.3020   0.4621   0.0031   0.9270   0.9994   0.3005  
0.40       0.3045   0.4623   0.0078   0.8500   0.9968   0.3010  
0.45       0.3120   0.4645   0.0193   0.8960   0.9948   0.3030   <--
0.50       0.5918   0.0991   0.8133   0.6723   0.0749   0.1467  
0.55       0.6995   0.0075   0.9976   0.7003   0.0038   0.4044  
0.60       0.6996   0.0001   0.9995   0.6999   0.0001   0.0405  
0.65       0.6997   0.0000   0.9996   0.6999   0.0000   0.0196  
0.70       0.6999   0.0000   0.9999   0.7000   0.0000   0.0588  
0.75       0.6999   0.0000   0.9999   0.7000   0.0000   0.0000  
0.80       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.3120, F1=0.4645, Normal Recall=0.0193, Normal Precision=0.8960, Attack Recall=0.9948, Attack Precision=0.3030

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
0.15       0.4002   0.5715   0.0004   0.9714   1.0000   0.4001  
0.20       0.4002   0.5715   0.0004   0.9714   1.0000   0.4001  
0.25       0.4002   0.5715   0.0004   0.9024   0.9999   0.4001  
0.30       0.4003   0.5715   0.0005   0.8980   0.9999   0.4001  
0.35       0.4016   0.5719   0.0030   0.8889   0.9994   0.4006  
0.40       0.4034   0.5720   0.0079   0.7851   0.9968   0.4011  
0.45       0.4095   0.5740   0.0193   0.8467   0.9948   0.4034   <--
0.50       0.5179   0.1105   0.8132   0.5687   0.0749   0.2109  
0.55       0.6001   0.0076   0.9976   0.6003   0.0038   0.5163  
0.60       0.5997   0.0001   0.9995   0.5999   0.0001   0.0698  
0.65       0.5998   0.0000   0.9997   0.5999   0.0000   0.0345  
0.70       0.5999   0.0000   0.9999   0.6000   0.0000   0.0769  
0.75       0.5999   0.0000   0.9999   0.6000   0.0000   0.0000  
0.80       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.4095, F1=0.5740, Normal Recall=0.0193, Normal Precision=0.8467, Attack Recall=0.9948, Attack Precision=0.4034

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
0.15       0.5002   0.6667   0.0004   0.9565   1.0000   0.5001  
0.20       0.5002   0.6667   0.0004   0.9565   1.0000   0.5001  
0.25       0.5002   0.6667   0.0004   0.8571   0.9999   0.5001  
0.30       0.5002   0.6667   0.0005   0.8529   0.9999   0.5001  
0.35       0.5013   0.6671   0.0032   0.8493   0.9994   0.5007  
0.40       0.5024   0.6670   0.0080   0.7121   0.9968   0.5012  
0.45       0.5070   0.6686   0.0193   0.7864   0.9948   0.5036   <--
0.50       0.4441   0.1187   0.8133   0.4678   0.0749   0.2863  
0.55       0.5008   0.0076   0.9977   0.5004   0.0038   0.6271  
0.60       0.4998   0.0001   0.9995   0.4999   0.0001   0.0938  
0.65       0.4998   0.0000   0.9997   0.4999   0.0000   0.0476  
0.70       0.4999   0.0000   0.9998   0.5000   0.0000   0.0909  
0.75       0.4999   0.0000   0.9998   0.5000   0.0000   0.0000  
0.80       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.5070, F1=0.6686, Normal Recall=0.0193, Normal Precision=0.7864, Attack Recall=0.9948, Attack Precision=0.5036

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
0.15       0.9584   0.8171   0.9616   0.9920   0.9298   0.7288  
0.20       0.9646   0.8395   0.9691   0.9915   0.9249   0.7686  
0.25       0.9665   0.8455   0.9722   0.9905   0.9158   0.7853  
0.30       0.9670   0.8461   0.9738   0.9894   0.9061   0.7935  
0.35       0.9677   0.8484   0.9747   0.9893   0.9047   0.7987  
0.40       0.9675   0.8464   0.9754   0.9883   0.8960   0.8021  
0.45       0.9677   0.8469   0.9761   0.9879   0.8922   0.8059  
0.50       0.9678   0.8465   0.9767   0.9874   0.8880   0.8087  
0.55       0.9675   0.8435   0.9776   0.9862   0.8766   0.8127  
0.60       0.9642   0.8196   0.9810   0.9793   0.8130   0.8262  
0.65       0.9727   0.8486   0.9958   0.9744   0.7649   0.9528   <--
0.70       0.9727   0.8444   0.9984   0.9720   0.7413   0.9808  
0.75       0.9681   0.8119   0.9992   0.9665   0.6881   0.9898  
0.80       0.9591   0.7435   0.9998   0.9567   0.5928   0.9970  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.65
  At threshold 0.65: Accuracy=0.9727, F1=0.8486, Normal Recall=0.9958, Normal Precision=0.9744, Attack Recall=0.7649, Attack Precision=0.9528

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
0.15       0.9551   0.8921   0.9617   0.9817   0.9284   0.8585  
0.20       0.9600   0.9024   0.9691   0.9807   0.9238   0.8819  
0.25       0.9607   0.9031   0.9722   0.9786   0.9148   0.8917   <--
0.30       0.9602   0.9009   0.9739   0.9763   0.9054   0.8965  
0.35       0.9606   0.9017   0.9747   0.9760   0.9039   0.8994  
0.40       0.9594   0.8982   0.9755   0.9739   0.8953   0.9012  
0.45       0.9592   0.8972   0.9761   0.9729   0.8913   0.9032  
0.50       0.9587   0.8958   0.9767   0.9719   0.8870   0.9048  
0.55       0.9573   0.8913   0.9776   0.9692   0.8759   0.9071  
0.60       0.9473   0.8603   0.9810   0.9543   0.8122   0.9146  
0.65       0.9494   0.8578   0.9958   0.9440   0.7636   0.9785  
0.70       0.9469   0.8482   0.9984   0.9391   0.7411   0.9915  
0.75       0.9369   0.8133   0.9992   0.9275   0.6874   0.9956  
0.80       0.9181   0.7428   0.9998   0.9073   0.5913   0.9986  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.9607, F1=0.9031, Normal Recall=0.9722, Normal Precision=0.9786, Attack Recall=0.9148, Attack Precision=0.8917

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
0.15       0.9515   0.9199   0.9614   0.9691   0.9284   0.9116  
0.20       0.9553   0.9254   0.9688   0.9674   0.9238   0.9269   <--
0.25       0.9548   0.9239   0.9719   0.9638   0.9148   0.9332  
0.30       0.9531   0.9206   0.9736   0.9600   0.9054   0.9363  
0.35       0.9533   0.9207   0.9744   0.9595   0.9039   0.9381  
0.40       0.9512   0.9167   0.9752   0.9560   0.8953   0.9392  
0.45       0.9504   0.9152   0.9758   0.9544   0.8913   0.9404  
0.50       0.9495   0.9134   0.9763   0.9527   0.8870   0.9414  
0.55       0.9468   0.9081   0.9772   0.9484   0.8759   0.9428  
0.60       0.9303   0.8748   0.9809   0.9242   0.8121   0.9480  
0.65       0.9262   0.8613   0.9959   0.9076   0.7636   0.9876  
0.70       0.9212   0.8495   0.9985   0.9000   0.7410   0.9952  
0.75       0.9058   0.8140   0.9993   0.8818   0.6874   0.9977  
0.80       0.8773   0.7430   0.9999   0.8509   0.5913   0.9994  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9553, F1=0.9254, Normal Recall=0.9688, Normal Precision=0.9674, Attack Recall=0.9238, Attack Precision=0.9269

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
0.15       0.9483   0.9350   0.9616   0.9527   0.9284   0.9416  
0.20       0.9507   0.9375   0.9686   0.9502   0.9238   0.9515   <--
0.25       0.9490   0.9348   0.9718   0.9448   0.9148   0.9558  
0.30       0.9463   0.9309   0.9735   0.9392   0.9054   0.9579  
0.35       0.9462   0.9308   0.9744   0.9383   0.9039   0.9592  
0.40       0.9432   0.9266   0.9752   0.9332   0.8953   0.9600  
0.45       0.9420   0.9248   0.9758   0.9309   0.8913   0.9608  
0.50       0.9406   0.9227   0.9763   0.9283   0.8870   0.9615  
0.55       0.9366   0.9170   0.9771   0.9219   0.8759   0.9622  
0.60       0.9134   0.8824   0.9809   0.8868   0.8122   0.9660  
0.65       0.9030   0.8629   0.9959   0.8634   0.7636   0.9920  
0.70       0.8954   0.8501   0.9984   0.8526   0.7411   0.9967  
0.75       0.8745   0.8142   0.9993   0.8275   0.6874   0.9984  
0.80       0.8364   0.7431   0.9999   0.7858   0.5913   0.9997  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9507, F1=0.9375, Normal Recall=0.9686, Normal Precision=0.9502, Attack Recall=0.9238, Attack Precision=0.9515

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
0.15       0.9450   0.9441   0.9617   0.9307   0.9284   0.9604  
0.20       0.9463   0.9451   0.9688   0.9271   0.9238   0.9673   <--
0.25       0.9435   0.9418   0.9722   0.9194   0.9148   0.9705  
0.30       0.9397   0.9375   0.9739   0.9115   0.9054   0.9720  
0.35       0.9394   0.9371   0.9748   0.9103   0.9039   0.9729  
0.40       0.9354   0.9327   0.9755   0.9031   0.8953   0.9734  
0.45       0.9337   0.9307   0.9760   0.8998   0.8913   0.9738  
0.50       0.9318   0.9286   0.9766   0.8963   0.8870   0.9743  
0.55       0.9267   0.9227   0.9774   0.8874   0.8759   0.9748  
0.60       0.8968   0.8872   0.9814   0.8393   0.8122   0.9776  
0.65       0.8798   0.8640   0.9960   0.8082   0.7636   0.9948  
0.70       0.8697   0.8505   0.9984   0.7941   0.7411   0.9979  
0.75       0.8434   0.8144   0.9993   0.7617   0.6874   0.9990  
0.80       0.7956   0.7431   0.9999   0.7098   0.5913   0.9998  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9463, F1=0.9451, Normal Recall=0.9688, Normal Precision=0.9271, Attack Recall=0.9238, Attack Precision=0.9673

```

