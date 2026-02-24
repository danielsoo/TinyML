# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-24 08:08:38 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0014 | 0.1011 | 0.2008 | 0.3005 | 0.4002 | 0.4997 | 0.5995 | 0.6991 | 0.7987 | 0.8983 | 0.9980 |
| noQAT+PTQ | 0.0521 | 0.1469 | 0.2418 | 0.3362 | 0.4319 | 0.5260 | 0.6211 | 0.7160 | 0.8101 | 0.9056 | 1.0000 |
| saved_model_traditional_qat | 0.9713 | 0.9678 | 0.9637 | 0.9600 | 0.9560 | 0.9516 | 0.9481 | 0.9438 | 0.9398 | 0.9359 | 0.9319 |
| QAT+PTQ | 0.0001 | 0.1001 | 0.2001 | 0.3000 | 0.4001 | 0.5000 | 0.6000 | 0.7000 | 0.8000 | 0.9000 | 1.0000 |
| Compressed (QAT) | 0.9278 | 0.9300 | 0.9319 | 0.9352 | 0.9374 | 0.9393 | 0.9420 | 0.9444 | 0.9466 | 0.9488 | 0.9510 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1817 | 0.3331 | 0.4612 | 0.5710 | 0.6661 | 0.7494 | 0.8228 | 0.8880 | 0.9464 | 0.9990 |
| noQAT+PTQ | 0.0000 | 0.1899 | 0.3454 | 0.4748 | 0.5847 | 0.6784 | 0.7600 | 0.8314 | 0.8939 | 0.9502 | 1.0000 |
| saved_model_traditional_qat | 0.0000 | 0.8527 | 0.9113 | 0.9332 | 0.9443 | 0.9507 | 0.9556 | 0.9587 | 0.9612 | 0.9632 | 0.9648 |
| QAT+PTQ | 0.0000 | 0.1818 | 0.3333 | 0.4615 | 0.5714 | 0.6667 | 0.7500 | 0.8235 | 0.8889 | 0.9474 | 1.0000 |
| Compressed (QAT) | 0.0000 | 0.7310 | 0.8482 | 0.8980 | 0.9239 | 0.9400 | 0.9516 | 0.9599 | 0.9661 | 0.9710 | 0.9749 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0014 | 0.0015 | 0.0015 | 0.0016 | 0.0016 | 0.0015 | 0.0016 | 0.0017 | 0.0013 | 0.0009 | 0.0000 |
| noQAT+PTQ | 0.0521 | 0.0521 | 0.0522 | 0.0518 | 0.0531 | 0.0519 | 0.0527 | 0.0535 | 0.0505 | 0.0562 | 0.0000 |
| saved_model_traditional_qat | 0.9713 | 0.9717 | 0.9717 | 0.9720 | 0.9720 | 0.9713 | 0.9723 | 0.9714 | 0.9713 | 0.9711 | 0.0000 |
| QAT+PTQ | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0000 | 0.0000 |
| Compressed (QAT) | 0.9278 | 0.9277 | 0.9271 | 0.9284 | 0.9283 | 0.9276 | 0.9284 | 0.9292 | 0.9289 | 0.9293 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0014 | 0.0000 | 0.0000 | 0.0000 | 0.0014 | 1.0000 |
| 90 | 10 | 299,940 | 0.1011 | 0.0999 | 0.9977 | 0.1817 | 0.0015 | 0.8544 |
| 80 | 20 | 291,350 | 0.2008 | 0.1999 | 0.9980 | 0.3331 | 0.0015 | 0.7564 |
| 70 | 30 | 194,230 | 0.3005 | 0.2999 | 0.9980 | 0.4612 | 0.0016 | 0.6536 |
| 60 | 40 | 145,675 | 0.4002 | 0.3999 | 0.9980 | 0.5710 | 0.0016 | 0.5472 |
| 50 | 50 | 116,540 | 0.4997 | 0.4999 | 0.9980 | 0.6661 | 0.0015 | 0.4250 |
| 40 | 60 | 97,115 | 0.5995 | 0.5999 | 0.9980 | 0.7494 | 0.0016 | 0.3575 |
| 30 | 70 | 83,240 | 0.6991 | 0.6999 | 0.9980 | 0.8228 | 0.0017 | 0.2722 |
| 20 | 80 | 72,835 | 0.7987 | 0.7999 | 0.9980 | 0.8880 | 0.0013 | 0.1418 |
| 10 | 90 | 64,740 | 0.8983 | 0.8999 | 0.9980 | 0.9464 | 0.0009 | 0.0496 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9713 | 0.0000 | 0.0000 | 0.0000 | 0.9713 | 1.0000 |
| 90 | 10 | 299,940 | 0.9678 | 0.7854 | 0.9326 | 0.8527 | 0.9717 | 0.9924 |
| 80 | 20 | 291,350 | 0.9637 | 0.8916 | 0.9319 | 0.9113 | 0.9717 | 0.9828 |
| 70 | 30 | 194,230 | 0.9600 | 0.9345 | 0.9319 | 0.9332 | 0.9720 | 0.9709 |
| 60 | 40 | 145,675 | 0.9560 | 0.9569 | 0.9319 | 0.9443 | 0.9720 | 0.9554 |
| 50 | 50 | 116,540 | 0.9516 | 0.9702 | 0.9319 | 0.9507 | 0.9713 | 0.9345 |
| 40 | 60 | 97,115 | 0.9481 | 0.9806 | 0.9319 | 0.9556 | 0.9723 | 0.9050 |
| 30 | 70 | 83,240 | 0.9438 | 0.9870 | 0.9319 | 0.9587 | 0.9714 | 0.8595 |
| 20 | 80 | 72,835 | 0.9398 | 0.9924 | 0.9320 | 0.9612 | 0.9713 | 0.7811 |
| 10 | 90 | 64,740 | 0.9359 | 0.9966 | 0.9319 | 0.9632 | 0.9711 | 0.6132 |
| 0 | 100 | 58,270 | 0.9319 | 1.0000 | 0.9319 | 0.9648 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 1.0000 |
| 90 | 10 | 299,940 | 0.1001 | 0.1000 | 1.0000 | 0.1818 | 0.0001 | 1.0000 |
| 80 | 20 | 291,350 | 0.2001 | 0.2000 | 1.0000 | 0.3333 | 0.0001 | 0.9130 |
| 70 | 30 | 194,230 | 0.3000 | 0.3000 | 1.0000 | 0.4615 | 0.0001 | 0.8182 |
| 60 | 40 | 145,675 | 0.4001 | 0.4000 | 1.0000 | 0.5714 | 0.0001 | 0.8333 |
| 50 | 50 | 116,540 | 0.5000 | 0.5000 | 1.0000 | 0.6667 | 0.0001 | 0.6667 |
| 40 | 60 | 97,115 | 0.6000 | 0.6000 | 1.0000 | 0.7500 | 0.0001 | 0.6667 |
| 30 | 70 | 83,240 | 0.7000 | 0.7000 | 1.0000 | 0.8235 | 0.0001 | 0.5000 |
| 20 | 80 | 72,835 | 0.8000 | 0.8000 | 1.0000 | 0.8889 | 0.0001 | 0.3333 |
| 10 | 90 | 64,740 | 0.9000 | 0.9000 | 1.0000 | 0.9474 | 0.0000 | 0.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### Compressed (QAT)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9278 | 0.0000 | 0.0000 | 0.0000 | 0.9278 | 1.0000 |
| 90 | 10 | 299,940 | 0.9300 | 0.5938 | 0.9508 | 0.7310 | 0.9277 | 0.9941 |
| 80 | 20 | 291,350 | 0.9319 | 0.7654 | 0.9510 | 0.8482 | 0.9271 | 0.9870 |
| 70 | 30 | 194,230 | 0.9352 | 0.8507 | 0.9510 | 0.8980 | 0.9284 | 0.9779 |
| 60 | 40 | 145,675 | 0.9374 | 0.8984 | 0.9510 | 0.9239 | 0.9283 | 0.9660 |
| 50 | 50 | 116,540 | 0.9393 | 0.9293 | 0.9510 | 0.9400 | 0.9276 | 0.9498 |
| 40 | 60 | 97,115 | 0.9420 | 0.9522 | 0.9510 | 0.9516 | 0.9284 | 0.9266 |
| 30 | 70 | 83,240 | 0.9444 | 0.9691 | 0.9510 | 0.9599 | 0.9292 | 0.8904 |
| 20 | 80 | 72,835 | 0.9466 | 0.9817 | 0.9510 | 0.9661 | 0.9289 | 0.8258 |
| 10 | 90 | 64,740 | 0.9488 | 0.9918 | 0.9510 | 0.9710 | 0.9293 | 0.6782 |
| 0 | 100 | 58,270 | 0.9510 | 1.0000 | 0.9510 | 0.9749 | 0.0000 | 0.0000 |


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
0.15       0.1001   0.1818   0.0001   0.9600   1.0000   0.1000  
0.20       0.1002   0.1818   0.0002   0.9592   0.9999   0.1000  
0.25       0.1010   0.1819   0.0011   0.9617   0.9996   0.1001  
0.30       0.1012   0.1818   0.0015   0.9020   0.9985   0.1000  
0.35       0.1022   0.1816   0.0029   0.8657   0.9960   0.0999  
0.40       0.1055   0.1821   0.0066   0.9300   0.9955   0.1002  
0.45       0.1225   0.1847   0.0257   0.9737   0.9938   0.1018  
0.50       0.5956   0.2357   0.5925   0.9341   0.6236   0.1453  
0.55       0.9010   0.3681   0.9690   0.9246   0.2884   0.5086   <--
0.60       0.9064   0.2032   0.9939   0.9104   0.1193   0.6851  
0.65       0.9007   0.0551   0.9975   0.9024   0.0289   0.5666  
0.70       0.8983   0.0003   0.9981   0.8998   0.0002   0.0098  
0.75       0.8992   0.0001   0.9991   0.8999   0.0001   0.0085  
0.80       0.8995   0.0001   0.9994   0.8999   0.0000   0.0063  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9010, F1=0.3681, Normal Recall=0.9690, Normal Precision=0.9246, Attack Recall=0.2884, Attack Precision=0.5086

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
0.15       0.2001   0.3333   0.0001   0.9130   1.0000   0.2000  
0.20       0.2001   0.3333   0.0002   0.8913   0.9999   0.2000  
0.25       0.2007   0.3334   0.0011   0.8658   0.9993   0.2001  
0.30       0.2008   0.3331   0.0015   0.7522   0.9980   0.1999  
0.35       0.2015   0.3328   0.0029   0.7299   0.9957   0.1998  
0.40       0.2043   0.3334   0.0066   0.8439   0.9951   0.2003  
0.45       0.2190   0.3372   0.0254   0.9377   0.9932   0.2031  
0.50       0.5988   0.3834   0.5926   0.8630   0.6237   0.2768  
0.55       0.8327   0.4070   0.9691   0.8447   0.2871   0.6991   <--
0.60       0.8188   0.2077   0.9939   0.8185   0.1187   0.8286  
0.65       0.8036   0.0543   0.9975   0.8041   0.0282   0.7382  
0.70       0.7985   0.0003   0.9981   0.7997   0.0002   0.0198  
0.75       0.7993   0.0001   0.9991   0.7999   0.0001   0.0190  
0.80       0.7995   0.0001   0.9994   0.7999   0.0001   0.0213  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.8327, F1=0.4070, Normal Recall=0.9691, Normal Precision=0.8447, Attack Recall=0.2871, Attack Precision=0.6991

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
0.15       0.3001   0.4616   0.0001   0.8750   1.0000   0.3000  
0.20       0.3001   0.4616   0.0002   0.8529   0.9999   0.3000  
0.25       0.3006   0.4616   0.0011   0.7884   0.9993   0.3001  
0.30       0.3005   0.4612   0.0015   0.6395   0.9980   0.2999  
0.35       0.3007   0.4607   0.0029   0.6121   0.9957   0.2997  
0.40       0.3032   0.4615   0.0066   0.7609   0.9951   0.3004  
0.45       0.3159   0.4656   0.0256   0.8984   0.9932   0.3040  
0.50       0.6015   0.4843   0.5921   0.7859   0.6237   0.3959   <--
0.55       0.7641   0.4221   0.9686   0.7602   0.2871   0.7966  
0.60       0.7313   0.2096   0.9939   0.7246   0.1187   0.8929  
0.65       0.7067   0.0545   0.9976   0.7055   0.0282   0.8317  
0.70       0.6988   0.0003   0.9982   0.6996   0.0002   0.0347  
0.75       0.6995   0.0001   0.9992   0.6998   0.0001   0.0357  
0.80       0.6996   0.0001   0.9995   0.6999   0.0001   0.0405  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.6015, F1=0.4843, Normal Recall=0.5921, Normal Precision=0.7859, Attack Recall=0.6237, Attack Precision=0.3959

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
0.15       0.4001   0.5714   0.0001   0.8333   1.0000   0.4000  
0.20       0.4001   0.5714   0.0002   0.7917   0.9999   0.4000  
0.25       0.4004   0.5714   0.0011   0.7015   0.9993   0.4001  
0.30       0.4001   0.5710   0.0014   0.5208   0.9980   0.3999  
0.35       0.4000   0.5704   0.0028   0.5000   0.9957   0.3997  
0.40       0.4021   0.5711   0.0067   0.6736   0.9951   0.4004  
0.45       0.4127   0.5750   0.0256   0.8502   0.9932   0.4046   <--
0.50       0.6050   0.5582   0.5926   0.7026   0.6237   0.5051  
0.55       0.6963   0.4306   0.9691   0.6709   0.2871   0.8611  
0.60       0.6440   0.2106   0.9941   0.6285   0.1187   0.9308  
0.65       0.6099   0.0546   0.9977   0.6063   0.0282   0.8909  
0.70       0.5991   0.0003   0.9983   0.5996   0.0002   0.0584  
0.75       0.5996   0.0001   0.9993   0.5998   0.0001   0.0615  
0.80       0.5997   0.0001   0.9995   0.5999   0.0001   0.0652  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.4127, F1=0.5750, Normal Recall=0.0256, Normal Precision=0.8502, Attack Recall=0.9932, Attack Precision=0.4046

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
0.15       0.5000   0.6667   0.0001   0.6667   1.0000   0.5000  
0.20       0.5001   0.6667   0.0002   0.6875   0.9999   0.5000  
0.25       0.5001   0.6666   0.0010   0.5833   0.9993   0.5001  
0.30       0.4997   0.6661   0.0013   0.3947   0.9980   0.4998  
0.35       0.4993   0.6654   0.0030   0.4100   0.9957   0.4997  
0.40       0.5009   0.6660   0.0067   0.5805   0.9951   0.5005  
0.45       0.5093   0.6693   0.0254   0.7900   0.9932   0.5047   <--
0.50       0.6072   0.6136   0.5907   0.6109   0.6237   0.6038  
0.55       0.6279   0.4355   0.9687   0.5761   0.2871   0.9018  
0.60       0.5566   0.2112   0.9944   0.5302   0.1187   0.9549  
0.65       0.5129   0.0547   0.9976   0.5065   0.0282   0.9224  
0.70       0.4993   0.0003   0.9984   0.4996   0.0002   0.0865  
0.75       0.4997   0.0001   0.9993   0.4998   0.0001   0.0870  
0.80       0.4998   0.0001   0.9995   0.4999   0.0001   0.0857  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.5093, F1=0.6693, Normal Recall=0.0254, Normal Precision=0.7900, Attack Recall=0.9932, Attack Precision=0.5047

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
0.15       0.9460   0.7825   0.9432   0.9966   0.9713   0.6552  
0.20       0.9586   0.8220   0.9589   0.9949   0.9557   0.7212  
0.25       0.9647   0.8411   0.9679   0.9926   0.9351   0.7642  
0.30       0.9679   0.8532   0.9717   0.9925   0.9336   0.7856  
0.35       0.9691   0.8575   0.9733   0.9922   0.9308   0.7949  
0.40       0.9698   0.8602   0.9743   0.9920   0.9293   0.8006  
0.45       0.9700   0.8608   0.9746   0.9919   0.9286   0.8023  
0.50       0.9699   0.8600   0.9749   0.9915   0.9251   0.8034  
0.55       0.9696   0.8581   0.9753   0.9908   0.9187   0.8049  
0.60       0.9680   0.8478   0.9763   0.9879   0.8926   0.8073  
0.65       0.9749   0.8643   0.9944   0.9781   0.7995   0.9404  
0.70       0.9753   0.8646   0.9958   0.9771   0.7901   0.9548   <--
0.75       0.9753   0.8631   0.9971   0.9760   0.7791   0.9674  
0.80       0.9751   0.8616   0.9974   0.9755   0.7748   0.9702  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.7
  At threshold 0.7: Accuracy=0.9753, F1=0.8646, Normal Recall=0.9958, Normal Precision=0.9771, Attack Recall=0.7901, Attack Precision=0.9548

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
0.15       0.9485   0.8828   0.9433   0.9919   0.9694   0.8104  
0.20       0.9580   0.9009   0.9590   0.9882   0.9541   0.8534  
0.25       0.9611   0.9058   0.9680   0.9831   0.9336   0.8795  
0.30       0.9638   0.9115   0.9718   0.9828   0.9319   0.8919  
0.35       0.9646   0.9130   0.9734   0.9822   0.9294   0.8971  
0.40       0.9650   0.9139   0.9743   0.9818   0.9279   0.9003  
0.45       0.9651   0.9139   0.9746   0.9816   0.9269   0.9013   <--
0.50       0.9646   0.9126   0.9749   0.9808   0.9236   0.9019  
0.55       0.9636   0.9098   0.9753   0.9791   0.9169   0.9027  
0.60       0.9593   0.8974   0.9764   0.9728   0.8908   0.9042  
0.65       0.9551   0.8768   0.9944   0.9517   0.7981   0.9726  
0.70       0.9545   0.8739   0.9959   0.9497   0.7889   0.9794  
0.75       0.9533   0.8695   0.9971   0.9473   0.7781   0.9852  
0.80       0.9527   0.8675   0.9974   0.9464   0.7740   0.9867  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9651, F1=0.9139, Normal Recall=0.9746, Normal Precision=0.9816, Attack Recall=0.9269, Attack Precision=0.9013

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
0.15       0.9513   0.9227   0.9435   0.9863   0.9694   0.8803  
0.20       0.9575   0.9309   0.9590   0.9799   0.9541   0.9089  
0.25       0.9576   0.9297   0.9679   0.9714   0.9336   0.9258  
0.30       0.9596   0.9327   0.9715   0.9708   0.9319   0.9334  
0.35       0.9600   0.9331   0.9731   0.9698   0.9294   0.9367  
0.40       0.9602   0.9333   0.9741   0.9692   0.9279   0.9388   <--
0.45       0.9601   0.9331   0.9743   0.9689   0.9269   0.9393  
0.50       0.9593   0.9316   0.9746   0.9675   0.9236   0.9398  
0.55       0.9576   0.9284   0.9750   0.9648   0.9169   0.9402  
0.60       0.9505   0.9153   0.9761   0.9542   0.8908   0.9411  
0.65       0.9355   0.8813   0.9944   0.9199   0.7981   0.9839  
0.70       0.9338   0.8772   0.9958   0.9167   0.7889   0.9878  
0.75       0.9313   0.8718   0.9970   0.9129   0.7781   0.9910  
0.80       0.9303   0.8695   0.9973   0.9115   0.7740   0.9920  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9602, F1=0.9333, Normal Recall=0.9741, Normal Precision=0.9692, Attack Recall=0.9279, Attack Precision=0.9388

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
0.15       0.9538   0.9438   0.9435   0.9788   0.9694   0.9196  
0.20       0.9571   0.9468   0.9592   0.9691   0.9541   0.9397   <--
0.25       0.9543   0.9423   0.9681   0.9563   0.9336   0.9512  
0.30       0.9557   0.9439   0.9715   0.9554   0.9319   0.9562  
0.35       0.9556   0.9437   0.9731   0.9539   0.9294   0.9584  
0.40       0.9556   0.9435   0.9740   0.9530   0.9279   0.9597  
0.45       0.9554   0.9432   0.9743   0.9524   0.9269   0.9601  
0.50       0.9542   0.9416   0.9746   0.9503   0.9236   0.9603  
0.55       0.9517   0.9382   0.9749   0.9462   0.9169   0.9606  
0.60       0.9420   0.9248   0.9762   0.9306   0.8908   0.9614  
0.65       0.9158   0.8835   0.9943   0.8808   0.7981   0.9893  
0.70       0.9130   0.8788   0.9957   0.8762   0.7889   0.9919  
0.75       0.9093   0.8728   0.9967   0.8708   0.7781   0.9938  
0.80       0.9079   0.8705   0.9971   0.8687   0.7740   0.9945  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9571, F1=0.9468, Normal Recall=0.9592, Normal Precision=0.9691, Attack Recall=0.9541, Attack Precision=0.9397

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
0.15       0.9565   0.9571   0.9437   0.9686   0.9694   0.9451   <--
0.20       0.9567   0.9566   0.9594   0.9543   0.9541   0.9592  
0.25       0.9511   0.9502   0.9685   0.9359   0.9336   0.9674  
0.30       0.9519   0.9509   0.9719   0.9346   0.9319   0.9707  
0.35       0.9514   0.9503   0.9735   0.9324   0.9294   0.9722  
0.40       0.9511   0.9499   0.9743   0.9311   0.9279   0.9731  
0.45       0.9508   0.9496   0.9747   0.9303   0.9269   0.9734  
0.50       0.9492   0.9479   0.9749   0.9273   0.9236   0.9736  
0.55       0.9461   0.9445   0.9753   0.9215   0.9169   0.9737  
0.60       0.9336   0.9306   0.9765   0.8994   0.8908   0.9742  
0.65       0.8961   0.8848   0.9942   0.8312   0.7981   0.9928  
0.70       0.8923   0.8799   0.9957   0.8251   0.7889   0.9945  
0.75       0.8874   0.8736   0.9967   0.8179   0.7781   0.9958  
0.80       0.8856   0.8712   0.9971   0.8152   0.7740   0.9963  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9565, F1=0.9571, Normal Recall=0.9437, Normal Precision=0.9686, Attack Recall=0.9694, Attack Precision=0.9451

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
0.15       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.20       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.25       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.30       0.1001   0.1818   0.0001   0.9615   1.0000   0.1000  
0.35       0.1001   0.1818   0.0001   0.9750   1.0000   0.1000  
0.40       0.1018   0.1821   0.0020   0.9982   1.0000   0.1002  
0.45       0.1124   0.1836   0.0140   0.9834   0.9979   0.1011   <--
0.50       0.3717   0.1370   0.3576   0.8652   0.4987   0.0794  
0.55       0.8964   0.0118   0.9953   0.9001   0.0062   0.1278  
0.60       0.8986   0.0028   0.9983   0.9000   0.0014   0.0865  
0.65       0.8994   0.0025   0.9992   0.9000   0.0013   0.1508  
0.70       0.8999   0.0000   0.9998   0.9000   0.0000   0.0000  
0.75       0.9000   0.0000   0.9999   0.9000   0.0000   0.0000  
0.80       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.1124, F1=0.1836, Normal Recall=0.0140, Normal Precision=0.9834, Attack Recall=0.9979, Attack Precision=0.1011

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
0.15       0.2000   0.3333   0.0001   0.9333   1.0000   0.2000  
0.20       0.2000   0.3333   0.0001   0.9333   1.0000   0.2000  
0.25       0.2001   0.3333   0.0001   0.9474   1.0000   0.2000  
0.30       0.2001   0.3333   0.0001   0.9130   1.0000   0.2000  
0.35       0.2001   0.3333   0.0001   0.9143   0.9999   0.2000  
0.40       0.2016   0.3338   0.0020   0.9914   0.9999   0.2003  
0.45       0.2108   0.3359   0.0140   0.9656   0.9980   0.2019   <--
0.50       0.3855   0.2449   0.3574   0.7402   0.4982   0.1624  
0.55       0.7975   0.0120   0.9953   0.8002   0.0061   0.2476  
0.60       0.7989   0.0028   0.9983   0.8000   0.0014   0.1718  
0.65       0.7996   0.0025   0.9992   0.8001   0.0012   0.2687  
0.70       0.7999   0.0000   0.9998   0.8000   0.0000   0.0250  
0.75       0.8000   0.0000   0.9999   0.8000   0.0000   0.0000  
0.80       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.2108, F1=0.3359, Normal Recall=0.0140, Normal Precision=0.9656, Attack Recall=0.9980, Attack Precision=0.2019

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
0.15       0.3000   0.4615   0.0001   0.8750   1.0000   0.3000  
0.20       0.3000   0.4615   0.0001   0.8750   1.0000   0.3000  
0.25       0.3000   0.4616   0.0001   0.9091   1.0000   0.3000  
0.30       0.3000   0.4615   0.0001   0.8462   1.0000   0.3000  
0.35       0.3001   0.4616   0.0001   0.8500   0.9999   0.3000  
0.40       0.3013   0.4620   0.0019   0.9851   0.9999   0.3004  
0.45       0.3091   0.4643   0.0139   0.9421   0.9980   0.3025   <--
0.50       0.3993   0.3323   0.3568   0.6240   0.4982   0.2493  
0.55       0.6988   0.0121   0.9957   0.7004   0.0061   0.3792  
0.60       0.6993   0.0028   0.9984   0.7000   0.0014   0.2721  
0.65       0.6998   0.0025   0.9992   0.7001   0.0012   0.4000  
0.70       0.6999   0.0000   0.9999   0.7000   0.0000   0.0526  
0.75       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.80       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.3091, F1=0.4643, Normal Recall=0.0139, Normal Precision=0.9421, Attack Recall=0.9980, Attack Precision=0.3025

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
0.15       0.4000   0.5714   0.0000   0.8000   1.0000   0.4000  
0.20       0.4000   0.5714   0.0000   0.8000   1.0000   0.4000  
0.25       0.4000   0.5714   0.0001   0.8750   1.0000   0.4000  
0.30       0.4000   0.5714   0.0001   0.8000   1.0000   0.4000  
0.35       0.4001   0.5714   0.0001   0.8000   0.9999   0.4000  
0.40       0.4011   0.5719   0.0019   0.9766   0.9999   0.4004  
0.45       0.4076   0.5741   0.0141   0.9138   0.9980   0.4029   <--
0.50       0.4141   0.4049   0.3581   0.5170   0.4982   0.3410  
0.55       0.6000   0.0121   0.9958   0.6005   0.0061   0.4965  
0.60       0.5996   0.0028   0.9984   0.6000   0.0014   0.3773  
0.65       0.6000   0.0025   0.9991   0.6001   0.0012   0.4898  
0.70       0.5999   0.0000   0.9999   0.6000   0.0000   0.0769  
0.75       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.80       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.4076, F1=0.5741, Normal Recall=0.0141, Normal Precision=0.9138, Attack Recall=0.9980, Attack Precision=0.4029

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
0.15       0.5000   0.6667   0.0000   0.6667   1.0000   0.5000  
0.20       0.5000   0.6667   0.0000   0.6667   1.0000   0.5000  
0.25       0.5000   0.6667   0.0001   0.7500   1.0000   0.5000  
0.30       0.5000   0.6667   0.0001   0.6000   1.0000   0.5000  
0.35       0.5000   0.6667   0.0001   0.6667   0.9999   0.5000  
0.40       0.5010   0.6671   0.0020   0.9667   0.9999   0.5005  
0.45       0.5060   0.6689   0.0140   0.8751   0.9980   0.5030   <--
0.50       0.4285   0.4657   0.3587   0.4169   0.4982   0.4372  
0.55       0.5010   0.0122   0.9958   0.5005   0.0061   0.5937  
0.60       0.4999   0.0028   0.9984   0.5000   0.0014   0.4770  
0.65       0.5002   0.0025   0.9992   0.5001   0.0012   0.5950  
0.70       0.5000   0.0000   1.0000   0.5000   0.0000   0.3333  
0.75       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.80       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.5060, F1=0.6689, Normal Recall=0.0140, Normal Precision=0.8751, Attack Recall=0.9980, Attack Precision=0.5030

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
0.15       0.9097   0.6807   0.9038   0.9954   0.9627   0.5265  
0.20       0.9167   0.6978   0.9118   0.9953   0.9612   0.5477  
0.25       0.9221   0.7106   0.9183   0.9947   0.9561   0.5654  
0.30       0.9302   0.7319   0.9277   0.9943   0.9525   0.5942  
0.35       0.9440   0.7720   0.9435   0.9940   0.9485   0.6509  
0.40       0.9596   0.8206   0.9635   0.9913   0.9242   0.7378  
0.45       0.9640   0.8362   0.9689   0.9909   0.9198   0.7665  
0.50       0.9662   0.8444   0.9718   0.9905   0.9164   0.7829  
0.55       0.9666   0.8448   0.9729   0.9898   0.9097   0.7886  
0.60       0.9673   0.8466   0.9744   0.9891   0.9035   0.7965  
0.65       0.9678   0.8486   0.9751   0.9890   0.9024   0.8009  
0.70       0.9681   0.8490   0.9759   0.9884   0.8973   0.8056   <--
0.75       0.9671   0.8417   0.9773   0.9860   0.8750   0.8108  
0.80       0.9594   0.7479   0.9990   0.9577   0.6026   0.9854  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.7
  At threshold 0.7: Accuracy=0.9681, F1=0.8490, Normal Recall=0.9759, Normal Precision=0.9884, Attack Recall=0.8973, Attack Precision=0.8056

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
0.15       0.9156   0.8199   0.9041   0.9894   0.9613   0.7148  
0.20       0.9216   0.8304   0.9121   0.9891   0.9597   0.7318  
0.25       0.9259   0.8374   0.9187   0.9878   0.9546   0.7458  
0.30       0.9326   0.8494   0.9279   0.9870   0.9510   0.7674  
0.35       0.9443   0.8719   0.9436   0.9862   0.9472   0.8076  
0.40       0.9556   0.8926   0.9637   0.9804   0.9231   0.8640  
0.45       0.9589   0.8995   0.9690   0.9795   0.9188   0.8810  
0.50       0.9606   0.9028   0.9718   0.9788   0.9157   0.8903   <--
0.55       0.9602   0.9015   0.9729   0.9773   0.9095   0.8936  
0.60       0.9602   0.9007   0.9744   0.9758   0.9033   0.8981  
0.65       0.9605   0.9013   0.9751   0.9755   0.9019   0.9006  
0.70       0.9601   0.8999   0.9760   0.9742   0.8967   0.9032  
0.75       0.9568   0.8900   0.9773   0.9689   0.8745   0.9061  
0.80       0.9191   0.7478   0.9990   0.9089   0.5996   0.9935  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.9606, F1=0.9028, Normal Recall=0.9718, Normal Precision=0.9788, Attack Recall=0.9157, Attack Precision=0.8903

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
0.15       0.9217   0.8805   0.9047   0.9820   0.9613   0.8122  
0.20       0.9267   0.8871   0.9126   0.9814   0.9597   0.8248  
0.25       0.9297   0.8907   0.9191   0.9793   0.9546   0.8348  
0.30       0.9350   0.8977   0.9281   0.9779   0.9510   0.8501  
0.35       0.9445   0.9111   0.9434   0.9766   0.9472   0.8777  
0.40       0.9512   0.9190   0.9632   0.9669   0.9231   0.9149  
0.45       0.9537   0.9225   0.9686   0.9653   0.9188   0.9262  
0.50       0.9547   0.9238   0.9714   0.9641   0.9157   0.9321   <--
0.55       0.9537   0.9217   0.9726   0.9616   0.9095   0.9343  
0.60       0.9528   0.9198   0.9739   0.9592   0.9033   0.9369  
0.65       0.9528   0.9198   0.9747   0.9587   0.9019   0.9385  
0.70       0.9519   0.9179   0.9755   0.9566   0.8967   0.9401  
0.75       0.9462   0.9070   0.9770   0.9478   0.8745   0.9421  
0.80       0.8792   0.7486   0.9991   0.8534   0.5996   0.9963  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.9547, F1=0.9238, Normal Recall=0.9714, Normal Precision=0.9641, Attack Recall=0.9157, Attack Precision=0.9321

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
0.15       0.9274   0.9137   0.9048   0.9723   0.9613   0.8706  
0.20       0.9316   0.9181   0.9128   0.9714   0.9597   0.8800  
0.25       0.9333   0.9196   0.9191   0.9681   0.9546   0.8872  
0.30       0.9373   0.9239   0.9282   0.9660   0.9510   0.8983  
0.35       0.9448   0.9321   0.9433   0.9640   0.9472   0.9176  
0.40       0.9472   0.9333   0.9632   0.9495   0.9231   0.9436  
0.45       0.9486   0.9346   0.9685   0.9471   0.9188   0.9510  
0.50       0.9491   0.9351   0.9714   0.9453   0.9157   0.9553   <--
0.55       0.9473   0.9325   0.9726   0.9416   0.9095   0.9567  
0.60       0.9457   0.9301   0.9739   0.9379   0.9033   0.9585  
0.65       0.9455   0.9298   0.9746   0.9371   0.9019   0.9595  
0.70       0.9439   0.9275   0.9755   0.9340   0.8967   0.9606  
0.75       0.9359   0.9161   0.9769   0.9211   0.8745   0.9619  
0.80       0.8392   0.7489   0.9989   0.7891   0.5996   0.9973  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.9491, F1=0.9351, Normal Recall=0.9714, Normal Precision=0.9453, Attack Recall=0.9157, Attack Precision=0.9553

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
0.15       0.9329   0.9347   0.9044   0.9590   0.9613   0.9096  
0.20       0.9358   0.9373   0.9120   0.9577   0.9597   0.9160  
0.25       0.9366   0.9377   0.9186   0.9529   0.9546   0.9214  
0.30       0.9394   0.9400   0.9277   0.9498   0.9510   0.9293  
0.35       0.9452   0.9453   0.9433   0.9470   0.9472   0.9435   <--
0.40       0.9434   0.9422   0.9637   0.9261   0.9231   0.9622  
0.45       0.9438   0.9423   0.9688   0.9227   0.9188   0.9671  
0.50       0.9437   0.9421   0.9718   0.9202   0.9157   0.9701  
0.55       0.9412   0.9393   0.9730   0.9149   0.9095   0.9712  
0.60       0.9388   0.9365   0.9742   0.9097   0.9033   0.9723  
0.65       0.9384   0.9361   0.9749   0.9086   0.9019   0.9730  
0.70       0.9362   0.9336   0.9758   0.9042   0.8967   0.9737  
0.75       0.9259   0.9218   0.9772   0.8862   0.8745   0.9746  
0.80       0.7992   0.7492   0.9989   0.7138   0.5996   0.9982  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9452, F1=0.9453, Normal Recall=0.9433, Normal Precision=0.9470, Attack Recall=0.9472, Attack Precision=0.9435

```

