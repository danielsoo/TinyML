# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-24 09:13:39 |

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
| Original (TFLite) | 0.0030 | 0.1028 | 0.2025 | 0.3020 | 0.4018 | 0.5016 | 0.6013 | 0.7008 | 0.8006 | 0.9003 | 0.9999 |
| noQAT+PTQ | 0.0521 | 0.1469 | 0.2418 | 0.3362 | 0.4319 | 0.5260 | 0.6211 | 0.7160 | 0.8101 | 0.9056 | 1.0000 |
| saved_model_traditional_qat | 0.9669 | 0.9646 | 0.9622 | 0.9600 | 0.9572 | 0.9546 | 0.9526 | 0.9496 | 0.9472 | 0.9446 | 0.9423 |
| QAT+PTQ | 0.0001 | 0.1001 | 0.2001 | 0.3001 | 0.4001 | 0.5001 | 0.6001 | 0.7001 | 0.8001 | 0.9000 | 1.0000 |
| Compressed (QAT) | 0.9705 | 0.9653 | 0.9592 | 0.9535 | 0.9472 | 0.9411 | 0.9353 | 0.9290 | 0.9232 | 0.9171 | 0.9111 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1823 | 0.3340 | 0.4622 | 0.5721 | 0.6674 | 0.7506 | 0.8239 | 0.8892 | 0.9475 | 1.0000 |
| noQAT+PTQ | 0.0000 | 0.1899 | 0.3454 | 0.4748 | 0.5847 | 0.6784 | 0.7600 | 0.8314 | 0.8939 | 0.9502 | 1.0000 |
| saved_model_traditional_qat | 0.0000 | 0.8421 | 0.9088 | 0.9339 | 0.9463 | 0.9541 | 0.9598 | 0.9632 | 0.9661 | 0.9684 | 0.9703 |
| QAT+PTQ | 0.0000 | 0.1818 | 0.3334 | 0.4616 | 0.5715 | 0.6667 | 0.7500 | 0.8236 | 0.8889 | 0.9474 | 1.0000 |
| Compressed (QAT) | 0.0000 | 0.8402 | 0.8994 | 0.9216 | 0.9324 | 0.9393 | 0.9441 | 0.9473 | 0.9499 | 0.9519 | 0.9535 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0030 | 0.0031 | 0.0031 | 0.0029 | 0.0030 | 0.0032 | 0.0033 | 0.0028 | 0.0032 | 0.0037 | 0.0000 |
| noQAT+PTQ | 0.0521 | 0.0521 | 0.0522 | 0.0518 | 0.0531 | 0.0519 | 0.0527 | 0.0535 | 0.0505 | 0.0562 | 0.0000 |
| saved_model_traditional_qat | 0.9669 | 0.9671 | 0.9671 | 0.9675 | 0.9672 | 0.9669 | 0.9681 | 0.9666 | 0.9665 | 0.9652 | 0.0000 |
| QAT+PTQ | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0003 | 0.0002 | 0.0000 |
| Compressed (QAT) | 0.9705 | 0.9712 | 0.9713 | 0.9717 | 0.9713 | 0.9712 | 0.9716 | 0.9708 | 0.9714 | 0.9713 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0030 | 0.0000 | 0.0000 | 0.0000 | 0.0030 | 1.0000 |
| 90 | 10 | 299,940 | 0.1028 | 0.1003 | 0.9999 | 0.1823 | 0.0031 | 0.9976 |
| 80 | 20 | 291,350 | 0.2025 | 0.2005 | 0.9999 | 0.3340 | 0.0031 | 0.9932 |
| 70 | 30 | 194,230 | 0.3020 | 0.3006 | 0.9999 | 0.4622 | 0.0029 | 0.9876 |
| 60 | 40 | 145,675 | 0.4018 | 0.4007 | 0.9999 | 0.5721 | 0.0030 | 0.9814 |
| 50 | 50 | 116,540 | 0.5016 | 0.5008 | 0.9999 | 0.6674 | 0.0032 | 0.9742 |
| 40 | 60 | 97,115 | 0.6013 | 0.6008 | 0.9999 | 0.7506 | 0.0033 | 0.9630 |
| 30 | 70 | 83,240 | 0.7008 | 0.7006 | 0.9999 | 0.8239 | 0.0028 | 0.9342 |
| 20 | 80 | 72,835 | 0.8006 | 0.8005 | 0.9999 | 0.8892 | 0.0032 | 0.9038 |
| 10 | 90 | 64,740 | 0.9003 | 0.9003 | 0.9999 | 0.9475 | 0.0037 | 0.8276 |
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
| 100 | 0 | 100,000 | 0.9669 | 0.0000 | 0.0000 | 0.0000 | 0.9669 | 1.0000 |
| 90 | 10 | 299,940 | 0.9646 | 0.7609 | 0.9425 | 0.8421 | 0.9671 | 0.9934 |
| 80 | 20 | 291,350 | 0.9622 | 0.8776 | 0.9423 | 0.9088 | 0.9671 | 0.9853 |
| 70 | 30 | 194,230 | 0.9600 | 0.9256 | 0.9423 | 0.9339 | 0.9675 | 0.9751 |
| 60 | 40 | 145,675 | 0.9572 | 0.9504 | 0.9423 | 0.9463 | 0.9672 | 0.9618 |
| 50 | 50 | 116,540 | 0.9546 | 0.9661 | 0.9423 | 0.9541 | 0.9669 | 0.9437 |
| 40 | 60 | 97,115 | 0.9526 | 0.9779 | 0.9423 | 0.9598 | 0.9681 | 0.9180 |
| 30 | 70 | 83,240 | 0.9496 | 0.9851 | 0.9423 | 0.9632 | 0.9666 | 0.8778 |
| 20 | 80 | 72,835 | 0.9472 | 0.9912 | 0.9423 | 0.9661 | 0.9665 | 0.8073 |
| 10 | 90 | 64,740 | 0.9446 | 0.9959 | 0.9423 | 0.9684 | 0.9652 | 0.6503 |
| 0 | 100 | 58,270 | 0.9423 | 1.0000 | 0.9423 | 0.9703 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 1.0000 |
| 90 | 10 | 299,940 | 0.1001 | 0.1000 | 1.0000 | 0.1818 | 0.0001 | 1.0000 |
| 80 | 20 | 291,350 | 0.2001 | 0.2000 | 1.0000 | 0.3334 | 0.0001 | 1.0000 |
| 70 | 30 | 194,230 | 0.3001 | 0.3000 | 1.0000 | 0.4616 | 0.0002 | 1.0000 |
| 60 | 40 | 145,675 | 0.4001 | 0.4000 | 1.0000 | 0.5715 | 0.0002 | 1.0000 |
| 50 | 50 | 116,540 | 0.5001 | 0.5000 | 1.0000 | 0.6667 | 0.0002 | 1.0000 |
| 40 | 60 | 97,115 | 0.6001 | 0.6000 | 1.0000 | 0.7500 | 0.0002 | 1.0000 |
| 30 | 70 | 83,240 | 0.7001 | 0.7001 | 1.0000 | 0.8236 | 0.0002 | 1.0000 |
| 20 | 80 | 72,835 | 0.8001 | 0.8000 | 1.0000 | 0.8889 | 0.0003 | 1.0000 |
| 10 | 90 | 64,740 | 0.9000 | 0.9000 | 1.0000 | 0.9474 | 0.0002 | 1.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### Compressed (QAT)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9705 | 0.0000 | 0.0000 | 0.0000 | 0.9705 | 1.0000 |
| 90 | 10 | 299,940 | 0.9653 | 0.7787 | 0.9123 | 0.8402 | 0.9712 | 0.9901 |
| 80 | 20 | 291,350 | 0.9592 | 0.8879 | 0.9111 | 0.8994 | 0.9713 | 0.9776 |
| 70 | 30 | 194,230 | 0.9535 | 0.9324 | 0.9111 | 0.9216 | 0.9717 | 0.9623 |
| 60 | 40 | 145,675 | 0.9472 | 0.9548 | 0.9111 | 0.9324 | 0.9713 | 0.9425 |
| 50 | 50 | 116,540 | 0.9411 | 0.9693 | 0.9111 | 0.9393 | 0.9712 | 0.9161 |
| 40 | 60 | 97,115 | 0.9353 | 0.9796 | 0.9111 | 0.9441 | 0.9716 | 0.8793 |
| 30 | 70 | 83,240 | 0.9290 | 0.9865 | 0.9111 | 0.9473 | 0.9708 | 0.8239 |
| 20 | 80 | 72,835 | 0.9232 | 0.9922 | 0.9111 | 0.9499 | 0.9714 | 0.7320 |
| 10 | 90 | 64,740 | 0.9171 | 0.9965 | 0.9111 | 0.9519 | 0.9713 | 0.5483 |
| 0 | 100 | 58,270 | 0.9111 | 1.0000 | 0.9111 | 0.9535 | 0.0000 | 0.0000 |


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
0.15       0.1000   0.1818   0.0000   0.0000   1.0000   0.1000  
0.20       0.1000   0.1818   0.0000   1.0000   1.0000   0.1000  
0.25       0.1008   0.1819   0.0009   0.9958   1.0000   0.1001  
0.30       0.1028   0.1823   0.0031   0.9988   1.0000   0.1003  
0.35       0.1076   0.1831   0.0085   0.9996   1.0000   0.1008  
0.40       0.1116   0.1837   0.0129   0.9957   0.9995   0.1011  
0.45       0.1157   0.1844   0.0175   0.9962   0.9994   0.1015  
0.50       0.4899   0.2643   0.4425   0.9794   0.9164   0.1544  
0.55       0.8273   0.3616   0.8649   0.9384   0.4890   0.2868  
0.60       0.9000   0.4203   0.9597   0.9313   0.3627   0.4998   <--
0.65       0.9101   0.2854   0.9913   0.9158   0.1795   0.6964  
0.70       0.9034   0.1206   0.9964   0.9057   0.0662   0.6708  
0.75       0.9004   0.0553   0.9972   0.9024   0.0292   0.5385  
0.80       0.8983   0.0048   0.9978   0.9000   0.0024   0.1090  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.6
  At threshold 0.6: Accuracy=0.9000, F1=0.4203, Normal Recall=0.9597, Normal Precision=0.9313, Attack Recall=0.3627, Attack Precision=0.4998

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
0.15       0.2000   0.3333   0.0000   0.0000   1.0000   0.2000  
0.20       0.2000   0.3333   0.0000   1.0000   1.0000   0.2000  
0.25       0.2007   0.3335   0.0009   0.9853   0.9999   0.2001  
0.30       0.2025   0.3340   0.0031   0.9932   0.9999   0.2005  
0.35       0.2067   0.3352   0.0084   0.9975   0.9999   0.2013  
0.40       0.2102   0.3361   0.0129   0.9882   0.9994   0.2020  
0.45       0.2138   0.3370   0.0174   0.9898   0.9993   0.2027  
0.50       0.5374   0.4418   0.4429   0.9544   0.9153   0.2911  
0.55       0.7896   0.4816   0.8648   0.8712   0.4887   0.4748   <--
0.60       0.8406   0.4777   0.9597   0.8580   0.3645   0.6931  
0.65       0.8296   0.3002   0.9912   0.8291   0.1828   0.8390  
0.70       0.8103   0.1220   0.9964   0.8101   0.0659   0.8188  
0.75       0.8037   0.0566   0.9972   0.8043   0.0295   0.7248  
0.80       0.7988   0.0055   0.9978   0.8001   0.0028   0.2350  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.7896, F1=0.4816, Normal Recall=0.8648, Normal Precision=0.8712, Attack Recall=0.4887, Attack Precision=0.4748

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
0.15       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.20       0.3000   0.4615   0.0000   1.0000   1.0000   0.3000  
0.25       0.3005   0.4617   0.0008   0.9730   0.9999   0.3002  
0.30       0.3022   0.4623   0.0031   0.9885   0.9999   0.3006  
0.35       0.3059   0.4636   0.0085   0.9957   0.9999   0.3018  
0.40       0.3089   0.4646   0.0129   0.9799   0.9994   0.3026  
0.45       0.3119   0.4656   0.0174   0.9825   0.9993   0.3035  
0.50       0.5841   0.5690   0.4421   0.9241   0.9153   0.4129   <--
0.55       0.7511   0.5409   0.8635   0.7976   0.4887   0.6055  
0.60       0.7808   0.4994   0.9593   0.7789   0.3645   0.7932  
0.65       0.7486   0.3038   0.9911   0.7389   0.1828   0.8981  
0.70       0.7173   0.1228   0.9964   0.7134   0.0659   0.8875  
0.75       0.7070   0.0569   0.9973   0.7057   0.0295   0.8255  
0.80       0.6993   0.0055   0.9979   0.7001   0.0028   0.3562  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.5841, F1=0.5690, Normal Recall=0.4421, Normal Precision=0.9241, Attack Recall=0.9153, Attack Precision=0.4129

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
0.15       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.20       0.4000   0.5714   0.0000   1.0000   1.0000   0.4000  
0.25       0.4005   0.5716   0.0008   0.9605   0.9999   0.4002  
0.30       0.4018   0.5721   0.0030   0.9813   0.9999   0.4007  
0.35       0.4049   0.5734   0.0081   0.9930   0.9999   0.4019  
0.40       0.4073   0.5743   0.0126   0.9684   0.9994   0.4029  
0.45       0.4100   0.5754   0.0171   0.9727   0.9993   0.4040  
0.50       0.6314   0.6652   0.4422   0.8867   0.9153   0.5224   <--
0.55       0.7137   0.5773   0.8637   0.7170   0.4887   0.7051  
0.60       0.7213   0.5113   0.9592   0.6936   0.3645   0.8561  
0.65       0.6678   0.3057   0.9911   0.6453   0.1828   0.9321  
0.70       0.6243   0.1231   0.9965   0.6154   0.0659   0.9262  
0.75       0.6102   0.0570   0.9974   0.6065   0.0295   0.8841  
0.80       0.5999   0.0055   0.9980   0.6002   0.0028   0.4763  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.6314, F1=0.6652, Normal Recall=0.4422, Normal Precision=0.8867, Attack Recall=0.9153, Attack Precision=0.5224

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
0.15       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.20       0.5000   0.6667   0.0000   1.0000   1.0000   0.5000  
0.25       0.5004   0.6668   0.0009   0.9455   0.9999   0.5002  
0.30       0.5015   0.6673   0.0030   0.9724   0.9999   0.5007  
0.35       0.5042   0.6685   0.0086   0.9901   0.9999   0.5021  
0.40       0.5060   0.6692   0.0127   0.9535   0.9994   0.5030  
0.45       0.5082   0.6702   0.0171   0.9596   0.9993   0.5041  
0.50       0.6785   0.7401   0.4418   0.8391   0.9153   0.6212   <--
0.55       0.6768   0.6019   0.8648   0.6285   0.4887   0.7834  
0.60       0.6616   0.5185   0.9587   0.6013   0.3645   0.8982  
0.65       0.5870   0.3069   0.9912   0.5481   0.1828   0.9542  
0.70       0.5312   0.1233   0.9965   0.5162   0.0659   0.9493  
0.75       0.5135   0.0571   0.9975   0.5069   0.0295   0.9226  
0.80       0.5004   0.0055   0.9980   0.5002   0.0028   0.5833  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.6785, F1=0.7401, Normal Recall=0.4418, Normal Precision=0.8391, Attack Recall=0.9153, Attack Precision=0.6212

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
0.15       0.9442   0.7769   0.9411   0.9967   0.9722   0.6470  
0.20       0.9606   0.8286   0.9615   0.9945   0.9523   0.7334  
0.25       0.9633   0.8376   0.9652   0.9938   0.9462   0.7514  
0.30       0.9648   0.8427   0.9671   0.9936   0.9437   0.7612  
0.35       0.9683   0.8553   0.9719   0.9927   0.9361   0.7873  
0.40       0.9699   0.8607   0.9743   0.9921   0.9301   0.8010  
0.45       0.9702   0.8617   0.9746   0.9921   0.9298   0.8029  
0.50       0.9703   0.8624   0.9749   0.9920   0.9294   0.8044  
0.55       0.9704   0.8626   0.9751   0.9919   0.9281   0.8057  
0.60       0.9703   0.8497   0.9849   0.9822   0.8392   0.8605  
0.65       0.9754   0.8658   0.9955   0.9775   0.7941   0.9517   <--
0.70       0.9751   0.8630   0.9965   0.9764   0.7829   0.9614  
0.75       0.9756   0.8648   0.9974   0.9760   0.7797   0.9709  
0.80       0.9756   0.8645   0.9976   0.9759   0.7781   0.9725  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.65
  At threshold 0.65: Accuracy=0.9754, F1=0.8658, Normal Recall=0.9955, Normal Precision=0.9775, Attack Recall=0.7941, Attack Precision=0.9517

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
0.15       0.9470   0.8799   0.9412   0.9922   0.9703   0.8050  
0.20       0.9595   0.9036   0.9617   0.9873   0.9504   0.8613  
0.25       0.9612   0.9070   0.9653   0.9859   0.9449   0.8720  
0.30       0.9623   0.9090   0.9673   0.9853   0.9423   0.8780  
0.35       0.9646   0.9135   0.9720   0.9835   0.9348   0.8931  
0.40       0.9653   0.9145   0.9744   0.9821   0.9289   0.9006  
0.45       0.9655   0.9150   0.9747   0.9820   0.9286   0.9017  
0.50       0.9656   0.9152   0.9749   0.9819   0.9282   0.9025   <--
0.55       0.9655   0.9148   0.9752   0.9816   0.9267   0.9032  
0.60       0.9554   0.8824   0.9849   0.9603   0.8372   0.9327  
0.65       0.9549   0.8756   0.9956   0.9505   0.7925   0.9781  
0.70       0.9535   0.8706   0.9965   0.9481   0.7817   0.9825  
0.75       0.9536   0.8704   0.9974   0.9474   0.7784   0.9869  
0.80       0.9534   0.8696   0.9976   0.9470   0.7767   0.9877  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.9656, F1=0.9152, Normal Recall=0.9749, Normal Precision=0.9819, Attack Recall=0.9282, Attack Precision=0.9025

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
0.15       0.9500   0.9209   0.9413   0.9867   0.9703   0.8763  
0.20       0.9582   0.9317   0.9616   0.9784   0.9504   0.9138  
0.25       0.9591   0.9327   0.9651   0.9761   0.9449   0.9207  
0.30       0.9595   0.9332   0.9669   0.9751   0.9423   0.9243  
0.35       0.9606   0.9344   0.9717   0.9721   0.9348   0.9340   <--
0.40       0.9605   0.9338   0.9741   0.9697   0.9289   0.9389  
0.45       0.9607   0.9340   0.9744   0.9696   0.9286   0.9395  
0.50       0.9607   0.9341   0.9746   0.9694   0.9282   0.9400  
0.55       0.9604   0.9335   0.9748   0.9688   0.9267   0.9403  
0.60       0.9406   0.8942   0.9848   0.9338   0.8372   0.9595  
0.65       0.9346   0.8791   0.9956   0.9180   0.7925   0.9871  
0.70       0.9320   0.8734   0.9965   0.9142   0.7817   0.9896  
0.75       0.9317   0.8724   0.9973   0.9131   0.7784   0.9921  
0.80       0.9313   0.8715   0.9975   0.9125   0.7767   0.9926  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9606, F1=0.9344, Normal Recall=0.9717, Normal Precision=0.9721, Attack Recall=0.9348, Attack Precision=0.9340

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
0.15       0.9529   0.9428   0.9413   0.9794   0.9703   0.9168  
0.20       0.9572   0.9467   0.9617   0.9667   0.9504   0.9431   <--
0.25       0.9572   0.9464   0.9654   0.9634   0.9449   0.9479  
0.30       0.9572   0.9463   0.9672   0.9618   0.9423   0.9503  
0.35       0.9570   0.9456   0.9717   0.9572   0.9348   0.9566  
0.40       0.9559   0.9440   0.9740   0.9536   0.9289   0.9596  
0.45       0.9560   0.9441   0.9743   0.9534   0.9286   0.9601  
0.50       0.9560   0.9440   0.9745   0.9532   0.9282   0.9604  
0.55       0.9555   0.9433   0.9746   0.9523   0.9267   0.9606  
0.60       0.9258   0.9002   0.9848   0.9007   0.8372   0.9735  
0.65       0.9143   0.8809   0.9954   0.8780   0.7925   0.9915  
0.70       0.9104   0.8747   0.9963   0.8725   0.7817   0.9929  
0.75       0.9097   0.8733   0.9972   0.8710   0.7784   0.9946  
0.80       0.9091   0.8724   0.9973   0.8701   0.7767   0.9949  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9572, F1=0.9467, Normal Recall=0.9617, Normal Precision=0.9667, Attack Recall=0.9504, Attack Precision=0.9431

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
0.15       0.9559   0.9565   0.9415   0.9694   0.9703   0.9432   <--
0.20       0.9563   0.9560   0.9622   0.9509   0.9504   0.9618  
0.25       0.9553   0.9549   0.9658   0.9460   0.9449   0.9651  
0.30       0.9549   0.9543   0.9675   0.9437   0.9423   0.9666  
0.35       0.9534   0.9525   0.9719   0.9372   0.9348   0.9708  
0.40       0.9516   0.9505   0.9743   0.9320   0.9289   0.9731  
0.45       0.9516   0.9505   0.9746   0.9318   0.9286   0.9734  
0.50       0.9515   0.9503   0.9748   0.9314   0.9282   0.9735  
0.55       0.9508   0.9496   0.9749   0.9301   0.9267   0.9737  
0.60       0.9112   0.9041   0.9851   0.8582   0.8372   0.9826  
0.65       0.8939   0.8820   0.9953   0.8275   0.7925   0.9942  
0.70       0.8890   0.8756   0.9963   0.8202   0.7817   0.9952  
0.75       0.8878   0.8740   0.9971   0.8182   0.7784   0.9963  
0.80       0.8870   0.8730   0.9973   0.8171   0.7767   0.9965  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9559, F1=0.9565, Normal Recall=0.9415, Normal Precision=0.9694, Attack Recall=0.9703, Attack Precision=0.9432

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
0.15       0.1000   0.1818   0.0000   1.0000   1.0000   0.1000  
0.20       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.25       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.30       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.35       0.1003   0.1819   0.0003   1.0000   1.0000   0.1000  
0.40       0.1031   0.1823   0.0034   1.0000   1.0000   0.1003  
0.45       0.1126   0.1836   0.0143   0.9820   0.9976   0.1011  
0.50       0.6105   0.2920   0.5891   0.9642   0.8031   0.1784   <--
0.55       0.9041   0.0980   0.9988   0.9046   0.0521   0.8238  
0.60       0.8994   0.0000   0.9993   0.8999   0.0000   0.0000  
0.65       0.8995   0.0000   0.9994   0.8999   0.0000   0.0000  
0.70       0.8995   0.0000   0.9995   0.9000   0.0000   0.0000  
0.75       0.8995   0.0000   0.9995   0.9000   0.0000   0.0000  
0.80       0.8996   0.0000   0.9995   0.9000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.6105, F1=0.2920, Normal Recall=0.5891, Normal Precision=0.9642, Attack Recall=0.8031, Attack Precision=0.1784

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
0.15       0.2000   0.3333   0.0000   0.0000   1.0000   0.2000  
0.20       0.2001   0.3333   0.0001   1.0000   1.0000   0.2000  
0.25       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.30       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.35       0.2003   0.3334   0.0003   1.0000   1.0000   0.2001  
0.40       0.2027   0.3341   0.0033   1.0000   1.0000   0.2005  
0.45       0.2108   0.3357   0.0142   0.9528   0.9972   0.2018  
0.50       0.6324   0.4663   0.5898   0.9229   0.8029   0.3286   <--
0.55       0.8093   0.0978   0.9987   0.8082   0.0517   0.9116  
0.60       0.7994   0.0000   0.9993   0.7999   0.0000   0.0061  
0.65       0.7995   0.0000   0.9994   0.7999   0.0000   0.0000  
0.70       0.7996   0.0000   0.9995   0.7999   0.0000   0.0000  
0.75       0.7996   0.0000   0.9995   0.7999   0.0000   0.0000  
0.80       0.7996   0.0000   0.9995   0.7999   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.6324, F1=0.4663, Normal Recall=0.5898, Normal Precision=0.9229, Attack Recall=0.8029, Attack Precision=0.3286

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
0.15       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.20       0.3000   0.4616   0.0001   1.0000   1.0000   0.3000  
0.25       0.3001   0.4616   0.0001   1.0000   1.0000   0.3000  
0.30       0.3001   0.4616   0.0001   1.0000   1.0000   0.3000  
0.35       0.3002   0.4616   0.0003   1.0000   1.0000   0.3001  
0.40       0.3024   0.4624   0.0034   1.0000   1.0000   0.3007  
0.45       0.3092   0.4641   0.0143   0.9221   0.9972   0.3024  
0.50       0.6545   0.5824   0.5909   0.8749   0.8029   0.4569   <--
0.55       0.7147   0.0980   0.9988   0.7108   0.0517   0.9493  
0.60       0.6996   0.0000   0.9994   0.6999   0.0000   0.0116  
0.65       0.6996   0.0000   0.9995   0.6999   0.0000   0.0000  
0.70       0.6997   0.0000   0.9995   0.6999   0.0000   0.0000  
0.75       0.6997   0.0000   0.9995   0.6999   0.0000   0.0000  
0.80       0.6997   0.0000   0.9996   0.6999   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.6545, F1=0.5824, Normal Recall=0.5909, Normal Precision=0.8749, Attack Recall=0.8029, Attack Precision=0.4569

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
0.15       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.20       0.4000   0.5714   0.0001   1.0000   1.0000   0.4000  
0.25       0.4000   0.5714   0.0001   1.0000   1.0000   0.4000  
0.30       0.4000   0.5714   0.0001   1.0000   1.0000   0.4000  
0.35       0.4002   0.5715   0.0003   1.0000   1.0000   0.4001  
0.40       0.4020   0.5722   0.0033   1.0000   1.0000   0.4008  
0.45       0.4072   0.5737   0.0138   0.8806   0.9972   0.4027  
0.50       0.6758   0.6646   0.5911   0.8181   0.8029   0.5669   <--
0.55       0.6200   0.0981   0.9988   0.6124   0.0517   0.9676  
0.60       0.5996   0.0000   0.9994   0.5999   0.0000   0.0182  
0.65       0.5997   0.0000   0.9995   0.5999   0.0000   0.0000  
0.70       0.5997   0.0000   0.9995   0.5999   0.0000   0.0000  
0.75       0.5997   0.0000   0.9995   0.5999   0.0000   0.0000  
0.80       0.5997   0.0000   0.9996   0.5999   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.6758, F1=0.6646, Normal Recall=0.5911, Normal Precision=0.8181, Attack Recall=0.8029, Attack Precision=0.5669

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
0.15       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.20       0.5000   0.6667   0.0001   1.0000   1.0000   0.5000  
0.25       0.5000   0.6667   0.0001   1.0000   1.0000   0.5000  
0.30       0.5000   0.6667   0.0001   1.0000   1.0000   0.5000  
0.35       0.5001   0.6667   0.0002   1.0000   1.0000   0.5001  
0.40       0.5017   0.6674   0.0033   1.0000   1.0000   0.5008  
0.45       0.5054   0.6685   0.0137   0.8295   0.9972   0.5027  
0.50       0.6971   0.7261   0.5913   0.7500   0.8029   0.6627   <--
0.55       0.5253   0.0982   0.9989   0.5130   0.0517   0.9786  
0.60       0.4997   0.0000   0.9995   0.4999   0.0000   0.0312  
0.65       0.4998   0.0000   0.9996   0.4999   0.0000   0.0000  
0.70       0.4998   0.0000   0.9996   0.4999   0.0000   0.0000  
0.75       0.4998   0.0000   0.9996   0.4999   0.0000   0.0000  
0.80       0.4998   0.0000   0.9996   0.4999   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.6971, F1=0.7261, Normal Recall=0.5913, Normal Precision=0.7500, Attack Recall=0.8029, Attack Precision=0.6627

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
0.15       0.9615   0.8281   0.9651   0.9918   0.9286   0.7473  
0.20       0.9636   0.8360   0.9676   0.9917   0.9274   0.7610  
0.25       0.9649   0.8402   0.9695   0.9912   0.9229   0.7710  
0.30       0.9653   0.8404   0.9712   0.9901   0.9125   0.7788  
0.35       0.9658   0.8414   0.9724   0.9894   0.9064   0.7851  
0.40       0.9665   0.8433   0.9736   0.9890   0.9024   0.7915  
0.45       0.9665   0.8429   0.9741   0.9885   0.8984   0.7939  
0.50       0.9667   0.8434   0.9744   0.9884   0.8971   0.7958   <--
0.55       0.9658   0.8344   0.9772   0.9847   0.8629   0.8078  
0.60       0.9597   0.7951   0.9796   0.9758   0.7811   0.8096  
0.65       0.9685   0.8222   0.9954   0.9704   0.7272   0.9456  
0.70       0.9676   0.8109   0.9981   0.9670   0.6937   0.9758  
0.75       0.9476   0.6479   0.9993   0.9456   0.4823   0.9864  
0.80       0.9462   0.6342   0.9996   0.9440   0.4661   0.9920  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.9667, F1=0.8434, Normal Recall=0.9744, Normal Precision=0.9884, Attack Recall=0.8971, Attack Precision=0.7958

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
0.15       0.9577   0.8976   0.9652   0.9816   0.9276   0.8695  
0.20       0.9594   0.9012   0.9677   0.9813   0.9263   0.8775  
0.25       0.9600   0.9021   0.9695   0.9802   0.9218   0.8833   <--
0.30       0.9592   0.8993   0.9712   0.9776   0.9111   0.8878  
0.35       0.9591   0.8984   0.9724   0.9763   0.9055   0.8914  
0.40       0.9592   0.8984   0.9736   0.9753   0.9016   0.8952  
0.45       0.9588   0.8972   0.9741   0.9744   0.8978   0.8966  
0.50       0.9589   0.8971   0.9744   0.9741   0.8965   0.8977  
0.55       0.9542   0.8827   0.9773   0.9659   0.8620   0.9045  
0.60       0.9396   0.8378   0.9796   0.9467   0.7796   0.9053  
0.65       0.9415   0.8321   0.9954   0.9355   0.7255   0.9754  
0.70       0.9370   0.8148   0.9981   0.9285   0.6927   0.9891  
0.75       0.8959   0.6497   0.9993   0.8854   0.4825   0.9940  
0.80       0.8930   0.6356   0.9996   0.8823   0.4666   0.9965  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.9600, F1=0.9021, Normal Recall=0.9695, Normal Precision=0.9802, Attack Recall=0.9218, Attack Precision=0.8833

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
0.15       0.9537   0.9232   0.9649   0.9689   0.9276   0.9188  
0.20       0.9550   0.9252   0.9674   0.9684   0.9263   0.9240   <--
0.25       0.9551   0.9249   0.9693   0.9666   0.9218   0.9279  
0.30       0.9530   0.9209   0.9710   0.9622   0.9111   0.9309  
0.35       0.9522   0.9192   0.9722   0.9600   0.9055   0.9332  
0.40       0.9519   0.9183   0.9734   0.9585   0.9016   0.9357  
0.45       0.9511   0.9167   0.9739   0.9569   0.8978   0.9365  
0.50       0.9509   0.9164   0.9742   0.9565   0.8965   0.9372  
0.55       0.9426   0.9001   0.9772   0.9429   0.8620   0.9419  
0.60       0.9195   0.8532   0.9795   0.9120   0.7796   0.9423  
0.65       0.9145   0.8359   0.9956   0.8943   0.7255   0.9859  
0.70       0.9066   0.8164   0.9982   0.8834   0.6927   0.9941  
0.75       0.8443   0.6503   0.9994   0.8184   0.4825   0.9969  
0.80       0.8397   0.6359   0.9996   0.8139   0.4666   0.9981  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9550, F1=0.9252, Normal Recall=0.9674, Normal Precision=0.9684, Attack Recall=0.9263, Attack Precision=0.9240

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
0.15       0.9499   0.9368   0.9647   0.9524   0.9276   0.9460  
0.20       0.9509   0.9378   0.9672   0.9517   0.9263   0.9496   <--
0.25       0.9503   0.9368   0.9692   0.9490   0.9218   0.9523  
0.30       0.9469   0.9321   0.9708   0.9425   0.9111   0.9541  
0.35       0.9455   0.9300   0.9721   0.9392   0.9055   0.9559  
0.40       0.9447   0.9288   0.9735   0.9368   0.9016   0.9578  
0.45       0.9435   0.9270   0.9739   0.9346   0.8978   0.9583  
0.50       0.9432   0.9266   0.9743   0.9339   0.8965   0.9587  
0.55       0.9312   0.9092   0.9773   0.9139   0.8620   0.9620  
0.60       0.8997   0.8614   0.9797   0.8696   0.7796   0.9625  
0.65       0.8876   0.8377   0.9956   0.8447   0.7255   0.9910  
0.70       0.8760   0.8171   0.9982   0.8297   0.6927   0.9962  
0.75       0.7927   0.6506   0.9994   0.7434   0.4825   0.9982  
0.80       0.7865   0.6361   0.9997   0.7376   0.4666   0.9990  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9509, F1=0.9378, Normal Recall=0.9672, Normal Precision=0.9517, Attack Recall=0.9263, Attack Precision=0.9496

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
0.15       0.9464   0.9454   0.9651   0.9303   0.9276   0.9638  
0.20       0.9470   0.9459   0.9677   0.9292   0.9263   0.9663   <--
0.25       0.9457   0.9444   0.9697   0.9254   0.9218   0.9681  
0.30       0.9411   0.9393   0.9712   0.9161   0.9111   0.9693  
0.35       0.9390   0.9369   0.9725   0.9115   0.9055   0.9706  
0.40       0.9377   0.9353   0.9738   0.9082   0.9016   0.9717  
0.45       0.9360   0.9335   0.9742   0.9050   0.8978   0.9721  
0.50       0.9356   0.9330   0.9746   0.9040   0.8965   0.9725  
0.55       0.9199   0.9150   0.9778   0.8763   0.8620   0.9749  
0.60       0.8798   0.8664   0.9801   0.8164   0.7796   0.9751  
0.65       0.8606   0.8388   0.9956   0.7839   0.7255   0.9940  
0.70       0.8455   0.8176   0.9982   0.7646   0.6927   0.9975  
0.75       0.7410   0.6507   0.9995   0.6589   0.4825   0.9989  
0.80       0.7331   0.6362   0.9997   0.6521   0.4666   0.9993  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9470, F1=0.9459, Normal Recall=0.9677, Normal Precision=0.9292, Attack Recall=0.9263, Attack Precision=0.9663

```

