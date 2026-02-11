# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-11 07:17:59 |

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
| Original (TFLite) | 0.3881 | 0.4444 | 0.5010 | 0.5565 | 0.6122 | 0.6698 | 0.7266 | 0.7804 | 0.8388 | 0.8952 | 0.9516 |
| QAT+Prune only | 0.9467 | 0.9477 | 0.9494 | 0.9514 | 0.9532 | 0.9544 | 0.9557 | 0.9570 | 0.9594 | 0.9611 | 0.9628 |
| QAT+PTQ | 0.9467 | 0.9478 | 0.9494 | 0.9513 | 0.9532 | 0.9543 | 0.9556 | 0.9569 | 0.9593 | 0.9609 | 0.9626 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9467 | 0.9478 | 0.9494 | 0.9513 | 0.9532 | 0.9543 | 0.9556 | 0.9569 | 0.9593 | 0.9609 | 0.9626 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2551 | 0.4327 | 0.5628 | 0.6625 | 0.7424 | 0.8068 | 0.8585 | 0.9043 | 0.9424 | 0.9752 |
| QAT+Prune only | 0.0000 | 0.7863 | 0.8838 | 0.9223 | 0.9427 | 0.9548 | 0.9631 | 0.9691 | 0.9743 | 0.9780 | 0.9811 |
| QAT+PTQ | 0.0000 | 0.7866 | 0.8839 | 0.9223 | 0.9427 | 0.9547 | 0.9630 | 0.9690 | 0.9742 | 0.9779 | 0.9810 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.7866 | 0.8839 | 0.9223 | 0.9427 | 0.9547 | 0.9630 | 0.9690 | 0.9742 | 0.9779 | 0.9810 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3881 | 0.3881 | 0.3883 | 0.3872 | 0.3859 | 0.3880 | 0.3891 | 0.3811 | 0.3876 | 0.3882 | 0.0000 |
| QAT+Prune only | 0.9467 | 0.9460 | 0.9460 | 0.9464 | 0.9468 | 0.9459 | 0.9450 | 0.9434 | 0.9455 | 0.9451 | 0.0000 |
| QAT+PTQ | 0.9467 | 0.9461 | 0.9461 | 0.9465 | 0.9469 | 0.9460 | 0.9450 | 0.9434 | 0.9458 | 0.9452 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9467 | 0.9461 | 0.9461 | 0.9465 | 0.9469 | 0.9460 | 0.9450 | 0.9434 | 0.9458 | 0.9452 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3881 | 0.0000 | 0.0000 | 0.0000 | 0.3881 | 1.0000 |
| 90 | 10 | 299,980 | 0.4444 | 0.1473 | 0.9513 | 0.2551 | 0.3881 | 0.9863 |
| 80 | 20 | 290,860 | 0.5010 | 0.2800 | 0.9516 | 0.4327 | 0.3883 | 0.9698 |
| 70 | 30 | 193,903 | 0.5565 | 0.3996 | 0.9516 | 0.5628 | 0.3872 | 0.9491 |
| 60 | 40 | 145,430 | 0.6122 | 0.5081 | 0.9516 | 0.6625 | 0.3859 | 0.9228 |
| 50 | 50 | 116,344 | 0.6698 | 0.6086 | 0.9516 | 0.7424 | 0.3880 | 0.8891 |
| 40 | 60 | 96,951 | 0.7266 | 0.7003 | 0.9516 | 0.8068 | 0.3891 | 0.8427 |
| 30 | 70 | 83,100 | 0.7804 | 0.7820 | 0.9516 | 0.8585 | 0.3811 | 0.7714 |
| 20 | 80 | 72,715 | 0.8388 | 0.8614 | 0.9516 | 0.9043 | 0.3876 | 0.6669 |
| 10 | 90 | 64,630 | 0.8952 | 0.9333 | 0.9516 | 0.9424 | 0.3882 | 0.4712 |
| 0 | 100 | 58,172 | 0.9516 | 1.0000 | 0.9516 | 0.9752 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9467 | 0.0000 | 0.0000 | 0.0000 | 0.9467 | 1.0000 |
| 90 | 10 | 299,980 | 0.9477 | 0.6645 | 0.9629 | 0.7863 | 0.9460 | 0.9957 |
| 80 | 20 | 290,860 | 0.9494 | 0.8167 | 0.9628 | 0.8838 | 0.9460 | 0.9903 |
| 70 | 30 | 193,903 | 0.9514 | 0.8851 | 0.9628 | 0.9223 | 0.9464 | 0.9834 |
| 60 | 40 | 145,430 | 0.9532 | 0.9235 | 0.9628 | 0.9427 | 0.9468 | 0.9745 |
| 50 | 50 | 116,344 | 0.9544 | 0.9468 | 0.9628 | 0.9548 | 0.9459 | 0.9622 |
| 40 | 60 | 96,951 | 0.9557 | 0.9633 | 0.9628 | 0.9631 | 0.9450 | 0.9443 |
| 30 | 70 | 83,100 | 0.9570 | 0.9754 | 0.9628 | 0.9691 | 0.9434 | 0.9158 |
| 20 | 80 | 72,715 | 0.9594 | 0.9860 | 0.9628 | 0.9743 | 0.9455 | 0.8641 |
| 10 | 90 | 64,630 | 0.9611 | 0.9937 | 0.9628 | 0.9780 | 0.9451 | 0.7386 |
| 0 | 100 | 58,172 | 0.9628 | 1.0000 | 0.9628 | 0.9811 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9467 | 0.0000 | 0.0000 | 0.0000 | 0.9467 | 1.0000 |
| 90 | 10 | 299,980 | 0.9478 | 0.6650 | 0.9628 | 0.7866 | 0.9461 | 0.9957 |
| 80 | 20 | 290,860 | 0.9494 | 0.8170 | 0.9626 | 0.8839 | 0.9461 | 0.9902 |
| 70 | 30 | 193,903 | 0.9513 | 0.8852 | 0.9626 | 0.9223 | 0.9465 | 0.9834 |
| 60 | 40 | 145,430 | 0.9532 | 0.9235 | 0.9626 | 0.9427 | 0.9469 | 0.9744 |
| 50 | 50 | 116,344 | 0.9543 | 0.9469 | 0.9626 | 0.9547 | 0.9460 | 0.9620 |
| 40 | 60 | 96,951 | 0.9556 | 0.9633 | 0.9626 | 0.9630 | 0.9450 | 0.9440 |
| 30 | 70 | 83,100 | 0.9569 | 0.9754 | 0.9626 | 0.9690 | 0.9434 | 0.9154 |
| 20 | 80 | 72,715 | 0.9593 | 0.9861 | 0.9626 | 0.9742 | 0.9458 | 0.8636 |
| 10 | 90 | 64,630 | 0.9609 | 0.9937 | 0.9626 | 0.9779 | 0.9452 | 0.7376 |
| 0 | 100 | 58,172 | 0.9626 | 1.0000 | 0.9626 | 0.9810 | 0.0000 | 0.0000 |

### noQAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 90 | 10 | 299,980 | 0.9000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.9000 |
| 80 | 20 | 290,860 | 0.8000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.8000 |
| 70 | 30 | 193,903 | 0.7000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.7000 |
| 60 | 40 | 145,430 | 0.6000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.6000 |
| 50 | 50 | 116,344 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.5000 |
| 40 | 60 | 96,951 | 0.4000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.4000 |
| 30 | 70 | 83,100 | 0.3000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.3000 |
| 20 | 80 | 72,715 | 0.2000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2000 |
| 10 | 90 | 64,630 | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1000 |
| 0 | 100 | 58,172 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Compressed (PTQ)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9467 | 0.0000 | 0.0000 | 0.0000 | 0.9467 | 1.0000 |
| 90 | 10 | 299,980 | 0.9478 | 0.6650 | 0.9628 | 0.7866 | 0.9461 | 0.9957 |
| 80 | 20 | 290,860 | 0.9494 | 0.8170 | 0.9626 | 0.8839 | 0.9461 | 0.9902 |
| 70 | 30 | 193,903 | 0.9513 | 0.8852 | 0.9626 | 0.9223 | 0.9465 | 0.9834 |
| 60 | 40 | 145,430 | 0.9532 | 0.9235 | 0.9626 | 0.9427 | 0.9469 | 0.9744 |
| 50 | 50 | 116,344 | 0.9543 | 0.9469 | 0.9626 | 0.9547 | 0.9460 | 0.9620 |
| 40 | 60 | 96,951 | 0.9556 | 0.9633 | 0.9626 | 0.9630 | 0.9450 | 0.9440 |
| 30 | 70 | 83,100 | 0.9569 | 0.9754 | 0.9626 | 0.9690 | 0.9434 | 0.9154 |
| 20 | 80 | 72,715 | 0.9593 | 0.9861 | 0.9626 | 0.9742 | 0.9458 | 0.8636 |
| 10 | 90 | 64,630 | 0.9609 | 0.9937 | 0.9626 | 0.9779 | 0.9452 | 0.7376 |
| 0 | 100 | 58,172 | 0.9626 | 1.0000 | 0.9626 | 0.9810 | 0.0000 | 0.0000 |


## Threshold Tuning (Original)

Model: `models/tflite/saved_model_original.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 90% : Attack 10%  (n=299,980, N=269,982, A=29,998)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.4443   0.2549   0.3881   0.9861   0.9506   0.1472   <--
0.20       0.4443   0.2549   0.3881   0.9861   0.9506   0.1472  
0.25       0.4443   0.2549   0.3881   0.9861   0.9506   0.1472  
0.30       0.4443   0.2549   0.3881   0.9861   0.9506   0.1472  
0.35       0.4443   0.2549   0.3881   0.9861   0.9506   0.1472  
0.40       0.4443   0.2549   0.3881   0.9861   0.9506   0.1472  
0.45       0.4443   0.2549   0.3881   0.9861   0.9506   0.1472  
0.50       0.4443   0.2549   0.3881   0.9861   0.9506   0.1472  
0.55       0.4443   0.2549   0.3881   0.9861   0.9506   0.1472  
0.60       0.4443   0.2549   0.3881   0.9861   0.9506   0.1472  
0.65       0.4443   0.2549   0.3881   0.9861   0.9506   0.1472  
0.70       0.4443   0.2549   0.3881   0.9861   0.9506   0.1472  
0.75       0.4443   0.2549   0.3881   0.9861   0.9506   0.1472  
0.80       0.4443   0.2549   0.3881   0.9861   0.9506   0.1472  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4443, F1=0.2549, Normal Recall=0.3881, Normal Precision=0.9861, Attack Recall=0.9506, Attack Precision=0.1472

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 80% : Attack 20%  (n=290,860, N=232,688, A=58,172)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.5010   0.4327   0.3884   0.9698   0.9516   0.2800   <--
0.20       0.5010   0.4327   0.3884   0.9698   0.9516   0.2800  
0.25       0.5010   0.4327   0.3884   0.9698   0.9516   0.2800  
0.30       0.5010   0.4327   0.3884   0.9698   0.9516   0.2800  
0.35       0.5010   0.4327   0.3884   0.9698   0.9516   0.2800  
0.40       0.5010   0.4327   0.3884   0.9698   0.9516   0.2800  
0.45       0.5010   0.4327   0.3884   0.9698   0.9516   0.2800  
0.50       0.5010   0.4327   0.3884   0.9698   0.9516   0.2800  
0.55       0.5010   0.4327   0.3884   0.9698   0.9516   0.2800  
0.60       0.5010   0.4327   0.3884   0.9698   0.9516   0.2800  
0.65       0.5010   0.4327   0.3884   0.9698   0.9516   0.2800  
0.70       0.5010   0.4327   0.3884   0.9698   0.9516   0.2800  
0.75       0.5010   0.4327   0.3884   0.9698   0.9516   0.2800  
0.80       0.5010   0.4327   0.3884   0.9698   0.9516   0.2800  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5010, F1=0.4327, Normal Recall=0.3884, Normal Precision=0.9698, Attack Recall=0.9516, Attack Precision=0.2800

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 70% : Attack 30%  (n=193,903, N=135,732, A=58,171)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.5570   0.5631   0.3879   0.9492   0.9516   0.3999   <--
0.20       0.5570   0.5631   0.3879   0.9492   0.9516   0.3999  
0.25       0.5570   0.5631   0.3879   0.9492   0.9516   0.3999  
0.30       0.5570   0.5631   0.3879   0.9492   0.9516   0.3999  
0.35       0.5570   0.5631   0.3879   0.9492   0.9516   0.3999  
0.40       0.5570   0.5631   0.3879   0.9492   0.9516   0.3999  
0.45       0.5570   0.5631   0.3879   0.9492   0.9516   0.3999  
0.50       0.5570   0.5631   0.3879   0.9492   0.9516   0.3999  
0.55       0.5570   0.5631   0.3879   0.9492   0.9516   0.3999  
0.60       0.5570   0.5631   0.3879   0.9492   0.9516   0.3999  
0.65       0.5570   0.5631   0.3879   0.9492   0.9516   0.3999  
0.70       0.5570   0.5631   0.3879   0.9492   0.9516   0.3999  
0.75       0.5570   0.5631   0.3879   0.9492   0.9516   0.3999  
0.80       0.5570   0.5631   0.3879   0.9492   0.9516   0.3999  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5570, F1=0.5631, Normal Recall=0.3879, Normal Precision=0.9492, Attack Recall=0.9516, Attack Precision=0.3999

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 60% : Attack 40%  (n=145,430, N=87,258, A=58,172)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.6137   0.6634   0.3884   0.9233   0.9516   0.5092   <--
0.20       0.6137   0.6634   0.3884   0.9233   0.9516   0.5092  
0.25       0.6137   0.6634   0.3884   0.9233   0.9516   0.5092  
0.30       0.6137   0.6634   0.3884   0.9233   0.9516   0.5092  
0.35       0.6137   0.6634   0.3884   0.9233   0.9516   0.5092  
0.40       0.6137   0.6634   0.3884   0.9233   0.9516   0.5092  
0.45       0.6137   0.6634   0.3884   0.9233   0.9516   0.5092  
0.50       0.6137   0.6634   0.3884   0.9233   0.9516   0.5092  
0.55       0.6137   0.6634   0.3884   0.9233   0.9516   0.5092  
0.60       0.6137   0.6634   0.3884   0.9233   0.9516   0.5092  
0.65       0.6137   0.6634   0.3884   0.9233   0.9516   0.5092  
0.70       0.6137   0.6634   0.3884   0.9233   0.9516   0.5092  
0.75       0.6137   0.6634   0.3884   0.9233   0.9516   0.5092  
0.80       0.6137   0.6634   0.3884   0.9233   0.9516   0.5092  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6137, F1=0.6634, Normal Recall=0.3884, Normal Precision=0.9233, Attack Recall=0.9516, Attack Precision=0.5092

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 50% : Attack 50%  (n=116,344, N=58,172, A=58,172)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.6699   0.7424   0.3882   0.8891   0.9516   0.6087   <--
0.20       0.6699   0.7424   0.3882   0.8891   0.9516   0.6087  
0.25       0.6699   0.7424   0.3882   0.8891   0.9516   0.6087  
0.30       0.6699   0.7424   0.3882   0.8891   0.9516   0.6087  
0.35       0.6699   0.7424   0.3882   0.8891   0.9516   0.6087  
0.40       0.6699   0.7424   0.3882   0.8891   0.9516   0.6087  
0.45       0.6699   0.7424   0.3882   0.8891   0.9516   0.6087  
0.50       0.6699   0.7424   0.3882   0.8891   0.9516   0.6087  
0.55       0.6699   0.7424   0.3882   0.8891   0.9516   0.6087  
0.60       0.6699   0.7424   0.3882   0.8891   0.9516   0.6087  
0.65       0.6699   0.7424   0.3882   0.8891   0.9516   0.6087  
0.70       0.6699   0.7424   0.3882   0.8891   0.9516   0.6087  
0.75       0.6699   0.7424   0.3882   0.8891   0.9516   0.6087  
0.80       0.6699   0.7424   0.3882   0.8891   0.9516   0.6087  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6699, F1=0.7424, Normal Recall=0.3882, Normal Precision=0.8891, Attack Recall=0.9516, Attack Precision=0.6087

```


## Threshold Tuning (QAT+Prune only)

Model: `models/tflite/saved_model_qat_pruned_float32.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_qat_pruned_float32.tflite

Test set: Normal 90% : Attack 10%  (n=299,980, N=269,982, A=29,998)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9477   0.7864   0.9460   0.9957   0.9630   0.6645   <--
0.20       0.9477   0.7864   0.9460   0.9957   0.9630   0.6645  
0.25       0.9477   0.7864   0.9460   0.9957   0.9630   0.6645  
0.30       0.9477   0.7864   0.9460   0.9957   0.9630   0.6645  
0.35       0.9477   0.7864   0.9460   0.9957   0.9630   0.6645  
0.40       0.9477   0.7864   0.9460   0.9957   0.9630   0.6645  
0.45       0.9477   0.7864   0.9460   0.9957   0.9630   0.6645  
0.50       0.9477   0.7864   0.9460   0.9957   0.9630   0.6645  
0.55       0.9477   0.7864   0.9460   0.9957   0.9630   0.6645  
0.60       0.9477   0.7864   0.9460   0.9957   0.9630   0.6645  
0.65       0.9477   0.7864   0.9460   0.9957   0.9630   0.6645  
0.70       0.9477   0.7864   0.9460   0.9957   0.9630   0.6645  
0.75       0.9477   0.7864   0.9460   0.9957   0.9630   0.6645  
0.80       0.9477   0.7864   0.9460   0.9957   0.9630   0.6645  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9477, F1=0.7864, Normal Recall=0.9460, Normal Precision=0.9957, Attack Recall=0.9630, Attack Precision=0.6645

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_qat_pruned_float32.tflite

Test set: Normal 80% : Attack 20%  (n=290,860, N=232,688, A=58,172)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9492   0.8835   0.9458   0.9903   0.9628   0.8163   <--
0.20       0.9492   0.8835   0.9458   0.9903   0.9628   0.8163  
0.25       0.9492   0.8835   0.9458   0.9903   0.9628   0.8163  
0.30       0.9492   0.8835   0.9458   0.9903   0.9628   0.8163  
0.35       0.9492   0.8835   0.9458   0.9903   0.9628   0.8163  
0.40       0.9492   0.8835   0.9458   0.9903   0.9628   0.8163  
0.45       0.9492   0.8835   0.9458   0.9903   0.9628   0.8163  
0.50       0.9492   0.8835   0.9458   0.9903   0.9628   0.8163  
0.55       0.9492   0.8835   0.9458   0.9903   0.9628   0.8163  
0.60       0.9492   0.8835   0.9458   0.9903   0.9628   0.8163  
0.65       0.9492   0.8835   0.9458   0.9903   0.9628   0.8163  
0.70       0.9492   0.8835   0.9458   0.9903   0.9628   0.8163  
0.75       0.9492   0.8835   0.9458   0.9903   0.9628   0.8163  
0.80       0.9492   0.8835   0.9458   0.9903   0.9628   0.8163  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9492, F1=0.8835, Normal Recall=0.9458, Normal Precision=0.9903, Attack Recall=0.9628, Attack Precision=0.8163

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_qat_pruned_float32.tflite

Test set: Normal 70% : Attack 30%  (n=193,903, N=135,732, A=58,171)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9510   0.9218   0.9460   0.9834   0.9628   0.8842   <--
0.20       0.9510   0.9218   0.9460   0.9834   0.9628   0.8842  
0.25       0.9510   0.9218   0.9460   0.9834   0.9628   0.8842  
0.30       0.9510   0.9218   0.9460   0.9834   0.9628   0.8842  
0.35       0.9510   0.9218   0.9460   0.9834   0.9628   0.8842  
0.40       0.9510   0.9218   0.9460   0.9834   0.9628   0.8842  
0.45       0.9510   0.9218   0.9460   0.9834   0.9628   0.8842  
0.50       0.9510   0.9218   0.9460   0.9834   0.9628   0.8842  
0.55       0.9510   0.9218   0.9460   0.9834   0.9628   0.8842  
0.60       0.9510   0.9218   0.9460   0.9834   0.9628   0.8842  
0.65       0.9510   0.9218   0.9460   0.9834   0.9628   0.8842  
0.70       0.9510   0.9218   0.9460   0.9834   0.9628   0.8842  
0.75       0.9510   0.9218   0.9460   0.9834   0.9628   0.8842  
0.80       0.9510   0.9218   0.9460   0.9834   0.9628   0.8842  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9510, F1=0.9218, Normal Recall=0.9460, Normal Precision=0.9834, Attack Recall=0.9628, Attack Precision=0.8842

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_qat_pruned_float32.tflite

Test set: Normal 60% : Attack 40%  (n=145,430, N=87,258, A=58,172)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9532   0.9428   0.9468   0.9745   0.9628   0.9235   <--
0.20       0.9532   0.9428   0.9468   0.9745   0.9628   0.9235  
0.25       0.9532   0.9428   0.9468   0.9745   0.9628   0.9235  
0.30       0.9532   0.9428   0.9468   0.9745   0.9628   0.9235  
0.35       0.9532   0.9428   0.9468   0.9745   0.9628   0.9235  
0.40       0.9532   0.9428   0.9468   0.9745   0.9628   0.9235  
0.45       0.9532   0.9428   0.9468   0.9745   0.9628   0.9235  
0.50       0.9532   0.9428   0.9468   0.9745   0.9628   0.9235  
0.55       0.9532   0.9428   0.9468   0.9745   0.9628   0.9235  
0.60       0.9532   0.9428   0.9468   0.9745   0.9628   0.9235  
0.65       0.9532   0.9428   0.9468   0.9745   0.9628   0.9235  
0.70       0.9532   0.9428   0.9468   0.9745   0.9628   0.9235  
0.75       0.9532   0.9428   0.9468   0.9745   0.9628   0.9235  
0.80       0.9532   0.9428   0.9468   0.9745   0.9628   0.9235  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9532, F1=0.9428, Normal Recall=0.9468, Normal Precision=0.9745, Attack Recall=0.9628, Attack Precision=0.9235

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_qat_pruned_float32.tflite

Test set: Normal 50% : Attack 50%  (n=116,344, N=58,172, A=58,172)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9553   0.9556   0.9478   0.9623   0.9628   0.9485   <--
0.20       0.9553   0.9556   0.9478   0.9623   0.9628   0.9485  
0.25       0.9553   0.9556   0.9478   0.9623   0.9628   0.9485  
0.30       0.9553   0.9556   0.9478   0.9623   0.9628   0.9485  
0.35       0.9553   0.9556   0.9478   0.9623   0.9628   0.9485  
0.40       0.9553   0.9556   0.9478   0.9623   0.9628   0.9485  
0.45       0.9553   0.9556   0.9478   0.9623   0.9628   0.9485  
0.50       0.9553   0.9556   0.9478   0.9623   0.9628   0.9485  
0.55       0.9553   0.9556   0.9478   0.9623   0.9628   0.9485  
0.60       0.9553   0.9556   0.9478   0.9623   0.9628   0.9485  
0.65       0.9553   0.9556   0.9478   0.9623   0.9628   0.9485  
0.70       0.9553   0.9556   0.9478   0.9623   0.9628   0.9485  
0.75       0.9553   0.9556   0.9478   0.9623   0.9628   0.9485  
0.80       0.9553   0.9556   0.9478   0.9623   0.9628   0.9485  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9553, F1=0.9556, Normal Recall=0.9478, Normal Precision=0.9623, Attack Recall=0.9628, Attack Precision=0.9485

```


## Threshold Tuning (QAT+PTQ)

Model: `models/tflite/saved_model_qat_ptq.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 90% : Attack 10%  (n=299,980, N=269,982, A=29,998)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650   <--
0.20       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.25       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.30       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.35       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.40       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.45       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.50       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.55       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.60       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.65       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.70       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.75       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.80       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9478, F1=0.7867, Normal Recall=0.9461, Normal Precision=0.9957, Attack Recall=0.9629, Attack Precision=0.6650

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 80% : Attack 20%  (n=290,860, N=232,688, A=58,172)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166   <--
0.20       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.25       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.30       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.35       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.40       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.45       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.50       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.55       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.60       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.65       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.70       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.75       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.80       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9493, F1=0.8836, Normal Recall=0.9460, Normal Precision=0.9902, Attack Recall=0.9626, Attack Precision=0.8166

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 70% : Attack 30%  (n=193,903, N=135,732, A=58,171)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843   <--
0.20       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.25       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.30       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.35       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.40       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.45       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.50       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.55       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.60       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.65       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.70       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.75       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.80       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9510, F1=0.9218, Normal Recall=0.9460, Normal Precision=0.9834, Attack Recall=0.9626, Attack Precision=0.8843

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 60% : Attack 40%  (n=145,430, N=87,258, A=58,172)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235   <--
0.20       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.25       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.30       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.35       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.40       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.45       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.50       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.55       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.60       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.65       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.70       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.75       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.80       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9532, F1=0.9427, Normal Recall=0.9469, Normal Precision=0.9744, Attack Recall=0.9626, Attack Precision=0.9235

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 50% : Attack 50%  (n=116,344, N=58,172, A=58,172)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485   <--
0.20       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.25       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.30       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.35       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.40       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.45       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.50       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.55       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.60       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.65       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.70       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.75       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.80       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9552, F1=0.9555, Normal Recall=0.9478, Normal Precision=0.9621, Attack Recall=0.9626, Attack Precision=0.9485

```


## Threshold Tuning (noQAT+PTQ)

Model: `models/tflite/saved_model_no_qat_ptq.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 90% : Attack 10%  (n=299,980, N=269,982, A=29,998)
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
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 80% : Attack 20%  (n=290,860, N=232,688, A=58,172)
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
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 70% : Attack 30%  (n=193,903, N=135,732, A=58,171)
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
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 60% : Attack 40%  (n=145,430, N=87,258, A=58,172)
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
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 50% : Attack 50%  (n=116,344, N=58,172, A=58,172)
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
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 90% : Attack 10%  (n=299,980, N=269,982, A=29,998)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650   <--
0.20       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.25       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.30       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.35       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.40       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.45       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.50       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.55       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.60       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.65       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.70       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.75       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
0.80       0.9478   0.7867   0.9461   0.9957   0.9629   0.6650  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9478, F1=0.7867, Normal Recall=0.9461, Normal Precision=0.9957, Attack Recall=0.9629, Attack Precision=0.6650

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 80% : Attack 20%  (n=290,860, N=232,688, A=58,172)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166   <--
0.20       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.25       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.30       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.35       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.40       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.45       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.50       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.55       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.60       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.65       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.70       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.75       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
0.80       0.9493   0.8836   0.9460   0.9902   0.9626   0.8166  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9493, F1=0.8836, Normal Recall=0.9460, Normal Precision=0.9902, Attack Recall=0.9626, Attack Precision=0.8166

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 70% : Attack 30%  (n=193,903, N=135,732, A=58,171)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843   <--
0.20       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.25       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.30       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.35       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.40       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.45       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.50       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.55       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.60       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.65       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.70       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.75       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
0.80       0.9510   0.9218   0.9460   0.9834   0.9626   0.8843  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9510, F1=0.9218, Normal Recall=0.9460, Normal Precision=0.9834, Attack Recall=0.9626, Attack Precision=0.8843

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 60% : Attack 40%  (n=145,430, N=87,258, A=58,172)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235   <--
0.20       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.25       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.30       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.35       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.40       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.45       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.50       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.55       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.60       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.65       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.70       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.75       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
0.80       0.9532   0.9427   0.9469   0.9744   0.9626   0.9235  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9532, F1=0.9427, Normal Recall=0.9469, Normal Precision=0.9744, Attack Recall=0.9626, Attack Precision=0.9235

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,342 duplicates (1,640,802 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349941, ATTACK=290861
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079952 -> 930756 (ratio<=4.0), total=1,163,445
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,861,512 (BENIGN=930,756, ATTACK=930,756)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,861,512, Test samples: 328,161
  Test: 328,161 (Normal=269,989, Attack=58,172)
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 50% : Attack 50%  (n=116,344, N=58,172, A=58,172)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485   <--
0.20       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.25       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.30       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.35       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.40       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.45       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.50       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.55       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.60       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.65       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.70       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.75       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
0.80       0.9552   0.9555   0.9478   0.9621   0.9626   0.9485  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9552, F1=0.9555, Normal Recall=0.9478, Normal Precision=0.9621, Attack Recall=0.9626, Attack Precision=0.9485

```

