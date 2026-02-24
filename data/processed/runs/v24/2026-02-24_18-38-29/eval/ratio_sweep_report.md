# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-24 18:42:19 |

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
| Original (TFLite) | 0.0000 | 0.1000 | 0.2000 | 0.3000 | 0.4000 | 0.5000 | 0.6000 | 0.7000 | 0.8000 | 0.9000 | 1.0000 |
| noQAT+PTQ | 0.0000 | 0.1000 | 0.2000 | 0.3000 | 0.4000 | 0.5000 | 0.6000 | 0.7000 | 0.8000 | 0.9000 | 1.0000 |
| saved_model_traditional_qat | 0.9690 | 0.9660 | 0.9621 | 0.9588 | 0.9548 | 0.9510 | 0.9474 | 0.9433 | 0.9397 | 0.9360 | 0.9325 |
| QAT+PTQ | 0.9465 | 0.9472 | 0.9473 | 0.9483 | 0.9485 | 0.9484 | 0.9492 | 0.9494 | 0.9493 | 0.9497 | 0.9500 |
| Compressed (QAT) | 0.9635 | 0.9620 | 0.9601 | 0.9589 | 0.9565 | 0.9546 | 0.9532 | 0.9508 | 0.9491 | 0.9469 | 0.9454 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1818 | 0.3333 | 0.4615 | 0.5714 | 0.6667 | 0.7500 | 0.8235 | 0.8889 | 0.9474 | 1.0000 |
| noQAT+PTQ | 0.0000 | 0.1818 | 0.3333 | 0.4615 | 0.5714 | 0.6667 | 0.7500 | 0.8235 | 0.8889 | 0.9474 | 1.0000 |
| saved_model_traditional_qat | 0.0000 | 0.8458 | 0.9078 | 0.9315 | 0.9428 | 0.9501 | 0.9551 | 0.9584 | 0.9612 | 0.9633 | 0.9651 |
| QAT+PTQ | 0.0000 | 0.7825 | 0.8781 | 0.9168 | 0.9365 | 0.9485 | 0.9574 | 0.9634 | 0.9677 | 0.9714 | 0.9743 |
| Compressed (QAT) | 0.0000 | 0.8328 | 0.9046 | 0.9325 | 0.9456 | 0.9541 | 0.9604 | 0.9641 | 0.9674 | 0.9697 | 0.9720 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| saved_model_traditional_qat | 0.9690 | 0.9695 | 0.9695 | 0.9701 | 0.9696 | 0.9696 | 0.9697 | 0.9686 | 0.9686 | 0.9677 | 0.0000 |
| QAT+PTQ | 0.9465 | 0.9470 | 0.9466 | 0.9475 | 0.9475 | 0.9468 | 0.9482 | 0.9481 | 0.9465 | 0.9476 | 0.0000 |
| Compressed (QAT) | 0.9635 | 0.9639 | 0.9638 | 0.9647 | 0.9638 | 0.9637 | 0.9647 | 0.9632 | 0.9636 | 0.9601 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 90 | 10 | 299,940 | 0.1000 | 0.1000 | 1.0000 | 0.1818 | 0.0000 | 1.0000 |
| 80 | 20 | 291,350 | 0.2000 | 0.2000 | 1.0000 | 0.3333 | 0.0000 | 1.0000 |
| 70 | 30 | 194,230 | 0.3000 | 0.3000 | 1.0000 | 0.4615 | 0.0000 | 0.0000 |
| 60 | 40 | 145,675 | 0.4000 | 0.4000 | 1.0000 | 0.5714 | 0.0000 | 1.0000 |
| 50 | 50 | 116,540 | 0.5000 | 0.5000 | 1.0000 | 0.6667 | 0.0000 | 1.0000 |
| 40 | 60 | 97,115 | 0.6000 | 0.6000 | 1.0000 | 0.7500 | 0.0000 | 0.0000 |
| 30 | 70 | 83,240 | 0.7000 | 0.7000 | 1.0000 | 0.8235 | 0.0000 | 0.0000 |
| 20 | 80 | 72,835 | 0.8000 | 0.8000 | 1.0000 | 0.8889 | 0.0000 | 0.0000 |
| 10 | 90 | 64,740 | 0.9000 | 0.9000 | 1.0000 | 0.9474 | 0.0000 | 0.0000 |
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
| 100 | 0 | 100,000 | 0.9690 | 0.0000 | 0.0000 | 0.0000 | 0.9690 | 1.0000 |
| 90 | 10 | 299,940 | 0.9660 | 0.7731 | 0.9336 | 0.8458 | 0.9695 | 0.9925 |
| 80 | 20 | 291,350 | 0.9621 | 0.8843 | 0.9325 | 0.9078 | 0.9695 | 0.9829 |
| 70 | 30 | 194,230 | 0.9588 | 0.9305 | 0.9325 | 0.9315 | 0.9701 | 0.9711 |
| 60 | 40 | 145,675 | 0.9548 | 0.9533 | 0.9325 | 0.9428 | 0.9696 | 0.9557 |
| 50 | 50 | 116,540 | 0.9510 | 0.9684 | 0.9325 | 0.9501 | 0.9696 | 0.9349 |
| 40 | 60 | 97,115 | 0.9474 | 0.9788 | 0.9325 | 0.9551 | 0.9697 | 0.9055 |
| 30 | 70 | 83,240 | 0.9433 | 0.9858 | 0.9325 | 0.9584 | 0.9686 | 0.8602 |
| 20 | 80 | 72,835 | 0.9397 | 0.9916 | 0.9325 | 0.9612 | 0.9686 | 0.7821 |
| 10 | 90 | 64,740 | 0.9360 | 0.9962 | 0.9325 | 0.9633 | 0.9677 | 0.6144 |
| 0 | 100 | 58,270 | 0.9325 | 1.0000 | 0.9325 | 0.9651 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9465 | 0.0000 | 0.0000 | 0.0000 | 0.9465 | 1.0000 |
| 90 | 10 | 299,940 | 0.9472 | 0.6655 | 0.9495 | 0.7825 | 0.9470 | 0.9941 |
| 80 | 20 | 291,350 | 0.9473 | 0.8164 | 0.9500 | 0.8781 | 0.9466 | 0.9870 |
| 70 | 30 | 194,230 | 0.9483 | 0.8858 | 0.9500 | 0.9168 | 0.9475 | 0.9779 |
| 60 | 40 | 145,675 | 0.9485 | 0.9234 | 0.9500 | 0.9365 | 0.9475 | 0.9660 |
| 50 | 50 | 116,540 | 0.9484 | 0.9470 | 0.9500 | 0.9485 | 0.9468 | 0.9498 |
| 40 | 60 | 97,115 | 0.9492 | 0.9649 | 0.9500 | 0.9574 | 0.9482 | 0.9266 |
| 30 | 70 | 83,240 | 0.9494 | 0.9771 | 0.9500 | 0.9634 | 0.9481 | 0.8903 |
| 20 | 80 | 72,835 | 0.9493 | 0.9861 | 0.9500 | 0.9677 | 0.9465 | 0.8254 |
| 10 | 90 | 64,740 | 0.9497 | 0.9939 | 0.9500 | 0.9714 | 0.9476 | 0.6779 |
| 0 | 100 | 58,270 | 0.9500 | 1.0000 | 0.9500 | 0.9743 | 0.0000 | 0.0000 |

### Compressed (QAT)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9635 | 0.0000 | 0.0000 | 0.0000 | 0.9635 | 1.0000 |
| 90 | 10 | 299,940 | 0.9620 | 0.7442 | 0.9453 | 0.8328 | 0.9639 | 0.9937 |
| 80 | 20 | 291,350 | 0.9601 | 0.8671 | 0.9454 | 0.9046 | 0.9638 | 0.9860 |
| 70 | 30 | 194,230 | 0.9589 | 0.9199 | 0.9454 | 0.9325 | 0.9647 | 0.9763 |
| 60 | 40 | 145,675 | 0.9565 | 0.9458 | 0.9454 | 0.9456 | 0.9638 | 0.9636 |
| 50 | 50 | 116,540 | 0.9546 | 0.9630 | 0.9454 | 0.9541 | 0.9637 | 0.9464 |
| 40 | 60 | 97,115 | 0.9532 | 0.9757 | 0.9454 | 0.9604 | 0.9647 | 0.9218 |
| 30 | 70 | 83,240 | 0.9508 | 0.9836 | 0.9454 | 0.9641 | 0.9632 | 0.8833 |
| 20 | 80 | 72,835 | 0.9491 | 0.9905 | 0.9455 | 0.9674 | 0.9636 | 0.8154 |
| 10 | 90 | 64,740 | 0.9469 | 0.9953 | 0.9454 | 0.9697 | 0.9601 | 0.6616 |
| 0 | 100 | 58,270 | 0.9454 | 1.0000 | 0.9454 | 0.9720 | 0.0000 | 0.0000 |


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
0.15       0.1000   0.1818   0.0000   1.0000   1.0000   0.1000   <--
0.20       0.1000   0.1818   0.0000   1.0000   1.0000   0.1000  
0.25       0.1000   0.1818   0.0000   1.0000   1.0000   0.1000  
0.30       0.1000   0.1818   0.0000   1.0000   1.0000   0.1000  
0.35       0.1000   0.1818   0.0000   1.0000   1.0000   0.1000  
0.40       0.1000   0.1818   0.0000   0.6667   1.0000   0.1000  
0.45       0.1002   0.1816   0.0005   0.6774   0.9980   0.0999  
0.50       0.1077   0.1221   0.0507   0.5458   0.6207   0.0677  
0.55       0.6706   0.1424   0.7147   0.8985   0.2736   0.0963  
0.60       0.8580   0.0916   0.9454   0.9016   0.0716   0.1271  
0.65       0.8890   0.0156   0.9868   0.8996   0.0088   0.0689  
0.70       0.8949   0.0034   0.9941   0.8996   0.0018   0.0322  
0.75       0.8984   0.0010   0.9982   0.8999   0.0005   0.0297  
0.80       0.8991   0.0001   0.9990   0.8999   0.0000   0.0035  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.1000, F1=0.1818, Normal Recall=0.0000, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1000

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
0.15       0.2000   0.3333   0.0000   1.0000   1.0000   0.2000   <--
0.20       0.2000   0.3333   0.0000   1.0000   1.0000   0.2000  
0.25       0.2000   0.3333   0.0000   1.0000   1.0000   0.2000  
0.30       0.2000   0.3333   0.0000   1.0000   1.0000   0.2000  
0.35       0.2000   0.3333   0.0000   1.0000   1.0000   0.2000  
0.40       0.2000   0.3333   0.0000   0.5000   1.0000   0.2000  
0.45       0.2000   0.3329   0.0005   0.5112   0.9981   0.1998  
0.50       0.1642   0.2283   0.0507   0.3469   0.6182   0.1400  
0.55       0.6260   0.2242   0.7149   0.7967   0.2703   0.1916  
0.60       0.7706   0.1116   0.9452   0.8029   0.0720   0.2474  
0.65       0.7911   0.0163   0.9867   0.7992   0.0086   0.1397  
0.70       0.7956   0.0035   0.9940   0.7993   0.0018   0.0697  
0.75       0.7987   0.0013   0.9982   0.7998   0.0006   0.0799  
0.80       0.7992   0.0002   0.9989   0.7998   0.0001   0.0234  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2000, F1=0.3333, Normal Recall=0.0000, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.2000

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
0.15       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000   <--
0.20       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.25       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.30       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.35       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.40       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.45       0.2998   0.4610   0.0005   0.3699   0.9981   0.2997  
0.50       0.2211   0.3226   0.0509   0.2372   0.6182   0.2182  
0.55       0.5804   0.2787   0.7133   0.6952   0.2703   0.2878  
0.60       0.6831   0.1200   0.9450   0.7038   0.0720   0.3593  
0.65       0.6932   0.0166   0.9866   0.6990   0.0086   0.2167  
0.70       0.6964   0.0035   0.9941   0.6991   0.0018   0.1140  
0.75       0.6990   0.0013   0.9983   0.6998   0.0006   0.1355  
0.80       0.6994   0.0002   0.9990   0.6998   0.0001   0.0438  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3000, F1=0.4615, Normal Recall=0.0000, Normal Precision=0.0000, Attack Recall=1.0000, Attack Precision=0.3000

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
0.15       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000   <--
0.20       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.25       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.30       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.35       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.40       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.45       0.3996   0.5708   0.0005   0.2968   0.9981   0.3997  
0.50       0.2774   0.4063   0.0502   0.1648   0.6182   0.3026  
0.55       0.5355   0.3176   0.7123   0.5942   0.2703   0.3851  
0.60       0.5959   0.1248   0.9451   0.6044   0.0720   0.4665  
0.65       0.5953   0.0168   0.9864   0.5988   0.0086   0.2984  
0.70       0.5971   0.0035   0.9940   0.5990   0.0018   0.1664  
0.75       0.5993   0.0013   0.9984   0.5998   0.0006   0.2067  
0.80       0.5995   0.0002   0.9992   0.5998   0.0001   0.0769  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4000, F1=0.5714, Normal Recall=0.0000, Normal Precision=0.0000, Attack Recall=1.0000, Attack Precision=0.4000

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
0.15       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000   <--
0.20       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.25       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.30       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.35       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.40       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.45       0.4993   0.6660   0.0006   0.2324   0.9981   0.4997  
0.50       0.3338   0.4813   0.0494   0.1145   0.6182   0.3940  
0.55       0.4916   0.3471   0.7130   0.4942   0.2703   0.4850  
0.60       0.5085   0.1278   0.9449   0.5045   0.0720   0.5668  
0.65       0.4975   0.0169   0.9864   0.4988   0.0086   0.3892  
0.70       0.4979   0.0035   0.9940   0.4989   0.0018   0.2296  
0.75       0.4996   0.0013   0.9985   0.4998   0.0006   0.3008  
0.80       0.4996   0.0002   0.9992   0.4998   0.0001   0.1111  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5000, F1=0.6667, Normal Recall=0.0000, Normal Precision=0.0000, Attack Recall=1.0000, Attack Precision=0.5000

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
0.40       0.1049   0.1826   0.0055   0.9986   0.9999   0.1005  
0.45       0.1304   0.1869   0.0339   0.9976   0.9993   0.1031  
0.50       0.9468   0.6396   0.9996   0.9446   0.4720   0.9917   <--
0.55       0.9441   0.6138   0.9997   0.9418   0.4439   0.9943  
0.60       0.9438   0.6099   0.9998   0.9414   0.4394   0.9967  
0.65       0.9432   0.6038   0.9999   0.9407   0.4328   0.9985  
0.70       0.9417   0.5885   0.9999   0.9392   0.4172   0.9985  
0.75       0.9393   0.5646   0.9999   0.9369   0.3935   0.9986  
0.80       0.9386   0.5572   0.9999   0.9362   0.3863   0.9987  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.9468, F1=0.6396, Normal Recall=0.9996, Normal Precision=0.9446, Attack Recall=0.4720, Attack Precision=0.9917

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
0.40       0.2044   0.3345   0.0055   0.9969   0.9999   0.2009  
0.45       0.2271   0.3409   0.0341   0.9946   0.9993   0.2055  
0.50       0.8940   0.6402   0.9996   0.8833   0.4716   0.9964   <--
0.55       0.8887   0.6150   0.9997   0.8780   0.4445   0.9975  
0.60       0.8879   0.6110   0.9998   0.8772   0.4402   0.9985  
0.65       0.8865   0.6042   0.9999   0.8758   0.4330   0.9993  
0.70       0.8835   0.5893   0.9999   0.8730   0.4179   0.9993  
0.75       0.8789   0.5659   0.9999   0.8686   0.3947   0.9994  
0.80       0.8773   0.5578   0.9999   0.8671   0.3868   0.9994  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.8940, F1=0.6402, Normal Recall=0.9996, Normal Precision=0.8833, Attack Recall=0.4716, Attack Precision=0.9964

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
0.40       0.3038   0.4629   0.0055   0.9947   0.9999   0.3011  
0.45       0.3237   0.4699   0.0342   0.9908   0.9993   0.3072  
0.50       0.8412   0.6405   0.9996   0.8153   0.4716   0.9978   <--
0.55       0.8331   0.6151   0.9997   0.8077   0.4445   0.9984  
0.60       0.8319   0.6111   0.9998   0.8065   0.4402   0.9989  
0.65       0.8298   0.6042   0.9999   0.8045   0.4330   0.9995  
0.70       0.8253   0.5894   0.9999   0.8003   0.4179   0.9995  
0.75       0.8183   0.5659   0.9999   0.7940   0.3947   0.9996  
0.80       0.8160   0.5578   0.9999   0.7919   0.3868   0.9996  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.8412, F1=0.6405, Normal Recall=0.9996, Normal Precision=0.8153, Attack Recall=0.4716, Attack Precision=0.9978

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
0.40       0.4032   0.5727   0.0054   0.9916   0.9999   0.4013  
0.45       0.4201   0.5796   0.0341   0.9858   0.9993   0.4082  
0.50       0.7884   0.6406   0.9996   0.7394   0.4716   0.9988   <--
0.55       0.7776   0.6153   0.9997   0.7297   0.4445   0.9991  
0.60       0.7759   0.6111   0.9998   0.7282   0.4402   0.9993  
0.65       0.7731   0.6043   0.9999   0.7257   0.4330   0.9997  
0.70       0.7671   0.5894   0.9999   0.7204   0.4179   0.9998  
0.75       0.7578   0.5659   1.0000   0.7125   0.3947   0.9998  
0.80       0.7547   0.5578   1.0000   0.7098   0.3868   0.9999  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.7884, F1=0.6406, Normal Recall=0.9996, Normal Precision=0.7394, Attack Recall=0.4716, Attack Precision=0.9988

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
0.40       0.5028   0.6679   0.0056   0.9878   0.9999   0.5014  
0.45       0.5169   0.6741   0.0346   0.9791   0.9993   0.5086   <--
0.50       0.7356   0.6407   0.9996   0.6542   0.4716   0.9992  
0.55       0.7221   0.6153   0.9997   0.6428   0.4445   0.9994  
0.60       0.7200   0.6112   0.9998   0.6410   0.4402   0.9996  
0.65       0.7164   0.6043   0.9999   0.6381   0.4330   0.9998  
0.70       0.7089   0.5894   0.9999   0.6320   0.4179   0.9998  
0.75       0.6973   0.5659   1.0000   0.6229   0.3947   0.9999  
0.80       0.6934   0.5579   1.0000   0.6199   0.3868   1.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.5169, F1=0.6741, Normal Recall=0.0346, Normal Precision=0.9791, Attack Recall=0.9993, Attack Precision=0.5086

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
0.15       0.9400   0.7632   0.9371   0.9960   0.9665   0.6306  
0.20       0.9475   0.7843   0.9468   0.9946   0.9538   0.6659  
0.25       0.9589   0.8220   0.9598   0.9942   0.9500   0.7244  
0.30       0.9661   0.8463   0.9695   0.9926   0.9347   0.7733  
0.35       0.9668   0.8481   0.9711   0.9918   0.9276   0.7811  
0.40       0.9693   0.8571   0.9746   0.9911   0.9214   0.8012   <--
0.45       0.9691   0.8553   0.9754   0.9901   0.9124   0.8050  
0.50       0.9563   0.7791   0.9770   0.9745   0.7700   0.7884  
0.55       0.9566   0.7789   0.9780   0.9739   0.7644   0.7941  
0.60       0.9604   0.7866   0.9861   0.9704   0.7293   0.8537  
0.65       0.9649   0.7981   0.9950   0.9670   0.6939   0.9390  
0.70       0.9645   0.7942   0.9957   0.9659   0.6839   0.9468  
0.75       0.9646   0.7924   0.9967   0.9651   0.6757   0.9578  
0.80       0.9627   0.7751   0.9984   0.9617   0.6420   0.9779  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9693, F1=0.8571, Normal Recall=0.9746, Normal Precision=0.9911, Attack Recall=0.9214, Attack Precision=0.8012

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
0.15       0.9426   0.8703   0.9373   0.9904   0.9638   0.7934  
0.20       0.9479   0.8796   0.9470   0.9874   0.9515   0.8178  
0.25       0.9576   0.8995   0.9601   0.9866   0.9479   0.8558  
0.30       0.9622   0.9080   0.9696   0.9829   0.9325   0.8847  
0.35       0.9622   0.9074   0.9712   0.9813   0.9261   0.8893  
0.40       0.9638   0.9103   0.9747   0.9799   0.9200   0.9009   <--
0.45       0.9626   0.9069   0.9755   0.9777   0.9109   0.9029  
0.50       0.9349   0.8249   0.9771   0.9435   0.7662   0.8933  
0.55       0.9345   0.8228   0.9780   0.9423   0.7603   0.8964  
0.60       0.9341   0.8152   0.9861   0.9351   0.7263   0.9287  
0.65       0.9341   0.8073   0.9950   0.9278   0.6905   0.9718  
0.70       0.9327   0.8020   0.9957   0.9258   0.6809   0.9754  
0.75       0.9319   0.7979   0.9967   0.9241   0.6725   0.9807  
0.80       0.9263   0.7760   0.9984   0.9169   0.6381   0.9899  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9638, F1=0.9103, Normal Recall=0.9747, Normal Precision=0.9799, Attack Recall=0.9200, Attack Precision=0.9009

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
0.15       0.9453   0.9136   0.9374   0.9837   0.9638   0.8684  
0.20       0.9483   0.9170   0.9469   0.9785   0.9515   0.8849  
0.25       0.9563   0.9286   0.9599   0.9773   0.9479   0.9101  
0.30       0.9582   0.9305   0.9692   0.9710   0.9325   0.9286   <--
0.35       0.9573   0.9286   0.9707   0.9684   0.9261   0.9312  
0.40       0.9580   0.9293   0.9743   0.9660   0.9200   0.9387  
0.45       0.9559   0.9253   0.9751   0.9623   0.9109   0.9401  
0.50       0.9136   0.8419   0.9769   0.9070   0.7662   0.9342  
0.55       0.9126   0.8392   0.9778   0.9049   0.7603   0.9362  
0.60       0.9082   0.8259   0.9861   0.8937   0.7264   0.9572  
0.65       0.9036   0.8112   0.9949   0.8824   0.6905   0.9832  
0.70       0.9012   0.8053   0.9956   0.8792   0.6809   0.9852  
0.75       0.8994   0.8005   0.9966   0.8766   0.6726   0.9884  
0.80       0.8903   0.7773   0.9984   0.8656   0.6381   0.9942  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9582, F1=0.9305, Normal Recall=0.9692, Normal Precision=0.9710, Attack Recall=0.9325, Attack Precision=0.9286

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
0.15       0.9480   0.9368   0.9375   0.9749   0.9638   0.9113  
0.20       0.9488   0.9370   0.9470   0.9670   0.9515   0.9229  
0.25       0.9550   0.9440   0.9598   0.9651   0.9479   0.9402   <--
0.30       0.9546   0.9426   0.9693   0.9556   0.9325   0.9529  
0.35       0.9529   0.9402   0.9708   0.9517   0.9261   0.9548  
0.40       0.9526   0.9394   0.9743   0.9481   0.9200   0.9597  
0.45       0.9494   0.9351   0.9751   0.9426   0.9109   0.9607  
0.50       0.8927   0.8510   0.9770   0.8624   0.7662   0.9569  
0.55       0.8909   0.8479   0.9780   0.8596   0.7603   0.9583  
0.60       0.8822   0.8315   0.9861   0.8439   0.7263   0.9722  
0.65       0.8731   0.8132   0.9949   0.8282   0.6905   0.9891  
0.70       0.8697   0.8070   0.9956   0.8239   0.6809   0.9904  
0.75       0.8670   0.8018   0.9966   0.8203   0.6725   0.9925  
0.80       0.8543   0.7779   0.9984   0.8054   0.6381   0.9961  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.9550, F1=0.9440, Normal Recall=0.9598, Normal Precision=0.9651, Attack Recall=0.9479, Attack Precision=0.9402

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
0.15       0.9509   0.9515   0.9379   0.9628   0.9638   0.9395  
0.20       0.9495   0.9496   0.9475   0.9513   0.9515   0.9477  
0.25       0.9543   0.9540   0.9606   0.9485   0.9479   0.9601   <--
0.30       0.9511   0.9502   0.9698   0.9349   0.9325   0.9686  
0.35       0.9486   0.9475   0.9711   0.9293   0.9261   0.9698  
0.40       0.9473   0.9458   0.9746   0.9241   0.9200   0.9731  
0.45       0.9432   0.9413   0.9755   0.9163   0.9109   0.9738  
0.50       0.8717   0.8565   0.9772   0.8069   0.7662   0.9711  
0.55       0.8692   0.8533   0.9782   0.8032   0.7603   0.9721  
0.60       0.8563   0.8349   0.9863   0.7828   0.7263   0.9815  
0.65       0.8426   0.8144   0.9947   0.7627   0.6905   0.9924  
0.70       0.8382   0.8080   0.9954   0.7573   0.6809   0.9933  
0.75       0.8345   0.8025   0.9965   0.7527   0.6725   0.9948  
0.80       0.8182   0.7783   0.9984   0.7340   0.6381   0.9974  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.9543, F1=0.9540, Normal Recall=0.9606, Normal Precision=0.9485, Attack Recall=0.9479, Attack Precision=0.9601

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
0.15       0.9372   0.7519   0.9356   0.9943   0.9513   0.6216  
0.20       0.9427   0.7683   0.9418   0.9942   0.9503   0.6448  
0.25       0.9427   0.7683   0.9418   0.9942   0.9503   0.6448  
0.30       0.9473   0.7828   0.9470   0.9942   0.9501   0.6656  
0.35       0.9525   0.7998   0.9529   0.9941   0.9489   0.6912  
0.40       0.9525   0.7998   0.9529   0.9941   0.9489   0.6912  
0.45       0.9621   0.8318   0.9647   0.9929   0.9380   0.7473  
0.50       0.9621   0.8318   0.9647   0.9929   0.9380   0.7473  
0.55       0.9636   0.8360   0.9675   0.9918   0.9281   0.7606  
0.60       0.9688   0.8541   0.9750   0.9902   0.9134   0.8020   <--
0.65       0.9688   0.8541   0.9750   0.9902   0.9134   0.8020  
0.70       0.9678   0.8477   0.9759   0.9882   0.8950   0.8052  
0.75       0.9673   0.8325   0.9844   0.9793   0.8132   0.8528  
0.80       0.9673   0.8325   0.9844   0.9793   0.8132   0.8528  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.6
  At threshold 0.6: Accuracy=0.9688, F1=0.8541, Normal Recall=0.9750, Normal Precision=0.9902, Attack Recall=0.9134, Attack Precision=0.8020

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
0.15       0.9389   0.8617   0.9358   0.9872   0.9513   0.7875  
0.20       0.9436   0.8708   0.9420   0.9870   0.9502   0.8036  
0.25       0.9436   0.8708   0.9420   0.9870   0.9502   0.8036  
0.30       0.9476   0.8788   0.9470   0.9870   0.9500   0.8175  
0.35       0.9520   0.8877   0.9528   0.9868   0.9489   0.8340  
0.40       0.9520   0.8877   0.9528   0.9868   0.9489   0.8340  
0.45       0.9594   0.9024   0.9647   0.9842   0.9380   0.8693  
0.50       0.9594   0.9024   0.9647   0.9842   0.9380   0.8693  
0.55       0.9597   0.9021   0.9676   0.9818   0.9283   0.8774  
0.60       0.9627   0.9075   0.9750   0.9783   0.9137   0.9014   <--
0.65       0.9627   0.9075   0.9750   0.9783   0.9137   0.9014  
0.70       0.9597   0.8988   0.9759   0.9738   0.8948   0.9029  
0.75       0.9499   0.8663   0.9843   0.9545   0.8122   0.9282  
0.80       0.9499   0.8663   0.9843   0.9545   0.8122   0.9282  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.6
  At threshold 0.6: Accuracy=0.9627, F1=0.9075, Normal Recall=0.9750, Normal Precision=0.9783, Attack Recall=0.9137, Attack Precision=0.9014

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
0.15       0.9406   0.9057   0.9360   0.9782   0.9513   0.8643  
0.20       0.9446   0.9114   0.9422   0.9778   0.9502   0.8757  
0.25       0.9446   0.9114   0.9422   0.9778   0.9502   0.8757  
0.30       0.9480   0.9164   0.9472   0.9779   0.9500   0.8852  
0.35       0.9517   0.9218   0.9529   0.9775   0.9489   0.8962  
0.40       0.9517   0.9218   0.9529   0.9775   0.9489   0.8962  
0.45       0.9566   0.9284   0.9645   0.9732   0.9380   0.9189   <--
0.50       0.9566   0.9284   0.9645   0.9732   0.9380   0.9189  
0.55       0.9556   0.9262   0.9673   0.9692   0.9283   0.9241  
0.60       0.9564   0.9263   0.9746   0.9634   0.9137   0.9392  
0.65       0.9564   0.9263   0.9746   0.9634   0.9137   0.9392  
0.70       0.9513   0.9169   0.9755   0.9558   0.8948   0.9401  
0.75       0.9325   0.8783   0.9841   0.9244   0.8121   0.9563  
0.80       0.9325   0.8783   0.9841   0.9244   0.8121   0.9563  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9566, F1=0.9284, Normal Recall=0.9645, Normal Precision=0.9732, Attack Recall=0.9380, Attack Precision=0.9189

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
0.15       0.9419   0.9290   0.9355   0.9665   0.9513   0.9077  
0.20       0.9451   0.9327   0.9418   0.9659   0.9502   0.9158  
0.25       0.9451   0.9327   0.9418   0.9659   0.9502   0.9158  
0.30       0.9481   0.9361   0.9469   0.9660   0.9500   0.9226  
0.35       0.9511   0.9395   0.9526   0.9655   0.9489   0.9303  
0.40       0.9511   0.9395   0.9526   0.9655   0.9489   0.9303  
0.45       0.9537   0.9419   0.9641   0.9589   0.9380   0.9457   <--
0.50       0.9537   0.9419   0.9641   0.9589   0.9380   0.9457  
0.55       0.9515   0.9388   0.9671   0.9529   0.9283   0.9495  
0.60       0.9502   0.9362   0.9745   0.9442   0.9137   0.9598  
0.65       0.9502   0.9362   0.9745   0.9442   0.9137   0.9598  
0.70       0.9432   0.9264   0.9754   0.9329   0.8948   0.9604  
0.75       0.9151   0.8844   0.9838   0.8871   0.8122   0.9709  
0.80       0.9151   0.8844   0.9838   0.8871   0.8122   0.9709  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9537, F1=0.9419, Normal Recall=0.9641, Normal Precision=0.9589, Attack Recall=0.9380, Attack Precision=0.9457

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
0.15       0.9435   0.9440   0.9358   0.9505   0.9513   0.9368  
0.20       0.9461   0.9463   0.9420   0.9498   0.9502   0.9425  
0.25       0.9461   0.9463   0.9420   0.9498   0.9502   0.9425  
0.30       0.9486   0.9487   0.9473   0.9498   0.9500   0.9474  
0.35       0.9509   0.9508   0.9529   0.9491   0.9489   0.9527   <--
0.40       0.9509   0.9508   0.9529   0.9491   0.9489   0.9527  
0.45       0.9511   0.9505   0.9642   0.9396   0.9380   0.9633  
0.50       0.9511   0.9505   0.9642   0.9396   0.9380   0.9633  
0.55       0.9478   0.9468   0.9673   0.9310   0.9283   0.9660  
0.60       0.9443   0.9425   0.9748   0.9187   0.9137   0.9732  
0.65       0.9443   0.9425   0.9748   0.9187   0.9137   0.9732  
0.70       0.9353   0.9325   0.9758   0.9027   0.8948   0.9736  
0.75       0.8981   0.8885   0.9840   0.8397   0.8122   0.9807  
0.80       0.8981   0.8885   0.9840   0.8397   0.8122   0.9807  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9509, F1=0.9508, Normal Recall=0.9529, Normal Precision=0.9491, Attack Recall=0.9489, Attack Precision=0.9527

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
0.15       0.9281   0.7315   0.9223   0.9976   0.9801   0.5835  
0.20       0.9341   0.7480   0.9292   0.9974   0.9781   0.6055  
0.25       0.9517   0.7999   0.9501   0.9960   0.9658   0.6827  
0.30       0.9621   0.8333   0.9639   0.9939   0.9464   0.7444  
0.35       0.9659   0.8461   0.9690   0.9929   0.9381   0.7705  
0.40       0.9667   0.8491   0.9699   0.9929   0.9378   0.7758  
0.45       0.9674   0.8519   0.9707   0.9929   0.9376   0.7805  
0.50       0.9677   0.8531   0.9711   0.9929   0.9375   0.7826  
0.55       0.9685   0.8562   0.9720   0.9929   0.9371   0.7881  
0.60       0.9686   0.8563   0.9724   0.9926   0.9347   0.7901  
0.65       0.9693   0.8590   0.9732   0.9926   0.9344   0.7948  
0.70       0.9695   0.8589   0.9740   0.9920   0.9290   0.7986  
0.75       0.9752   0.8668   0.9938   0.9790   0.8078   0.9351   <--
0.80       0.9753   0.8654   0.9956   0.9775   0.7933   0.9520  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.75
  At threshold 0.75: Accuracy=0.9752, F1=0.8668, Normal Recall=0.9938, Normal Precision=0.9790, Attack Recall=0.8078, Attack Precision=0.9351

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
0.15       0.9339   0.8557   0.9225   0.9945   0.9796   0.7597  
0.20       0.9390   0.8650   0.9294   0.9940   0.9775   0.7758  
0.25       0.9532   0.8919   0.9502   0.9909   0.9651   0.8290  
0.30       0.9603   0.9051   0.9641   0.9860   0.9454   0.8680  
0.35       0.9627   0.9095   0.9690   0.9841   0.9372   0.8833  
0.40       0.9633   0.9109   0.9700   0.9840   0.9369   0.8863  
0.45       0.9640   0.9122   0.9707   0.9840   0.9368   0.8889  
0.50       0.9642   0.9128   0.9711   0.9840   0.9367   0.8901  
0.55       0.9649   0.9142   0.9720   0.9839   0.9362   0.8932  
0.60       0.9647   0.9138   0.9724   0.9833   0.9340   0.8944  
0.65       0.9653   0.9150   0.9732   0.9833   0.9338   0.8971   <--
0.70       0.9649   0.9137   0.9740   0.9820   0.9286   0.8993  
0.75       0.9564   0.8810   0.9938   0.9537   0.8070   0.9700  
0.80       0.9550   0.8757   0.9956   0.9505   0.7927   0.9782  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.65
  At threshold 0.65: Accuracy=0.9653, F1=0.9150, Normal Recall=0.9732, Normal Precision=0.9833, Attack Recall=0.9338, Attack Precision=0.8971

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
0.15       0.9398   0.9072   0.9228   0.9906   0.9796   0.8447  
0.20       0.9440   0.9129   0.9297   0.9897   0.9775   0.8563  
0.25       0.9547   0.9275   0.9503   0.9845   0.9651   0.8926  
0.30       0.9583   0.9315   0.9638   0.9763   0.9454   0.9179  
0.35       0.9592   0.9324   0.9686   0.9730   0.9372   0.9275  
0.40       0.9598   0.9333   0.9697   0.9729   0.9369   0.9298  
0.45       0.9604   0.9342   0.9706   0.9728   0.9368   0.9317  
0.50       0.9606   0.9345   0.9708   0.9728   0.9367   0.9323  
0.55       0.9610   0.9351   0.9717   0.9726   0.9362   0.9341   <--
0.60       0.9607   0.9344   0.9721   0.9717   0.9340   0.9348  
0.65       0.9611   0.9351   0.9729   0.9717   0.9338   0.9365  
0.70       0.9601   0.9332   0.9737   0.9695   0.9286   0.9380  
0.75       0.9377   0.8861   0.9938   0.9231   0.8069   0.9824  
0.80       0.9347   0.8793   0.9956   0.9181   0.7927   0.9871  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9610, F1=0.9351, Normal Recall=0.9717, Normal Precision=0.9726, Attack Recall=0.9362, Attack Precision=0.9341

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
0.15       0.9456   0.9351   0.9229   0.9855   0.9796   0.8944  
0.20       0.9488   0.9386   0.9297   0.9841   0.9775   0.9026  
0.25       0.9562   0.9463   0.9502   0.9761   0.9651   0.9282  
0.30       0.9565   0.9457   0.9639   0.9636   0.9454   0.9459  
0.35       0.9560   0.9446   0.9686   0.9586   0.9372   0.9521  
0.40       0.9565   0.9452   0.9696   0.9584   0.9369   0.9536  
0.45       0.9571   0.9459   0.9707   0.9584   0.9368   0.9551  
0.50       0.9572   0.9460   0.9710   0.9583   0.9367   0.9555  
0.55       0.9575   0.9463   0.9718   0.9581   0.9362   0.9567   <--
0.60       0.9569   0.9454   0.9721   0.9567   0.9340   0.9572  
0.65       0.9572   0.9458   0.9728   0.9566   0.9338   0.9581  
0.70       0.9556   0.9435   0.9735   0.9534   0.9286   0.9590  
0.75       0.9190   0.8885   0.9937   0.8853   0.8070   0.9884  
0.80       0.9143   0.8809   0.9954   0.8781   0.7927   0.9913  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9575, F1=0.9463, Normal Recall=0.9718, Normal Precision=0.9581, Attack Recall=0.9362, Attack Precision=0.9567

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
0.15       0.9512   0.9526   0.9229   0.9784   0.9796   0.9270  
0.20       0.9536   0.9547   0.9297   0.9764   0.9775   0.9329  
0.25       0.9577   0.9580   0.9503   0.9646   0.9651   0.9510   <--
0.30       0.9548   0.9544   0.9642   0.9464   0.9454   0.9635  
0.35       0.9531   0.9524   0.9690   0.9392   0.9372   0.9680  
0.40       0.9535   0.9527   0.9700   0.9389   0.9369   0.9690  
0.45       0.9539   0.9531   0.9711   0.9389   0.9368   0.9701  
0.50       0.9541   0.9532   0.9714   0.9388   0.9367   0.9704  
0.55       0.9542   0.9534   0.9722   0.9384   0.9362   0.9712  
0.60       0.9533   0.9524   0.9726   0.9364   0.9340   0.9715  
0.65       0.9535   0.9526   0.9732   0.9363   0.9338   0.9721  
0.70       0.9512   0.9501   0.9738   0.9317   0.9286   0.9726  
0.75       0.9004   0.8901   0.9938   0.8373   0.8070   0.9924  
0.80       0.8941   0.8821   0.9955   0.8276   0.7927   0.9943  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.9577, F1=0.9580, Normal Recall=0.9503, Normal Precision=0.9646, Attack Recall=0.9651, Attack Precision=0.9510

```

