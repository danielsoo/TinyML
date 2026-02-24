# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e2_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-15 13:27:20 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 60 |
| **Local epochs** | 2 |
| **Batch size** | 256 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8797 | 0.8920 | 0.9032 | 0.9153 | 0.9260 | 0.9368 | 0.9496 | 0.9613 | 0.9730 | 0.9838 | 0.9956 |
| QAT+Prune only | 0.9430 | 0.9161 | 0.8895 | 0.8634 | 0.8368 | 0.8104 | 0.7837 | 0.7570 | 0.7313 | 0.7038 | 0.6781 |
| QAT+PTQ | 0.9429 | 0.9164 | 0.8903 | 0.8647 | 0.8386 | 0.8127 | 0.7865 | 0.7602 | 0.7350 | 0.7081 | 0.6827 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9429 | 0.9164 | 0.8903 | 0.8647 | 0.8386 | 0.8127 | 0.7865 | 0.7602 | 0.7350 | 0.7081 | 0.6827 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6485 | 0.8044 | 0.8758 | 0.9150 | 0.9403 | 0.9595 | 0.9730 | 0.9833 | 0.9911 | 0.9978 |
| QAT+Prune only | 0.0000 | 0.6178 | 0.7105 | 0.7487 | 0.7687 | 0.7815 | 0.7900 | 0.7962 | 0.8015 | 0.8047 | 0.8081 |
| QAT+PTQ | 0.0000 | 0.6203 | 0.7134 | 0.7517 | 0.7718 | 0.7847 | 0.7932 | 0.7994 | 0.8048 | 0.8080 | 0.8114 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.6203 | 0.7134 | 0.7517 | 0.7718 | 0.7847 | 0.7932 | 0.7994 | 0.8048 | 0.8080 | 0.8114 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8797 | 0.8805 | 0.8801 | 0.8808 | 0.8795 | 0.8779 | 0.8806 | 0.8812 | 0.8826 | 0.8777 | 0.0000 |
| QAT+Prune only | 0.9430 | 0.9426 | 0.9423 | 0.9429 | 0.9426 | 0.9428 | 0.9423 | 0.9411 | 0.9441 | 0.9359 | 0.0000 |
| QAT+PTQ | 0.9429 | 0.9424 | 0.9422 | 0.9427 | 0.9425 | 0.9426 | 0.9421 | 0.9410 | 0.9441 | 0.9364 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9429 | 0.9424 | 0.9422 | 0.9427 | 0.9425 | 0.9426 | 0.9421 | 0.9410 | 0.9441 | 0.9364 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8797 | 0.0000 | 0.0000 | 0.0000 | 0.8797 | 1.0000 |
| 90 | 10 | 299,940 | 0.8920 | 0.4808 | 0.9958 | 0.6485 | 0.8805 | 0.9995 |
| 80 | 20 | 291,350 | 0.9032 | 0.6748 | 0.9956 | 0.8044 | 0.8801 | 0.9988 |
| 70 | 30 | 194,230 | 0.9153 | 0.7817 | 0.9956 | 0.8758 | 0.8808 | 0.9979 |
| 60 | 40 | 145,675 | 0.9260 | 0.8464 | 0.9956 | 0.9150 | 0.8795 | 0.9967 |
| 50 | 50 | 116,540 | 0.9368 | 0.8908 | 0.9956 | 0.9403 | 0.8779 | 0.9950 |
| 40 | 60 | 97,115 | 0.9496 | 0.9259 | 0.9956 | 0.9595 | 0.8806 | 0.9926 |
| 30 | 70 | 83,240 | 0.9613 | 0.9514 | 0.9956 | 0.9730 | 0.8812 | 0.9885 |
| 20 | 80 | 72,835 | 0.9730 | 0.9714 | 0.9956 | 0.9833 | 0.8826 | 0.9806 |
| 10 | 90 | 64,740 | 0.9838 | 0.9865 | 0.9956 | 0.9911 | 0.8777 | 0.9570 |
| 0 | 100 | 58,270 | 0.9956 | 1.0000 | 0.9956 | 0.9978 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9430 | 0.0000 | 0.0000 | 0.0000 | 0.9430 | 1.0000 |
| 90 | 10 | 299,940 | 0.9161 | 0.5674 | 0.6779 | 0.6178 | 0.9426 | 0.9634 |
| 80 | 20 | 291,350 | 0.8895 | 0.7461 | 0.6781 | 0.7105 | 0.9423 | 0.9213 |
| 70 | 30 | 194,230 | 0.8634 | 0.8357 | 0.6780 | 0.7487 | 0.9429 | 0.8723 |
| 60 | 40 | 145,675 | 0.8368 | 0.8874 | 0.6781 | 0.7687 | 0.9426 | 0.8145 |
| 50 | 50 | 116,540 | 0.8104 | 0.9222 | 0.6781 | 0.7815 | 0.9428 | 0.7454 |
| 40 | 60 | 97,115 | 0.7837 | 0.9463 | 0.6780 | 0.7900 | 0.9423 | 0.6612 |
| 30 | 70 | 83,240 | 0.7570 | 0.9641 | 0.6781 | 0.7962 | 0.9411 | 0.5561 |
| 20 | 80 | 72,835 | 0.7313 | 0.9798 | 0.6781 | 0.8015 | 0.9441 | 0.4230 |
| 10 | 90 | 64,740 | 0.7038 | 0.9896 | 0.6780 | 0.8047 | 0.9359 | 0.2441 |
| 0 | 100 | 58,270 | 0.6781 | 1.0000 | 0.6781 | 0.8081 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9429 | 0.0000 | 0.0000 | 0.0000 | 0.9429 | 1.0000 |
| 90 | 10 | 299,940 | 0.9164 | 0.5684 | 0.6826 | 0.6203 | 0.9424 | 0.9639 |
| 80 | 20 | 291,350 | 0.8903 | 0.7469 | 0.6827 | 0.7134 | 0.9422 | 0.9223 |
| 70 | 30 | 194,230 | 0.8647 | 0.8362 | 0.6827 | 0.7517 | 0.9427 | 0.8739 |
| 60 | 40 | 145,675 | 0.8386 | 0.8878 | 0.6827 | 0.7718 | 0.9425 | 0.8167 |
| 50 | 50 | 116,540 | 0.8127 | 0.9225 | 0.6827 | 0.7847 | 0.9426 | 0.7482 |
| 40 | 60 | 97,115 | 0.7865 | 0.9465 | 0.6827 | 0.7932 | 0.9421 | 0.6644 |
| 30 | 70 | 83,240 | 0.7602 | 0.9643 | 0.6827 | 0.7994 | 0.9410 | 0.5597 |
| 20 | 80 | 72,835 | 0.7350 | 0.9799 | 0.6827 | 0.8048 | 0.9441 | 0.4266 |
| 10 | 90 | 64,740 | 0.7081 | 0.9897 | 0.6827 | 0.8080 | 0.9364 | 0.2469 |
| 0 | 100 | 58,270 | 0.6827 | 1.0000 | 0.6827 | 0.8114 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9429 | 0.0000 | 0.0000 | 0.0000 | 0.9429 | 1.0000 |
| 90 | 10 | 299,940 | 0.9164 | 0.5684 | 0.6826 | 0.6203 | 0.9424 | 0.9639 |
| 80 | 20 | 291,350 | 0.8903 | 0.7469 | 0.6827 | 0.7134 | 0.9422 | 0.9223 |
| 70 | 30 | 194,230 | 0.8647 | 0.8362 | 0.6827 | 0.7517 | 0.9427 | 0.8739 |
| 60 | 40 | 145,675 | 0.8386 | 0.8878 | 0.6827 | 0.7718 | 0.9425 | 0.8167 |
| 50 | 50 | 116,540 | 0.8127 | 0.9225 | 0.6827 | 0.7847 | 0.9426 | 0.7482 |
| 40 | 60 | 97,115 | 0.7865 | 0.9465 | 0.6827 | 0.7932 | 0.9421 | 0.6644 |
| 30 | 70 | 83,240 | 0.7602 | 0.9643 | 0.6827 | 0.7994 | 0.9410 | 0.5597 |
| 20 | 80 | 72,835 | 0.7350 | 0.9799 | 0.6827 | 0.8048 | 0.9441 | 0.4266 |
| 10 | 90 | 64,740 | 0.7081 | 0.9897 | 0.6827 | 0.8080 | 0.9364 | 0.2469 |
| 0 | 100 | 58,270 | 0.6827 | 1.0000 | 0.6827 | 0.8114 | 0.0000 | 0.0000 |


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
0.15       0.8921   0.6486   0.8805   0.9995   0.9961   0.4808   <--
0.20       0.8921   0.6486   0.8805   0.9995   0.9961   0.4808  
0.25       0.8921   0.6486   0.8805   0.9995   0.9961   0.4808  
0.30       0.8921   0.6486   0.8805   0.9995   0.9961   0.4808  
0.35       0.8921   0.6486   0.8805   0.9995   0.9961   0.4808  
0.40       0.8921   0.6486   0.8805   0.9995   0.9961   0.4808  
0.45       0.8921   0.6486   0.8805   0.9995   0.9961   0.4808  
0.50       0.8921   0.6486   0.8805   0.9995   0.9961   0.4808  
0.55       0.8921   0.6486   0.8805   0.9995   0.9961   0.4808  
0.60       0.8921   0.6486   0.8805   0.9995   0.9961   0.4808  
0.65       0.8921   0.6486   0.8805   0.9995   0.9961   0.4808  
0.70       0.8921   0.6486   0.8805   0.9995   0.9961   0.4808  
0.75       0.8921   0.6486   0.8805   0.9995   0.9961   0.4808  
0.80       0.8921   0.6486   0.8805   0.9995   0.9961   0.4808  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8921, F1=0.6486, Normal Recall=0.8805, Normal Precision=0.9995, Attack Recall=0.9961, Attack Precision=0.4808

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
0.15       0.9036   0.8051   0.8806   0.9988   0.9956   0.6758   <--
0.20       0.9036   0.8051   0.8806   0.9988   0.9956   0.6758  
0.25       0.9036   0.8051   0.8806   0.9988   0.9956   0.6758  
0.30       0.9036   0.8051   0.8806   0.9988   0.9956   0.6758  
0.35       0.9036   0.8051   0.8806   0.9988   0.9956   0.6758  
0.40       0.9036   0.8051   0.8806   0.9988   0.9956   0.6758  
0.45       0.9036   0.8051   0.8806   0.9988   0.9956   0.6758  
0.50       0.9036   0.8051   0.8806   0.9988   0.9956   0.6758  
0.55       0.9036   0.8051   0.8806   0.9988   0.9956   0.6758  
0.60       0.9036   0.8051   0.8806   0.9988   0.9956   0.6758  
0.65       0.9036   0.8051   0.8806   0.9988   0.9956   0.6758  
0.70       0.9036   0.8051   0.8806   0.9988   0.9956   0.6758  
0.75       0.9036   0.8051   0.8806   0.9988   0.9956   0.6758  
0.80       0.9036   0.8051   0.8806   0.9988   0.9956   0.6758  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9036, F1=0.8051, Normal Recall=0.8806, Normal Precision=0.9988, Attack Recall=0.9956, Attack Precision=0.6758

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
0.15       0.9151   0.8756   0.8806   0.9979   0.9956   0.7813   <--
0.20       0.9151   0.8756   0.8806   0.9979   0.9956   0.7813  
0.25       0.9151   0.8756   0.8806   0.9979   0.9956   0.7813  
0.30       0.9151   0.8756   0.8806   0.9979   0.9956   0.7813  
0.35       0.9151   0.8756   0.8806   0.9979   0.9956   0.7813  
0.40       0.9151   0.8756   0.8806   0.9979   0.9956   0.7813  
0.45       0.9151   0.8756   0.8806   0.9979   0.9956   0.7813  
0.50       0.9151   0.8756   0.8806   0.9979   0.9956   0.7813  
0.55       0.9151   0.8756   0.8806   0.9979   0.9956   0.7813  
0.60       0.9151   0.8756   0.8806   0.9979   0.9956   0.7813  
0.65       0.9151   0.8756   0.8806   0.9979   0.9956   0.7813  
0.70       0.9151   0.8756   0.8806   0.9979   0.9956   0.7813  
0.75       0.9151   0.8756   0.8806   0.9979   0.9956   0.7813  
0.80       0.9151   0.8756   0.8806   0.9979   0.9956   0.7813  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9151, F1=0.8756, Normal Recall=0.8806, Normal Precision=0.9979, Attack Recall=0.9956, Attack Precision=0.7813

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
0.15       0.9262   0.9152   0.8799   0.9967   0.9956   0.8468   <--
0.20       0.9262   0.9152   0.8799   0.9967   0.9956   0.8468  
0.25       0.9262   0.9152   0.8799   0.9967   0.9956   0.8468  
0.30       0.9262   0.9152   0.8799   0.9967   0.9956   0.8468  
0.35       0.9262   0.9152   0.8799   0.9967   0.9956   0.8468  
0.40       0.9262   0.9152   0.8799   0.9967   0.9956   0.8468  
0.45       0.9262   0.9152   0.8799   0.9967   0.9956   0.8468  
0.50       0.9262   0.9152   0.8799   0.9967   0.9956   0.8468  
0.55       0.9262   0.9152   0.8799   0.9967   0.9956   0.8468  
0.60       0.9262   0.9152   0.8799   0.9967   0.9956   0.8468  
0.65       0.9262   0.9152   0.8799   0.9967   0.9956   0.8468  
0.70       0.9262   0.9152   0.8799   0.9967   0.9956   0.8468  
0.75       0.9262   0.9152   0.8799   0.9967   0.9956   0.8468  
0.80       0.9262   0.9152   0.8799   0.9967   0.9956   0.8468  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9262, F1=0.9152, Normal Recall=0.8799, Normal Precision=0.9967, Attack Recall=0.9956, Attack Precision=0.8468

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
0.15       0.9376   0.9410   0.8796   0.9950   0.9956   0.8921   <--
0.20       0.9376   0.9410   0.8796   0.9950   0.9956   0.8921  
0.25       0.9376   0.9410   0.8796   0.9950   0.9956   0.8921  
0.30       0.9376   0.9410   0.8796   0.9950   0.9956   0.8921  
0.35       0.9376   0.9410   0.8796   0.9950   0.9956   0.8921  
0.40       0.9376   0.9410   0.8796   0.9950   0.9956   0.8921  
0.45       0.9376   0.9410   0.8796   0.9950   0.9956   0.8921  
0.50       0.9376   0.9410   0.8796   0.9950   0.9956   0.8921  
0.55       0.9376   0.9410   0.8796   0.9950   0.9956   0.8921  
0.60       0.9376   0.9410   0.8796   0.9950   0.9956   0.8921  
0.65       0.9376   0.9410   0.8796   0.9950   0.9956   0.8921  
0.70       0.9376   0.9410   0.8796   0.9950   0.9956   0.8921  
0.75       0.9376   0.9410   0.8796   0.9950   0.9956   0.8921  
0.80       0.9376   0.9410   0.8796   0.9950   0.9956   0.8921  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9376, F1=0.9410, Normal Recall=0.8796, Normal Precision=0.9950, Attack Recall=0.9956, Attack Precision=0.8921

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
0.15       0.9161   0.6177   0.9426   0.9634   0.6779   0.5674   <--
0.20       0.9161   0.6177   0.9426   0.9634   0.6779   0.5674  
0.25       0.9161   0.6177   0.9426   0.9634   0.6779   0.5674  
0.30       0.9161   0.6177   0.9426   0.9634   0.6779   0.5674  
0.35       0.9161   0.6177   0.9426   0.9634   0.6779   0.5674  
0.40       0.9161   0.6177   0.9426   0.9634   0.6779   0.5674  
0.45       0.9161   0.6177   0.9426   0.9634   0.6779   0.5674  
0.50       0.9161   0.6177   0.9426   0.9634   0.6779   0.5674  
0.55       0.9161   0.6177   0.9426   0.9634   0.6779   0.5674  
0.60       0.9161   0.6177   0.9426   0.9634   0.6779   0.5674  
0.65       0.9161   0.6177   0.9426   0.9634   0.6779   0.5674  
0.70       0.9161   0.6177   0.9426   0.9634   0.6779   0.5674  
0.75       0.9161   0.6177   0.9426   0.9634   0.6779   0.5674  
0.80       0.9161   0.6177   0.9426   0.9634   0.6779   0.5674  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9161, F1=0.6177, Normal Recall=0.9426, Normal Precision=0.9634, Attack Recall=0.6779, Attack Precision=0.5674

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
0.15       0.8899   0.7113   0.9429   0.9213   0.6781   0.7479   <--
0.20       0.8899   0.7113   0.9429   0.9213   0.6781   0.7479  
0.25       0.8899   0.7113   0.9429   0.9213   0.6781   0.7479  
0.30       0.8899   0.7113   0.9429   0.9213   0.6781   0.7479  
0.35       0.8899   0.7113   0.9429   0.9213   0.6781   0.7479  
0.40       0.8899   0.7113   0.9429   0.9213   0.6781   0.7479  
0.45       0.8899   0.7113   0.9429   0.9213   0.6781   0.7479  
0.50       0.8899   0.7113   0.9429   0.9213   0.6781   0.7479  
0.55       0.8899   0.7113   0.9429   0.9213   0.6781   0.7479  
0.60       0.8899   0.7113   0.9429   0.9213   0.6781   0.7479  
0.65       0.8899   0.7113   0.9429   0.9213   0.6781   0.7479  
0.70       0.8899   0.7113   0.9429   0.9213   0.6781   0.7479  
0.75       0.8899   0.7113   0.9429   0.9213   0.6781   0.7479  
0.80       0.8899   0.7113   0.9429   0.9213   0.6781   0.7479  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8899, F1=0.7113, Normal Recall=0.9429, Normal Precision=0.9213, Attack Recall=0.6781, Attack Precision=0.7479

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
0.15       0.8634   0.7486   0.9429   0.8723   0.6780   0.8357   <--
0.20       0.8634   0.7486   0.9429   0.8723   0.6780   0.8357  
0.25       0.8634   0.7486   0.9429   0.8723   0.6780   0.8357  
0.30       0.8634   0.7486   0.9429   0.8723   0.6780   0.8357  
0.35       0.8634   0.7486   0.9429   0.8723   0.6780   0.8357  
0.40       0.8634   0.7486   0.9429   0.8723   0.6780   0.8357  
0.45       0.8634   0.7486   0.9429   0.8723   0.6780   0.8357  
0.50       0.8634   0.7486   0.9429   0.8723   0.6780   0.8357  
0.55       0.8634   0.7486   0.9429   0.8723   0.6780   0.8357  
0.60       0.8634   0.7486   0.9429   0.8723   0.6780   0.8357  
0.65       0.8634   0.7486   0.9429   0.8723   0.6780   0.8357  
0.70       0.8634   0.7486   0.9429   0.8723   0.6780   0.8357  
0.75       0.8634   0.7486   0.9429   0.8723   0.6780   0.8357  
0.80       0.8634   0.7486   0.9429   0.8723   0.6780   0.8357  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8634, F1=0.7486, Normal Recall=0.9429, Normal Precision=0.8723, Attack Recall=0.6780, Attack Precision=0.8357

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
0.15       0.8366   0.7685   0.9423   0.8145   0.6781   0.8867   <--
0.20       0.8366   0.7685   0.9423   0.8145   0.6781   0.8867  
0.25       0.8366   0.7685   0.9423   0.8145   0.6781   0.8867  
0.30       0.8366   0.7685   0.9423   0.8145   0.6781   0.8867  
0.35       0.8366   0.7685   0.9423   0.8145   0.6781   0.8867  
0.40       0.8366   0.7685   0.9423   0.8145   0.6781   0.8867  
0.45       0.8366   0.7685   0.9423   0.8145   0.6781   0.8867  
0.50       0.8366   0.7685   0.9423   0.8145   0.6781   0.8867  
0.55       0.8366   0.7685   0.9423   0.8145   0.6781   0.8867  
0.60       0.8366   0.7685   0.9423   0.8145   0.6781   0.8867  
0.65       0.8366   0.7685   0.9423   0.8145   0.6781   0.8867  
0.70       0.8366   0.7685   0.9423   0.8145   0.6781   0.8867  
0.75       0.8366   0.7685   0.9423   0.8145   0.6781   0.8867  
0.80       0.8366   0.7685   0.9423   0.8145   0.6781   0.8867  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8366, F1=0.7685, Normal Recall=0.9423, Normal Precision=0.8145, Attack Recall=0.6781, Attack Precision=0.8867

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
0.15       0.8093   0.7805   0.9405   0.7450   0.6781   0.9194   <--
0.20       0.8093   0.7805   0.9405   0.7450   0.6781   0.9194  
0.25       0.8093   0.7805   0.9405   0.7450   0.6781   0.9194  
0.30       0.8093   0.7805   0.9405   0.7450   0.6781   0.9194  
0.35       0.8093   0.7805   0.9405   0.7450   0.6781   0.9194  
0.40       0.8093   0.7805   0.9405   0.7450   0.6781   0.9194  
0.45       0.8093   0.7805   0.9405   0.7450   0.6781   0.9194  
0.50       0.8093   0.7805   0.9405   0.7450   0.6781   0.9194  
0.55       0.8093   0.7805   0.9405   0.7450   0.6781   0.9194  
0.60       0.8093   0.7805   0.9405   0.7450   0.6781   0.9194  
0.65       0.8093   0.7805   0.9405   0.7450   0.6781   0.9194  
0.70       0.8093   0.7805   0.9405   0.7450   0.6781   0.9194  
0.75       0.8093   0.7805   0.9405   0.7450   0.6781   0.9194  
0.80       0.8093   0.7805   0.9405   0.7450   0.6781   0.9194  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8093, F1=0.7805, Normal Recall=0.9405, Normal Precision=0.7450, Attack Recall=0.6781, Attack Precision=0.9194

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
0.15       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682   <--
0.20       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.25       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.30       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.35       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.40       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.45       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.50       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.55       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.60       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.65       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.70       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.75       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.80       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9164, F1=0.6200, Normal Recall=0.9424, Normal Precision=0.9639, Attack Recall=0.6821, Attack Precision=0.5682

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
0.15       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486   <--
0.20       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.25       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.30       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.35       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.40       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.45       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.50       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.55       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.60       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.65       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.70       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.75       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.80       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8907, F1=0.7141, Normal Recall=0.9427, Normal Precision=0.9224, Attack Recall=0.6827, Attack Precision=0.7486

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
0.15       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365   <--
0.20       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.25       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.30       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.35       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.40       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.45       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.50       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.55       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.60       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.65       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.70       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.75       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.80       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8648, F1=0.7518, Normal Recall=0.9428, Normal Precision=0.8739, Attack Recall=0.6827, Attack Precision=0.8365

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
0.15       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874   <--
0.20       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.25       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.30       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.35       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.40       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.45       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.50       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.55       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.60       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.65       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.70       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.75       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.80       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8384, F1=0.7717, Normal Recall=0.9422, Normal Precision=0.8167, Attack Recall=0.6827, Attack Precision=0.8874

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
0.15       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200   <--
0.20       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.25       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.30       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.35       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.40       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.45       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.50       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.55       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.60       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.65       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.70       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.75       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.80       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8117, F1=0.7838, Normal Recall=0.9406, Normal Precision=0.7478, Attack Recall=0.6827, Attack Precision=0.9200

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
0.15       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682   <--
0.20       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.25       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.30       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.35       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.40       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.45       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.50       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.55       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.60       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.65       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.70       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.75       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
0.80       0.9164   0.6200   0.9424   0.9639   0.6821   0.5682  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9164, F1=0.6200, Normal Recall=0.9424, Normal Precision=0.9639, Attack Recall=0.6821, Attack Precision=0.5682

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
0.15       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486   <--
0.20       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.25       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.30       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.35       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.40       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.45       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.50       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.55       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.60       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.65       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.70       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.75       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
0.80       0.8907   0.7141   0.9427   0.9224   0.6827   0.7486  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8907, F1=0.7141, Normal Recall=0.9427, Normal Precision=0.9224, Attack Recall=0.6827, Attack Precision=0.7486

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
0.15       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365   <--
0.20       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.25       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.30       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.35       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.40       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.45       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.50       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.55       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.60       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.65       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.70       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.75       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
0.80       0.8648   0.7518   0.9428   0.8739   0.6827   0.8365  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8648, F1=0.7518, Normal Recall=0.9428, Normal Precision=0.8739, Attack Recall=0.6827, Attack Precision=0.8365

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
0.15       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874   <--
0.20       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.25       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.30       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.35       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.40       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.45       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.50       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.55       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.60       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.65       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.70       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.75       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
0.80       0.8384   0.7717   0.9422   0.8167   0.6827   0.8874  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8384, F1=0.7717, Normal Recall=0.9422, Normal Precision=0.8167, Attack Recall=0.6827, Attack Precision=0.8874

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
0.15       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200   <--
0.20       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.25       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.30       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.35       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.40       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.45       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.50       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.55       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.60       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.65       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.70       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.75       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
0.80       0.8117   0.7838   0.9406   0.7478   0.6827   0.9200  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8117, F1=0.7838, Normal Recall=0.9406, Normal Precision=0.7478, Attack Recall=0.6827, Attack Precision=0.9200

```

