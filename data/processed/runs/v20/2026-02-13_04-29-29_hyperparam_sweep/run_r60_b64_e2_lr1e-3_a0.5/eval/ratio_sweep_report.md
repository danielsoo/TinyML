# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e2_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-14 02:49:39 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7651 | 0.7558 | 0.7464 | 0.7363 | 0.7264 | 0.7187 | 0.7095 | 0.7015 | 0.6923 | 0.6823 | 0.6736 |
| QAT+Prune only | 0.9574 | 0.9554 | 0.9524 | 0.9504 | 0.9467 | 0.9441 | 0.9412 | 0.9383 | 0.9358 | 0.9328 | 0.9302 |
| QAT+PTQ | 0.9578 | 0.9547 | 0.9506 | 0.9474 | 0.9424 | 0.9387 | 0.9347 | 0.9306 | 0.9268 | 0.9227 | 0.9190 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9578 | 0.9547 | 0.9506 | 0.9474 | 0.9424 | 0.9387 | 0.9347 | 0.9306 | 0.9268 | 0.9227 | 0.9190 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3565 | 0.5152 | 0.6051 | 0.6632 | 0.7054 | 0.7356 | 0.7595 | 0.7779 | 0.7923 | 0.8050 |
| QAT+Prune only | 0.0000 | 0.8067 | 0.8867 | 0.9184 | 0.9331 | 0.9433 | 0.9499 | 0.9547 | 0.9586 | 0.9614 | 0.9638 |
| QAT+PTQ | 0.0000 | 0.8023 | 0.8815 | 0.9129 | 0.9274 | 0.9374 | 0.9441 | 0.9488 | 0.9526 | 0.9554 | 0.9578 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.8023 | 0.8815 | 0.9129 | 0.9274 | 0.9374 | 0.9441 | 0.9488 | 0.9526 | 0.9554 | 0.9578 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7651 | 0.7647 | 0.7646 | 0.7631 | 0.7616 | 0.7639 | 0.7633 | 0.7666 | 0.7671 | 0.7603 | 0.0000 |
| QAT+Prune only | 0.9574 | 0.9581 | 0.9580 | 0.9591 | 0.9576 | 0.9580 | 0.9576 | 0.9571 | 0.9579 | 0.9561 | 0.0000 |
| QAT+PTQ | 0.9578 | 0.9585 | 0.9585 | 0.9595 | 0.9580 | 0.9584 | 0.9582 | 0.9576 | 0.9582 | 0.9560 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9578 | 0.9585 | 0.9585 | 0.9595 | 0.9580 | 0.9584 | 0.9582 | 0.9576 | 0.9582 | 0.9560 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7651 | 0.0000 | 0.0000 | 0.0000 | 0.7651 | 1.0000 |
| 90 | 10 | 299,940 | 0.7558 | 0.2420 | 0.6763 | 0.3565 | 0.7647 | 0.9551 |
| 80 | 20 | 291,350 | 0.7464 | 0.4171 | 0.6736 | 0.5152 | 0.7646 | 0.9036 |
| 70 | 30 | 194,230 | 0.7363 | 0.5493 | 0.6736 | 0.6051 | 0.7631 | 0.8451 |
| 60 | 40 | 145,675 | 0.7264 | 0.6532 | 0.6736 | 0.6632 | 0.7616 | 0.7778 |
| 50 | 50 | 116,540 | 0.7187 | 0.7404 | 0.6736 | 0.7054 | 0.7639 | 0.7006 |
| 40 | 60 | 97,115 | 0.7095 | 0.8102 | 0.6736 | 0.7356 | 0.7633 | 0.6092 |
| 30 | 70 | 83,240 | 0.7015 | 0.8707 | 0.6736 | 0.7595 | 0.7666 | 0.5016 |
| 20 | 80 | 72,835 | 0.6923 | 0.9204 | 0.6736 | 0.7779 | 0.7671 | 0.3701 |
| 10 | 90 | 64,740 | 0.6823 | 0.9620 | 0.6736 | 0.7923 | 0.7603 | 0.2056 |
| 0 | 100 | 58,270 | 0.6736 | 1.0000 | 0.6736 | 0.8050 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9574 | 0.0000 | 0.0000 | 0.0000 | 0.9574 | 1.0000 |
| 90 | 10 | 299,940 | 0.9554 | 0.7116 | 0.9311 | 0.8067 | 0.9581 | 0.9921 |
| 80 | 20 | 291,350 | 0.9524 | 0.8470 | 0.9302 | 0.8867 | 0.9580 | 0.9821 |
| 70 | 30 | 194,230 | 0.9504 | 0.9069 | 0.9302 | 0.9184 | 0.9591 | 0.9698 |
| 60 | 40 | 145,675 | 0.9467 | 0.9361 | 0.9302 | 0.9331 | 0.9576 | 0.9537 |
| 50 | 50 | 116,540 | 0.9441 | 0.9568 | 0.9302 | 0.9433 | 0.9580 | 0.9321 |
| 40 | 60 | 97,115 | 0.9412 | 0.9705 | 0.9302 | 0.9499 | 0.9576 | 0.9015 |
| 30 | 70 | 83,240 | 0.9383 | 0.9806 | 0.9302 | 0.9547 | 0.9571 | 0.8546 |
| 20 | 80 | 72,835 | 0.9358 | 0.9888 | 0.9302 | 0.9586 | 0.9579 | 0.7744 |
| 10 | 90 | 64,740 | 0.9328 | 0.9948 | 0.9302 | 0.9614 | 0.9561 | 0.6036 |
| 0 | 100 | 58,270 | 0.9302 | 1.0000 | 0.9302 | 0.9638 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9578 | 0.0000 | 0.0000 | 0.0000 | 0.9578 | 1.0000 |
| 90 | 10 | 299,940 | 0.9547 | 0.7113 | 0.9199 | 0.8023 | 0.9585 | 0.9908 |
| 80 | 20 | 291,350 | 0.9506 | 0.8470 | 0.9190 | 0.8815 | 0.9585 | 0.9793 |
| 70 | 30 | 194,230 | 0.9474 | 0.9068 | 0.9190 | 0.9129 | 0.9595 | 0.9651 |
| 60 | 40 | 145,675 | 0.9424 | 0.9359 | 0.9190 | 0.9274 | 0.9580 | 0.9466 |
| 50 | 50 | 116,540 | 0.9387 | 0.9567 | 0.9190 | 0.9374 | 0.9584 | 0.9221 |
| 40 | 60 | 97,115 | 0.9347 | 0.9706 | 0.9190 | 0.9441 | 0.9582 | 0.8874 |
| 30 | 70 | 83,240 | 0.9306 | 0.9806 | 0.9190 | 0.9488 | 0.9576 | 0.8351 |
| 20 | 80 | 72,835 | 0.9268 | 0.9888 | 0.9190 | 0.9526 | 0.9582 | 0.7473 |
| 10 | 90 | 64,740 | 0.9227 | 0.9947 | 0.9190 | 0.9554 | 0.9560 | 0.5673 |
| 0 | 100 | 58,270 | 0.9190 | 1.0000 | 0.9190 | 0.9578 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9578 | 0.0000 | 0.0000 | 0.0000 | 0.9578 | 1.0000 |
| 90 | 10 | 299,940 | 0.9547 | 0.7113 | 0.9199 | 0.8023 | 0.9585 | 0.9908 |
| 80 | 20 | 291,350 | 0.9506 | 0.8470 | 0.9190 | 0.8815 | 0.9585 | 0.9793 |
| 70 | 30 | 194,230 | 0.9474 | 0.9068 | 0.9190 | 0.9129 | 0.9595 | 0.9651 |
| 60 | 40 | 145,675 | 0.9424 | 0.9359 | 0.9190 | 0.9274 | 0.9580 | 0.9466 |
| 50 | 50 | 116,540 | 0.9387 | 0.9567 | 0.9190 | 0.9374 | 0.9584 | 0.9221 |
| 40 | 60 | 97,115 | 0.9347 | 0.9706 | 0.9190 | 0.9441 | 0.9582 | 0.8874 |
| 30 | 70 | 83,240 | 0.9306 | 0.9806 | 0.9190 | 0.9488 | 0.9576 | 0.8351 |
| 20 | 80 | 72,835 | 0.9268 | 0.9888 | 0.9190 | 0.9526 | 0.9582 | 0.7473 |
| 10 | 90 | 64,740 | 0.9227 | 0.9947 | 0.9190 | 0.9554 | 0.9560 | 0.5673 |
| 0 | 100 | 58,270 | 0.9190 | 1.0000 | 0.9190 | 0.9578 | 0.0000 | 0.0000 |


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
0.15       0.7556   0.3554   0.7647   0.9547   0.6738   0.2414   <--
0.20       0.7556   0.3554   0.7647   0.9547   0.6738   0.2414  
0.25       0.7556   0.3554   0.7647   0.9547   0.6738   0.2414  
0.30       0.7556   0.3554   0.7647   0.9547   0.6738   0.2414  
0.35       0.7556   0.3554   0.7647   0.9547   0.6738   0.2414  
0.40       0.7556   0.3554   0.7647   0.9547   0.6738   0.2414  
0.45       0.7556   0.3554   0.7647   0.9547   0.6738   0.2414  
0.50       0.7556   0.3554   0.7647   0.9547   0.6738   0.2414  
0.55       0.7556   0.3554   0.7647   0.9547   0.6738   0.2414  
0.60       0.7556   0.3554   0.7647   0.9547   0.6738   0.2414  
0.65       0.7556   0.3554   0.7647   0.9547   0.6738   0.2414  
0.70       0.7556   0.3554   0.7647   0.9547   0.6738   0.2414  
0.75       0.7556   0.3554   0.7647   0.9547   0.6738   0.2414  
0.80       0.7556   0.3554   0.7647   0.9547   0.6738   0.2414  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7556, F1=0.3554, Normal Recall=0.7647, Normal Precision=0.9547, Attack Recall=0.6738, Attack Precision=0.2414

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
0.15       0.7469   0.5157   0.7653   0.9036   0.6736   0.4177   <--
0.20       0.7469   0.5157   0.7653   0.9036   0.6736   0.4177  
0.25       0.7469   0.5157   0.7653   0.9036   0.6736   0.4177  
0.30       0.7469   0.5157   0.7653   0.9036   0.6736   0.4177  
0.35       0.7469   0.5157   0.7653   0.9036   0.6736   0.4177  
0.40       0.7469   0.5157   0.7653   0.9036   0.6736   0.4177  
0.45       0.7469   0.5157   0.7653   0.9036   0.6736   0.4177  
0.50       0.7469   0.5157   0.7653   0.9036   0.6736   0.4177  
0.55       0.7469   0.5157   0.7653   0.9036   0.6736   0.4177  
0.60       0.7469   0.5157   0.7653   0.9036   0.6736   0.4177  
0.65       0.7469   0.5157   0.7653   0.9036   0.6736   0.4177  
0.70       0.7469   0.5157   0.7653   0.9036   0.6736   0.4177  
0.75       0.7469   0.5157   0.7653   0.9036   0.6736   0.4177  
0.80       0.7469   0.5157   0.7653   0.9036   0.6736   0.4177  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7469, F1=0.5157, Normal Recall=0.7653, Normal Precision=0.9036, Attack Recall=0.6736, Attack Precision=0.4177

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
0.15       0.7376   0.6063   0.7651   0.8454   0.6736   0.5513   <--
0.20       0.7376   0.6063   0.7651   0.8454   0.6736   0.5513  
0.25       0.7376   0.6063   0.7651   0.8454   0.6736   0.5513  
0.30       0.7376   0.6063   0.7651   0.8454   0.6736   0.5513  
0.35       0.7376   0.6063   0.7651   0.8454   0.6736   0.5513  
0.40       0.7376   0.6063   0.7651   0.8454   0.6736   0.5513  
0.45       0.7376   0.6063   0.7651   0.8454   0.6736   0.5513  
0.50       0.7376   0.6063   0.7651   0.8454   0.6736   0.5513  
0.55       0.7376   0.6063   0.7651   0.8454   0.6736   0.5513  
0.60       0.7376   0.6063   0.7651   0.8454   0.6736   0.5513  
0.65       0.7376   0.6063   0.7651   0.8454   0.6736   0.5513  
0.70       0.7376   0.6063   0.7651   0.8454   0.6736   0.5513  
0.75       0.7376   0.6063   0.7651   0.8454   0.6736   0.5513  
0.80       0.7376   0.6063   0.7651   0.8454   0.6736   0.5513  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7376, F1=0.6063, Normal Recall=0.7651, Normal Precision=0.8454, Attack Recall=0.6736, Attack Precision=0.5513

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
0.15       0.7284   0.6649   0.7650   0.7785   0.6736   0.6565   <--
0.20       0.7284   0.6649   0.7650   0.7785   0.6736   0.6565  
0.25       0.7284   0.6649   0.7650   0.7785   0.6736   0.6565  
0.30       0.7284   0.6649   0.7650   0.7785   0.6736   0.6565  
0.35       0.7284   0.6649   0.7650   0.7785   0.6736   0.6565  
0.40       0.7284   0.6649   0.7650   0.7785   0.6736   0.6565  
0.45       0.7284   0.6649   0.7650   0.7785   0.6736   0.6565  
0.50       0.7284   0.6649   0.7650   0.7785   0.6736   0.6565  
0.55       0.7284   0.6649   0.7650   0.7785   0.6736   0.6565  
0.60       0.7284   0.6649   0.7650   0.7785   0.6736   0.6565  
0.65       0.7284   0.6649   0.7650   0.7785   0.6736   0.6565  
0.70       0.7284   0.6649   0.7650   0.7785   0.6736   0.6565  
0.75       0.7284   0.6649   0.7650   0.7785   0.6736   0.6565  
0.80       0.7284   0.6649   0.7650   0.7785   0.6736   0.6565  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7284, F1=0.6649, Normal Recall=0.7650, Normal Precision=0.7785, Attack Recall=0.6736, Attack Precision=0.6565

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
0.15       0.7192   0.7058   0.7648   0.7009   0.6736   0.7412   <--
0.20       0.7192   0.7058   0.7648   0.7009   0.6736   0.7412  
0.25       0.7192   0.7058   0.7648   0.7009   0.6736   0.7412  
0.30       0.7192   0.7058   0.7648   0.7009   0.6736   0.7412  
0.35       0.7192   0.7058   0.7648   0.7009   0.6736   0.7412  
0.40       0.7192   0.7058   0.7648   0.7009   0.6736   0.7412  
0.45       0.7192   0.7058   0.7648   0.7009   0.6736   0.7412  
0.50       0.7192   0.7058   0.7648   0.7009   0.6736   0.7412  
0.55       0.7192   0.7058   0.7648   0.7009   0.6736   0.7412  
0.60       0.7192   0.7058   0.7648   0.7009   0.6736   0.7412  
0.65       0.7192   0.7058   0.7648   0.7009   0.6736   0.7412  
0.70       0.7192   0.7058   0.7648   0.7009   0.6736   0.7412  
0.75       0.7192   0.7058   0.7648   0.7009   0.6736   0.7412  
0.80       0.7192   0.7058   0.7648   0.7009   0.6736   0.7412  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7192, F1=0.7058, Normal Recall=0.7648, Normal Precision=0.7009, Attack Recall=0.6736, Attack Precision=0.7412

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
0.15       0.9553   0.8063   0.9581   0.9920   0.9304   0.7115   <--
0.20       0.9553   0.8063   0.9581   0.9920   0.9304   0.7115  
0.25       0.9553   0.8063   0.9581   0.9920   0.9304   0.7115  
0.30       0.9553   0.8063   0.9581   0.9920   0.9304   0.7115  
0.35       0.9553   0.8063   0.9581   0.9920   0.9304   0.7115  
0.40       0.9553   0.8063   0.9581   0.9920   0.9304   0.7115  
0.45       0.9553   0.8063   0.9581   0.9920   0.9304   0.7115  
0.50       0.9553   0.8063   0.9581   0.9920   0.9304   0.7115  
0.55       0.9553   0.8063   0.9581   0.9920   0.9304   0.7115  
0.60       0.9553   0.8063   0.9581   0.9920   0.9304   0.7115  
0.65       0.9553   0.8063   0.9581   0.9920   0.9304   0.7115  
0.70       0.9553   0.8063   0.9581   0.9920   0.9304   0.7115  
0.75       0.9553   0.8063   0.9581   0.9920   0.9304   0.7115  
0.80       0.9553   0.8063   0.9581   0.9920   0.9304   0.7115  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9553, F1=0.8063, Normal Recall=0.9581, Normal Precision=0.9920, Attack Recall=0.9304, Attack Precision=0.7115

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
0.15       0.9525   0.8869   0.9581   0.9821   0.9302   0.8473   <--
0.20       0.9525   0.8869   0.9581   0.9821   0.9302   0.8473  
0.25       0.9525   0.8869   0.9581   0.9821   0.9302   0.8473  
0.30       0.9525   0.8869   0.9581   0.9821   0.9302   0.8473  
0.35       0.9525   0.8869   0.9581   0.9821   0.9302   0.8473  
0.40       0.9525   0.8869   0.9581   0.9821   0.9302   0.8473  
0.45       0.9525   0.8869   0.9581   0.9821   0.9302   0.8473  
0.50       0.9525   0.8869   0.9581   0.9821   0.9302   0.8473  
0.55       0.9525   0.8869   0.9581   0.9821   0.9302   0.8473  
0.60       0.9525   0.8869   0.9581   0.9821   0.9302   0.8473  
0.65       0.9525   0.8869   0.9581   0.9821   0.9302   0.8473  
0.70       0.9525   0.8869   0.9581   0.9821   0.9302   0.8473  
0.75       0.9525   0.8869   0.9581   0.9821   0.9302   0.8473  
0.80       0.9525   0.8869   0.9581   0.9821   0.9302   0.8473  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9525, F1=0.8869, Normal Recall=0.9581, Normal Precision=0.9821, Attack Recall=0.9302, Attack Precision=0.8473

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
0.15       0.9494   0.9169   0.9577   0.9697   0.9302   0.9040   <--
0.20       0.9494   0.9169   0.9577   0.9697   0.9302   0.9040  
0.25       0.9494   0.9169   0.9577   0.9697   0.9302   0.9040  
0.30       0.9494   0.9169   0.9577   0.9697   0.9302   0.9040  
0.35       0.9494   0.9169   0.9577   0.9697   0.9302   0.9040  
0.40       0.9494   0.9169   0.9577   0.9697   0.9302   0.9040  
0.45       0.9494   0.9169   0.9577   0.9697   0.9302   0.9040  
0.50       0.9494   0.9169   0.9577   0.9697   0.9302   0.9040  
0.55       0.9494   0.9169   0.9577   0.9697   0.9302   0.9040  
0.60       0.9494   0.9169   0.9577   0.9697   0.9302   0.9040  
0.65       0.9494   0.9169   0.9577   0.9697   0.9302   0.9040  
0.70       0.9494   0.9169   0.9577   0.9697   0.9302   0.9040  
0.75       0.9494   0.9169   0.9577   0.9697   0.9302   0.9040  
0.80       0.9494   0.9169   0.9577   0.9697   0.9302   0.9040  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9494, F1=0.9169, Normal Recall=0.9577, Normal Precision=0.9697, Attack Recall=0.9302, Attack Precision=0.9040

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
0.15       0.9467   0.9331   0.9576   0.9537   0.9302   0.9360   <--
0.20       0.9467   0.9331   0.9576   0.9537   0.9302   0.9360  
0.25       0.9467   0.9331   0.9576   0.9537   0.9302   0.9360  
0.30       0.9467   0.9331   0.9576   0.9537   0.9302   0.9360  
0.35       0.9467   0.9331   0.9576   0.9537   0.9302   0.9360  
0.40       0.9467   0.9331   0.9576   0.9537   0.9302   0.9360  
0.45       0.9467   0.9331   0.9576   0.9537   0.9302   0.9360  
0.50       0.9467   0.9331   0.9576   0.9537   0.9302   0.9360  
0.55       0.9467   0.9331   0.9576   0.9537   0.9302   0.9360  
0.60       0.9467   0.9331   0.9576   0.9537   0.9302   0.9360  
0.65       0.9467   0.9331   0.9576   0.9537   0.9302   0.9360  
0.70       0.9467   0.9331   0.9576   0.9537   0.9302   0.9360  
0.75       0.9467   0.9331   0.9576   0.9537   0.9302   0.9360  
0.80       0.9467   0.9331   0.9576   0.9537   0.9302   0.9360  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9467, F1=0.9331, Normal Recall=0.9576, Normal Precision=0.9537, Attack Recall=0.9302, Attack Precision=0.9360

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
0.15       0.9441   0.9433   0.9579   0.9321   0.9302   0.9567   <--
0.20       0.9441   0.9433   0.9579   0.9321   0.9302   0.9567  
0.25       0.9441   0.9433   0.9579   0.9321   0.9302   0.9567  
0.30       0.9441   0.9433   0.9579   0.9321   0.9302   0.9567  
0.35       0.9441   0.9433   0.9579   0.9321   0.9302   0.9567  
0.40       0.9441   0.9433   0.9579   0.9321   0.9302   0.9567  
0.45       0.9441   0.9433   0.9579   0.9321   0.9302   0.9567  
0.50       0.9441   0.9433   0.9579   0.9321   0.9302   0.9567  
0.55       0.9441   0.9433   0.9579   0.9321   0.9302   0.9567  
0.60       0.9441   0.9433   0.9579   0.9321   0.9302   0.9567  
0.65       0.9441   0.9433   0.9579   0.9321   0.9302   0.9567  
0.70       0.9441   0.9433   0.9579   0.9321   0.9302   0.9567  
0.75       0.9441   0.9433   0.9579   0.9321   0.9302   0.9567  
0.80       0.9441   0.9433   0.9579   0.9321   0.9302   0.9567  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9441, F1=0.9433, Normal Recall=0.9579, Normal Precision=0.9321, Attack Recall=0.9302, Attack Precision=0.9567

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
0.15       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111   <--
0.20       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.25       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.30       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.35       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.40       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.45       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.50       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.55       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.60       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.65       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.70       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.75       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.80       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9546, F1=0.8018, Normal Recall=0.9585, Normal Precision=0.9907, Attack Recall=0.9190, Attack Precision=0.7111

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
0.15       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471   <--
0.20       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.25       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.30       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.35       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.40       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.45       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.50       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.55       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.60       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.65       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.70       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.75       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.80       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9506, F1=0.8816, Normal Recall=0.9585, Normal Precision=0.9793, Attack Recall=0.9190, Attack Precision=0.8471

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
0.15       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038   <--
0.20       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.25       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.30       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.35       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.40       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.45       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.50       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.55       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.60       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.65       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.70       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.75       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.80       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9464, F1=0.9113, Normal Recall=0.9581, Normal Precision=0.9650, Attack Recall=0.9190, Attack Precision=0.9038

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
0.15       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359   <--
0.20       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.25       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.30       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.35       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.40       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.45       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.50       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.55       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.60       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.65       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.70       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.75       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.80       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9424, F1=0.9274, Normal Recall=0.9580, Normal Precision=0.9466, Attack Recall=0.9190, Attack Precision=0.9359

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
0.15       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566   <--
0.20       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.25       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.30       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.35       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.40       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.45       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.50       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.55       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.60       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.65       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.70       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.75       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.80       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9387, F1=0.9374, Normal Recall=0.9583, Normal Precision=0.9220, Attack Recall=0.9190, Attack Precision=0.9566

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
0.15       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111   <--
0.20       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.25       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.30       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.35       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.40       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.45       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.50       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.55       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.60       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.65       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.70       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.75       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
0.80       0.9546   0.8018   0.9585   0.9907   0.9190   0.7111  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9546, F1=0.8018, Normal Recall=0.9585, Normal Precision=0.9907, Attack Recall=0.9190, Attack Precision=0.7111

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
0.15       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471   <--
0.20       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.25       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.30       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.35       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.40       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.45       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.50       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.55       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.60       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.65       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.70       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.75       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
0.80       0.9506   0.8816   0.9585   0.9793   0.9190   0.8471  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9506, F1=0.8816, Normal Recall=0.9585, Normal Precision=0.9793, Attack Recall=0.9190, Attack Precision=0.8471

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
0.15       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038   <--
0.20       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.25       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.30       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.35       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.40       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.45       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.50       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.55       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.60       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.65       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.70       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.75       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
0.80       0.9464   0.9113   0.9581   0.9650   0.9190   0.9038  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9464, F1=0.9113, Normal Recall=0.9581, Normal Precision=0.9650, Attack Recall=0.9190, Attack Precision=0.9038

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
0.15       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359   <--
0.20       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.25       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.30       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.35       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.40       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.45       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.50       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.55       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.60       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.65       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.70       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.75       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
0.80       0.9424   0.9274   0.9580   0.9466   0.9190   0.9359  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9424, F1=0.9274, Normal Recall=0.9580, Normal Precision=0.9466, Attack Recall=0.9190, Attack Precision=0.9359

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
0.15       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566   <--
0.20       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.25       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.30       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.35       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.40       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.45       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.50       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.55       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.60       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.65       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.70       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.75       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
0.80       0.9387   0.9374   0.9583   0.9220   0.9190   0.9566  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9387, F1=0.9374, Normal Recall=0.9583, Normal Precision=0.9220, Attack Recall=0.9190, Attack Precision=0.9566

```

