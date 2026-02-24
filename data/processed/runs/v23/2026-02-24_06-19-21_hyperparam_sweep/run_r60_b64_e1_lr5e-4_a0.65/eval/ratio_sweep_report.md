# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-24 11:20:08 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1000 | 0.2000 | 0.3000 | 0.4000 | 0.5000 | 0.6000 | 0.7000 | 0.8000 | 0.9000 | 1.0000 |
| noQAT+PTQ | 0.0521 | 0.1469 | 0.2418 | 0.3362 | 0.4319 | 0.5260 | 0.6211 | 0.7160 | 0.8101 | 0.9056 | 1.0000 |
| saved_model_traditional_qat | 0.9518 | 0.9527 | 0.9531 | 0.9540 | 0.9547 | 0.9550 | 0.9560 | 0.9568 | 0.9571 | 0.9578 | 0.9583 |
| QAT+PTQ | 0.0000 | 0.1000 | 0.2000 | 0.3000 | 0.4000 | 0.5000 | 0.6000 | 0.7000 | 0.8000 | 0.9000 | 1.0000 |
| Compressed (QAT) | 0.9327 | 0.9338 | 0.9351 | 0.9375 | 0.9391 | 0.9407 | 0.9425 | 0.9447 | 0.9462 | 0.9478 | 0.9492 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1818 | 0.3333 | 0.4615 | 0.5714 | 0.6667 | 0.7500 | 0.8235 | 0.8889 | 0.9474 | 1.0000 |
| noQAT+PTQ | 0.0000 | 0.1899 | 0.3454 | 0.4748 | 0.5847 | 0.6784 | 0.7600 | 0.8314 | 0.8939 | 0.9502 | 1.0000 |
| saved_model_traditional_qat | 0.0000 | 0.8020 | 0.8909 | 0.9259 | 0.9442 | 0.9551 | 0.9632 | 0.9688 | 0.9728 | 0.9761 | 0.9787 |
| QAT+PTQ | 0.0000 | 0.1818 | 0.3333 | 0.4615 | 0.5714 | 0.6667 | 0.7500 | 0.8235 | 0.8889 | 0.9474 | 1.0000 |
| Compressed (QAT) | 0.0000 | 0.7414 | 0.8540 | 0.9011 | 0.9258 | 0.9412 | 0.9520 | 0.9600 | 0.9658 | 0.9704 | 0.9739 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| noQAT+PTQ | 0.0521 | 0.0521 | 0.0522 | 0.0518 | 0.0531 | 0.0519 | 0.0527 | 0.0535 | 0.0505 | 0.0562 | 0.0000 |
| saved_model_traditional_qat | 0.9518 | 0.9520 | 0.9517 | 0.9522 | 0.9522 | 0.9517 | 0.9527 | 0.9534 | 0.9522 | 0.9534 | 0.0000 |
| QAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (QAT) | 0.9327 | 0.9321 | 0.9315 | 0.9324 | 0.9324 | 0.9323 | 0.9325 | 0.9342 | 0.9343 | 0.9356 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

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
| 100 | 0 | 100,000 | 0.9518 | 0.0000 | 0.0000 | 0.0000 | 0.9518 | 1.0000 |
| 90 | 10 | 299,940 | 0.9527 | 0.6893 | 0.9586 | 0.8020 | 0.9520 | 0.9952 |
| 80 | 20 | 291,350 | 0.9531 | 0.8323 | 0.9583 | 0.8909 | 0.9517 | 0.9892 |
| 70 | 30 | 194,230 | 0.9540 | 0.8957 | 0.9583 | 0.9259 | 0.9522 | 0.9816 |
| 60 | 40 | 145,675 | 0.9547 | 0.9304 | 0.9583 | 0.9442 | 0.9522 | 0.9716 |
| 50 | 50 | 116,540 | 0.9550 | 0.9520 | 0.9583 | 0.9551 | 0.9517 | 0.9580 |
| 40 | 60 | 97,115 | 0.9560 | 0.9681 | 0.9583 | 0.9632 | 0.9527 | 0.9384 |
| 30 | 70 | 83,240 | 0.9568 | 0.9796 | 0.9583 | 0.9688 | 0.9534 | 0.9074 |
| 20 | 80 | 72,835 | 0.9571 | 0.9877 | 0.9583 | 0.9728 | 0.9522 | 0.8509 |
| 10 | 90 | 64,740 | 0.9578 | 0.9946 | 0.9583 | 0.9761 | 0.9534 | 0.7175 |
| 0 | 100 | 58,270 | 0.9583 | 1.0000 | 0.9583 | 0.9787 | 0.0000 | 0.0000 |

### QAT+PTQ

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

### Compressed (QAT)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9327 | 0.0000 | 0.0000 | 0.0000 | 0.9327 | 1.0000 |
| 90 | 10 | 299,940 | 0.9338 | 0.6082 | 0.9491 | 0.7414 | 0.9321 | 0.9940 |
| 80 | 20 | 291,350 | 0.9351 | 0.7761 | 0.9492 | 0.8540 | 0.9315 | 0.9866 |
| 70 | 30 | 194,230 | 0.9375 | 0.8576 | 0.9492 | 0.9011 | 0.9324 | 0.9772 |
| 60 | 40 | 145,675 | 0.9391 | 0.9035 | 0.9492 | 0.9258 | 0.9324 | 0.9650 |
| 50 | 50 | 116,540 | 0.9407 | 0.9334 | 0.9492 | 0.9412 | 0.9323 | 0.9483 |
| 40 | 60 | 97,115 | 0.9425 | 0.9547 | 0.9492 | 0.9520 | 0.9325 | 0.9245 |
| 30 | 70 | 83,240 | 0.9447 | 0.9712 | 0.9492 | 0.9600 | 0.9342 | 0.8874 |
| 20 | 80 | 72,835 | 0.9462 | 0.9830 | 0.9492 | 0.9658 | 0.9343 | 0.8214 |
| 10 | 90 | 64,740 | 0.9478 | 0.9925 | 0.9492 | 0.9704 | 0.9356 | 0.6717 |
| 0 | 100 | 58,270 | 0.9492 | 1.0000 | 0.9492 | 0.9739 | 0.0000 | 0.0000 |


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
0.20       0.1000   0.1818   0.0000   0.0000   1.0000   0.1000  
0.25       0.1000   0.1818   0.0000   0.0000   1.0000   0.1000  
0.30       0.1000   0.1818   0.0000   0.0000   1.0000   0.1000  
0.35       0.1000   0.1818   0.0000   0.0000   1.0000   0.1000  
0.40       0.1000   0.1818   0.0000   1.0000   1.0000   0.1000  
0.45       0.1000   0.1818   0.0000   1.0000   1.0000   0.1000  
0.50       0.4532   0.2674   0.3927   0.9994   0.9980   0.1544  
0.55       0.7397   0.3440   0.7460   0.9548   0.6825   0.2299  
0.60       0.8801   0.4290   0.9278   0.9383   0.4506   0.4094   <--
0.65       0.8967   0.3095   0.9706   0.9191   0.2316   0.4665  
0.70       0.8870   0.0434   0.9827   0.9008   0.0256   0.1414  
0.75       0.8906   0.0015   0.9895   0.8991   0.0008   0.0084  
0.80       0.8950   0.0006   0.9944   0.8995   0.0003   0.0060  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.6
  At threshold 0.6: Accuracy=0.8801, F1=0.4290, Normal Recall=0.9278, Normal Precision=0.9383, Attack Recall=0.4506, Attack Precision=0.4094

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
0.20       0.2000   0.3333   0.0000   0.0000   1.0000   0.2000  
0.25       0.2000   0.3333   0.0000   0.0000   1.0000   0.2000  
0.30       0.2000   0.3333   0.0000   0.0000   1.0000   0.2000  
0.35       0.2000   0.3333   0.0000   0.0000   1.0000   0.2000  
0.40       0.2000   0.3333   0.0000   1.0000   1.0000   0.2000  
0.45       0.2000   0.3333   0.0000   1.0000   1.0000   0.2000  
0.50       0.5145   0.4512   0.3936   0.9986   0.9979   0.2915  
0.55       0.7342   0.5076   0.7465   0.9046   0.6850   0.4032  
0.60       0.8326   0.5192   0.9279   0.8713   0.4518   0.6103   <--
0.65       0.8228   0.3430   0.9707   0.8347   0.2312   0.6635  
0.70       0.7915   0.0477   0.9828   0.8015   0.0261   0.2752  
0.75       0.7919   0.0022   0.9895   0.7985   0.0011   0.0264  
0.80       0.7956   0.0008   0.9944   0.7992   0.0004   0.0173  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.6
  At threshold 0.6: Accuracy=0.8326, F1=0.5192, Normal Recall=0.9279, Normal Precision=0.8713, Attack Recall=0.4518, Attack Precision=0.6103

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
0.20       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.25       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.30       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.35       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.40       0.3000   0.4615   0.0000   1.0000   1.0000   0.3000  
0.45       0.3000   0.4615   0.0000   1.0000   1.0000   0.3000  
0.50       0.5752   0.5849   0.3940   0.9977   0.9979   0.4137  
0.55       0.7278   0.6016   0.7462   0.8468   0.6850   0.5363   <--
0.60       0.7844   0.5570   0.9270   0.7978   0.4518   0.7262  
0.65       0.7486   0.3556   0.9703   0.7465   0.2312   0.7696  
0.70       0.6957   0.0490   0.9826   0.7019   0.0261   0.3917  
0.75       0.6928   0.0022   0.9893   0.6980   0.0011   0.0433  
0.80       0.6961   0.0008   0.9943   0.6989   0.0004   0.0287  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.7278, F1=0.6016, Normal Recall=0.7462, Normal Precision=0.8468, Attack Recall=0.6850, Attack Precision=0.5363

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
0.20       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.25       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.30       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.35       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.40       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.45       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.50       0.6347   0.6860   0.3926   0.9964   0.9979   0.5227   <--
0.55       0.7215   0.6631   0.7459   0.7803   0.6850   0.6425  
0.60       0.7369   0.5787   0.9270   0.7172   0.4518   0.8050  
0.65       0.6750   0.3627   0.9708   0.6545   0.2312   0.8406  
0.70       0.6004   0.0497   0.9832   0.6023   0.0261   0.5087  
0.75       0.5943   0.0022   0.9898   0.5978   0.0011   0.0687  
0.80       0.5967   0.0008   0.9943   0.5987   0.0004   0.0442  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.6347, F1=0.6860, Normal Recall=0.3926, Normal Precision=0.9964, Attack Recall=0.9979, Attack Precision=0.5227

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
0.20       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.25       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.30       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.35       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.40       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.45       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.50       0.6950   0.7659   0.3922   0.9946   0.9979   0.6214   <--
0.55       0.7158   0.7068   0.7466   0.7033   0.6850   0.7300  
0.60       0.6894   0.5926   0.9270   0.6284   0.4518   0.8610  
0.65       0.6012   0.3670   0.9712   0.5582   0.2312   0.8891  
0.70       0.5047   0.0501   0.9832   0.5024   0.0261   0.6086  
0.75       0.4954   0.0022   0.9896   0.4977   0.0011   0.0985  
0.80       0.4973   0.0008   0.9942   0.4986   0.0004   0.0639  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.6950, F1=0.7659, Normal Recall=0.3922, Normal Precision=0.9946, Attack Recall=0.9979, Attack Precision=0.6214

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
0.15       0.9397   0.7630   0.9362   0.9966   0.9712   0.6284  
0.20       0.9426   0.7717   0.9396   0.9964   0.9698   0.6409  
0.25       0.9442   0.7761   0.9418   0.9961   0.9665   0.6484  
0.30       0.9528   0.8025   0.9520   0.9953   0.9597   0.6896  
0.35       0.9632   0.8360   0.9660   0.9929   0.9380   0.7541  
0.40       0.9662   0.8463   0.9700   0.9922   0.9314   0.7755  
0.45       0.9673   0.8492   0.9723   0.9911   0.9218   0.7873  
0.50       0.9671   0.8441   0.9755   0.9878   0.8912   0.8017  
0.55       0.9681   0.8478   0.9767   0.9876   0.8899   0.8095  
0.60       0.9759   0.8734   0.9920   0.9814   0.8312   0.9202   <--
0.65       0.9742   0.8595   0.9948   0.9769   0.7885   0.9445  
0.70       0.9748   0.8608   0.9967   0.9758   0.7780   0.9634  
0.75       0.9744   0.8566   0.9975   0.9746   0.7663   0.9711  
0.80       0.9748   0.8584   0.9980   0.9745   0.7652   0.9776  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.6
  At threshold 0.6: Accuracy=0.9759, F1=0.8734, Normal Recall=0.9920, Normal Precision=0.9814, Attack Recall=0.8312, Attack Precision=0.9202

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
0.15       0.9431   0.8721   0.9363   0.9921   0.9701   0.7920  
0.20       0.9455   0.8767   0.9397   0.9917   0.9687   0.8007  
0.25       0.9466   0.8784   0.9418   0.9909   0.9654   0.8058  
0.30       0.9533   0.8913   0.9520   0.9892   0.9583   0.8331  
0.35       0.9602   0.9040   0.9660   0.9839   0.9369   0.8732  
0.40       0.9623   0.9081   0.9701   0.9825   0.9310   0.8863   <--
0.45       0.9622   0.9071   0.9724   0.9802   0.9215   0.8931  
0.50       0.9585   0.8956   0.9756   0.9726   0.8902   0.9011  
0.55       0.9592   0.8971   0.9767   0.9724   0.8891   0.9052  
0.60       0.9596   0.8916   0.9920   0.9589   0.8301   0.9630  
0.65       0.9533   0.8708   0.9949   0.9492   0.7870   0.9745  
0.70       0.9528   0.8681   0.9967   0.9470   0.7770   0.9834  
0.75       0.9511   0.8622   0.9975   0.9445   0.7655   0.9870  
0.80       0.9513   0.8625   0.9980   0.9442   0.7643   0.9898  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9623, F1=0.9081, Normal Recall=0.9701, Normal Precision=0.9825, Attack Recall=0.9310, Attack Precision=0.8863

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
0.15       0.9465   0.9158   0.9363   0.9865   0.9701   0.8672  
0.20       0.9485   0.9186   0.9398   0.9859   0.9687   0.8734  
0.25       0.9491   0.9192   0.9421   0.9845   0.9654   0.8772  
0.30       0.9540   0.9259   0.9521   0.9816   0.9583   0.8956  
0.35       0.9570   0.9290   0.9656   0.9728   0.9369   0.9212  
0.40       0.9582   0.9304   0.9698   0.9704   0.9310   0.9297   <--
0.45       0.9569   0.9277   0.9721   0.9665   0.9215   0.9340  
0.50       0.9498   0.9141   0.9753   0.9540   0.8902   0.9393  
0.55       0.9503   0.9148   0.9765   0.9536   0.8891   0.9419  
0.60       0.9435   0.8981   0.9921   0.9316   0.8301   0.9782  
0.65       0.9325   0.8749   0.9948   0.9160   0.7870   0.9849  
0.70       0.9308   0.8707   0.9967   0.9125   0.7769   0.9901  
0.75       0.9278   0.8642   0.9974   0.9084   0.7654   0.9921  
0.80       0.9279   0.8641   0.9980   0.9081   0.7642   0.9938  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9582, F1=0.9304, Normal Recall=0.9698, Normal Precision=0.9704, Attack Recall=0.9310, Attack Precision=0.9297

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
0.15       0.9499   0.9394   0.9365   0.9792   0.9701   0.9106  
0.20       0.9515   0.9412   0.9401   0.9783   0.9687   0.9152  
0.25       0.9515   0.9409   0.9423   0.9761   0.9654   0.9177  
0.30       0.9546   0.9441   0.9522   0.9716   0.9583   0.9304   <--
0.35       0.9543   0.9425   0.9659   0.9583   0.9369   0.9482  
0.40       0.9544   0.9423   0.9700   0.9547   0.9310   0.9539  
0.45       0.9519   0.9387   0.9722   0.9489   0.9215   0.9566  
0.50       0.9412   0.9238   0.9752   0.9302   0.8902   0.9599  
0.55       0.9415   0.9239   0.9764   0.9296   0.8891   0.9616  
0.60       0.9272   0.9012   0.9919   0.8975   0.8301   0.9855  
0.65       0.9116   0.8769   0.9946   0.8751   0.7870   0.9899  
0.70       0.9087   0.8719   0.9965   0.8702   0.7770   0.9933  
0.75       0.9045   0.8651   0.9972   0.8645   0.7655   0.9946  
0.80       0.9044   0.8648   0.9978   0.8639   0.7643   0.9957  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9546, F1=0.9441, Normal Recall=0.9522, Normal Precision=0.9716, Attack Recall=0.9583, Attack Precision=0.9304

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
0.15       0.9533   0.9541   0.9366   0.9691   0.9701   0.9386  
0.20       0.9545   0.9551   0.9403   0.9678   0.9687   0.9419  
0.25       0.9539   0.9544   0.9424   0.9646   0.9654   0.9437  
0.30       0.9552   0.9554   0.9522   0.9580   0.9583   0.9525   <--
0.35       0.9513   0.9506   0.9657   0.9387   0.9369   0.9647  
0.40       0.9506   0.9496   0.9702   0.9336   0.9310   0.9689  
0.45       0.9470   0.9456   0.9725   0.9253   0.9215   0.9710  
0.50       0.9329   0.9299   0.9756   0.8989   0.8902   0.9734  
0.55       0.9329   0.9299   0.9768   0.8980   0.8891   0.9745  
0.60       0.9110   0.9032   0.9920   0.8538   0.8301   0.9904  
0.65       0.8909   0.8782   0.9947   0.8236   0.7870   0.9934  
0.70       0.8867   0.8728   0.9965   0.8171   0.7770   0.9955  
0.75       0.8813   0.8658   0.9972   0.8096   0.7655   0.9964  
0.80       0.8810   0.8653   0.9978   0.8089   0.7643   0.9971  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9552, F1=0.9554, Normal Recall=0.9522, Normal Precision=0.9580, Attack Recall=0.9583, Attack Precision=0.9525

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
0.15       0.1000   0.1818   0.0000   0.0000   1.0000   0.1000  
0.20       0.1000   0.1818   0.0000   0.0000   1.0000   0.1000  
0.25       0.1000   0.1818   0.0000   0.0000   1.0000   0.1000  
0.30       0.1000   0.1818   0.0000   0.0000   1.0000   0.1000  
0.35       0.1000   0.1818   0.0000   0.0000   1.0000   0.1000  
0.40       0.1000   0.1818   0.0000   0.0000   1.0000   0.1000  
0.45       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.50       0.1647   0.1887   0.0750   0.9595   0.9715   0.1045   <--
0.55       0.8780   0.0594   0.9712   0.9009   0.0385   0.1296  
0.60       0.8927   0.0067   0.9915   0.8996   0.0036   0.0451  
0.65       0.8991   0.0015   0.9989   0.9000   0.0007   0.0681  
0.70       0.8994   0.0001   0.9993   0.8999   0.0000   0.0056  
0.75       0.8995   0.0000   0.9994   0.8999   0.0000   0.0000  
0.80       0.8995   0.0000   0.9995   0.9000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.1647, F1=0.1887, Normal Recall=0.0750, Normal Precision=0.9595, Attack Recall=0.9715, Attack Precision=0.1045

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
0.20       0.2000   0.3333   0.0000   0.0000   1.0000   0.2000  
0.25       0.2000   0.3333   0.0000   0.0000   1.0000   0.2000  
0.30       0.2000   0.3333   0.0000   0.0000   1.0000   0.2000  
0.35       0.2000   0.3333   0.0000   0.0000   1.0000   0.2000  
0.40       0.2000   0.3333   0.0000   0.0000   1.0000   0.2000  
0.45       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.50       0.2545   0.3427   0.0751   0.9144   0.9719   0.2080   <--
0.55       0.7846   0.0668   0.9712   0.8016   0.0385   0.2504  
0.60       0.7939   0.0072   0.9914   0.7992   0.0037   0.0985  
0.65       0.7993   0.0020   0.9988   0.8000   0.0010   0.1763  
0.70       0.7995   0.0001   0.9993   0.7999   0.0001   0.0189  
0.75       0.7995   0.0001   0.9994   0.7999   0.0000   0.0139  
0.80       0.7996   0.0000   0.9995   0.7999   0.0000   0.0083  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.2545, F1=0.3427, Normal Recall=0.0751, Normal Precision=0.9144, Attack Recall=0.9719, Attack Precision=0.2080

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
0.20       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.25       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.30       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.35       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.40       0.3000   0.4615   0.0000   0.0000   1.0000   0.3000  
0.45       0.3000   0.4616   0.0001   1.0000   1.0000   0.3000  
0.50       0.3441   0.4706   0.0751   0.8616   0.9719   0.3105   <--
0.55       0.6913   0.0696   0.9711   0.7021   0.0385   0.3635  
0.60       0.6951   0.0073   0.9914   0.6990   0.0037   0.1574  
0.65       0.6995   0.0020   0.9989   0.7000   0.0010   0.2736  
0.70       0.6996   0.0001   0.9994   0.6999   0.0001   0.0366  
0.75       0.6996   0.0001   0.9995   0.6999   0.0000   0.0267  
0.80       0.6997   0.0000   0.9996   0.6999   0.0000   0.0161  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.3441, F1=0.4706, Normal Recall=0.0751, Normal Precision=0.8616, Attack Recall=0.9719, Attack Precision=0.3105

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
0.20       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.25       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.30       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.35       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.40       0.4000   0.5714   0.0000   0.0000   1.0000   0.4000  
0.45       0.4000   0.5714   0.0000   1.0000   1.0000   0.4000  
0.50       0.4334   0.5785   0.0745   0.7988   0.9719   0.4118   <--
0.55       0.5980   0.0712   0.9710   0.6024   0.0385   0.4695  
0.60       0.5963   0.0074   0.9914   0.5988   0.0037   0.2250  
0.65       0.5997   0.0020   0.9989   0.6000   0.0010   0.3694  
0.70       0.5997   0.0001   0.9994   0.5999   0.0001   0.0577  
0.75       0.5997   0.0001   0.9995   0.5999   0.0000   0.0417  
0.80       0.5998   0.0000   0.9996   0.5999   0.0000   0.0278  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.4334, F1=0.5785, Normal Recall=0.0745, Normal Precision=0.7988, Attack Recall=0.9719, Attack Precision=0.4118

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
0.20       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.25       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.30       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.35       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.40       0.5000   0.6667   0.0000   0.0000   1.0000   0.5000  
0.45       0.5000   0.6667   0.0000   1.0000   1.0000   0.5000  
0.50       0.5230   0.6708   0.0742   0.7252   0.9719   0.5121   <--
0.55       0.5049   0.0722   0.9712   0.5025   0.0385   0.5722  
0.60       0.4976   0.0074   0.9914   0.4988   0.0037   0.3024  
0.65       0.4999   0.0020   0.9988   0.5000   0.0010   0.4603  
0.70       0.4997   0.0001   0.9994   0.4999   0.0001   0.0833  
0.75       0.4997   0.0001   0.9995   0.4999   0.0000   0.0588  
0.80       0.4998   0.0000   0.9996   0.4999   0.0000   0.0417  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.5230, F1=0.6708, Normal Recall=0.0742, Normal Precision=0.7252, Attack Recall=0.9719, Attack Precision=0.5121

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
0.15       0.9142   0.6897   0.9098   0.9944   0.9538   0.5402  
0.20       0.9211   0.7072   0.9175   0.9944   0.9535   0.5620  
0.25       0.9285   0.7272   0.9257   0.9944   0.9531   0.5878  
0.30       0.9339   0.7420   0.9321   0.9941   0.9505   0.6086  
0.35       0.9498   0.7888   0.9511   0.9928   0.9378   0.6806  
0.40       0.9517   0.7950   0.9533   0.9927   0.9372   0.6902  
0.45       0.9606   0.8247   0.9642   0.9917   0.9276   0.7424  
0.50       0.9638   0.8359   0.9686   0.9910   0.9211   0.7651  
0.55       0.9649   0.8396   0.9702   0.9906   0.9175   0.7739  
0.60       0.9661   0.8433   0.9721   0.9901   0.9121   0.7842  
0.65       0.9647   0.8332   0.9740   0.9866   0.8808   0.7904  
0.70       0.9638   0.8246   0.9763   0.9833   0.8510   0.7999  
0.75       0.9727   0.8541   0.9918   0.9781   0.8004   0.9155   <--
0.80       0.9720   0.8467   0.9940   0.9753   0.7737   0.9349  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.75
  At threshold 0.75: Accuracy=0.9727, F1=0.8541, Normal Recall=0.9918, Normal Precision=0.9781, Attack Recall=0.8004, Attack Precision=0.9155

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
0.15       0.9184   0.8236   0.9099   0.9871   0.9524   0.7255  
0.20       0.9245   0.8346   0.9177   0.9871   0.9519   0.7430  
0.25       0.9311   0.8468   0.9260   0.9871   0.9517   0.7628  
0.30       0.9357   0.8552   0.9323   0.9866   0.9492   0.7781  
0.35       0.9484   0.8789   0.9513   0.9836   0.9367   0.8278  
0.40       0.9499   0.8820   0.9534   0.9835   0.9359   0.8339  
0.45       0.9566   0.8951   0.9642   0.9812   0.9261   0.8661  
0.50       0.9590   0.8998   0.9687   0.9798   0.9203   0.8801  
0.55       0.9596   0.9007   0.9703   0.9790   0.9168   0.8852  
0.60       0.9600   0.9012   0.9722   0.9777   0.9115   0.8912   <--
0.65       0.9552   0.8869   0.9741   0.9700   0.8794   0.8946  
0.70       0.9510   0.8740   0.9764   0.9629   0.8495   0.8999  
0.75       0.9533   0.8724   0.9918   0.9517   0.7989   0.9608  
0.80       0.9496   0.8596   0.9940   0.9457   0.7718   0.9699  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.6
  At threshold 0.6: Accuracy=0.9600, F1=0.9012, Normal Recall=0.9722, Normal Precision=0.9777, Attack Recall=0.9115, Attack Precision=0.8912

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
0.15       0.9232   0.8816   0.9107   0.9781   0.9524   0.8205  
0.20       0.9285   0.8887   0.9185   0.9781   0.9519   0.8334  
0.25       0.9342   0.8966   0.9267   0.9781   0.9517   0.8476  
0.30       0.9378   0.9016   0.9330   0.9772   0.9492   0.8585  
0.35       0.9469   0.9137   0.9512   0.9723   0.9367   0.8917  
0.40       0.9482   0.9155   0.9534   0.9720   0.9359   0.8959  
0.45       0.9526   0.9215   0.9640   0.9682   0.9261   0.9168  
0.50       0.9540   0.9232   0.9685   0.9659   0.9202   0.9261   <--
0.55       0.9541   0.9229   0.9700   0.9645   0.9168   0.9292  
0.60       0.9538   0.9222   0.9720   0.9624   0.9115   0.9331  
0.65       0.9457   0.9067   0.9741   0.9496   0.8794   0.9357  
0.70       0.9382   0.8919   0.9762   0.9380   0.8495   0.9387  
0.75       0.9340   0.8790   0.9920   0.9200   0.7989   0.9771  
0.80       0.9274   0.8644   0.9940   0.9104   0.7718   0.9822  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.9540, F1=0.9232, Normal Recall=0.9685, Normal Precision=0.9659, Attack Recall=0.9202, Attack Precision=0.9261

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
0.15       0.9272   0.9128   0.9104   0.9663   0.9524   0.8763  
0.20       0.9318   0.9178   0.9184   0.9663   0.9519   0.8861  
0.25       0.9367   0.9233   0.9268   0.9664   0.9517   0.8965  
0.30       0.9395   0.9262   0.9330   0.9650   0.9492   0.9042  
0.35       0.9455   0.9322   0.9514   0.9575   0.9367   0.9278  
0.40       0.9465   0.9333   0.9535   0.9571   0.9359   0.9307  
0.45       0.9489   0.9355   0.9641   0.9514   0.9261   0.9450  
0.50       0.9492   0.9355   0.9685   0.9480   0.9203   0.9512   <--
0.55       0.9487   0.9346   0.9700   0.9459   0.9168   0.9532  
0.60       0.9478   0.9332   0.9720   0.9428   0.9115   0.9560  
0.65       0.9362   0.9169   0.9741   0.9238   0.8794   0.9578  
0.70       0.9256   0.9014   0.9764   0.9068   0.8495   0.9600  
0.75       0.9147   0.8822   0.9919   0.8809   0.7989   0.9850  
0.80       0.9050   0.8667   0.9939   0.8673   0.7718   0.9882  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.9492, F1=0.9355, Normal Recall=0.9685, Normal Precision=0.9480, Attack Recall=0.9203, Attack Precision=0.9512

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
0.15       0.9312   0.9326   0.9100   0.9503   0.9524   0.9136  
0.20       0.9349   0.9360   0.9179   0.9502   0.9519   0.9206  
0.25       0.9390   0.9397   0.9263   0.9504   0.9517   0.9281  
0.30       0.9409   0.9414   0.9326   0.9483   0.9492   0.9337  
0.35       0.9442   0.9438   0.9516   0.9377   0.9367   0.9509  
0.40       0.9449   0.9444   0.9539   0.9371   0.9359   0.9531   <--
0.45       0.9452   0.9441   0.9642   0.9288   0.9261   0.9628  
0.50       0.9446   0.9432   0.9689   0.9240   0.9203   0.9673  
0.55       0.9436   0.9420   0.9703   0.9210   0.9168   0.9687  
0.60       0.9419   0.9401   0.9723   0.9165   0.9115   0.9705  
0.65       0.9270   0.9233   0.9746   0.8899   0.8794   0.9719  
0.70       0.9131   0.9072   0.9767   0.8665   0.8495   0.9733  
0.75       0.8954   0.8842   0.9920   0.8314   0.7989   0.9900  
0.80       0.8828   0.8682   0.9938   0.8133   0.7718   0.9920  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9449, F1=0.9444, Normal Recall=0.9539, Normal Precision=0.9371, Attack Recall=0.9359, Attack Precision=0.9531

```

