# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e2_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-16 15:36:49 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6703 | 0.6663 | 0.6621 | 0.6588 | 0.6534 | 0.6507 | 0.6460 | 0.6414 | 0.6369 | 0.6339 | 0.6287 |
| QAT+Prune only | 0.8407 | 0.8559 | 0.8706 | 0.8863 | 0.9010 | 0.9151 | 0.9317 | 0.9461 | 0.9603 | 0.9758 | 0.9908 |
| QAT+PTQ | 0.8401 | 0.8555 | 0.8702 | 0.8859 | 0.9007 | 0.9147 | 0.9313 | 0.9459 | 0.9602 | 0.9756 | 0.9906 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8401 | 0.8555 | 0.8702 | 0.8859 | 0.9007 | 0.9147 | 0.9313 | 0.9459 | 0.9602 | 0.9756 | 0.9906 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2730 | 0.4267 | 0.5250 | 0.5920 | 0.6428 | 0.6806 | 0.7105 | 0.7347 | 0.7555 | 0.7720 |
| QAT+Prune only | 0.0000 | 0.5790 | 0.7539 | 0.8395 | 0.8890 | 0.9211 | 0.9457 | 0.9626 | 0.9756 | 0.9866 | 0.9954 |
| QAT+PTQ | 0.0000 | 0.5782 | 0.7533 | 0.8389 | 0.8886 | 0.9207 | 0.9454 | 0.9624 | 0.9755 | 0.9865 | 0.9953 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5782 | 0.7533 | 0.8389 | 0.8886 | 0.9207 | 0.9454 | 0.9624 | 0.9755 | 0.9865 | 0.9953 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6703 | 0.6707 | 0.6705 | 0.6717 | 0.6699 | 0.6726 | 0.6719 | 0.6710 | 0.6697 | 0.6807 | 0.0000 |
| QAT+Prune only | 0.8407 | 0.8409 | 0.8406 | 0.8416 | 0.8411 | 0.8395 | 0.8431 | 0.8417 | 0.8385 | 0.8412 | 0.0000 |
| QAT+PTQ | 0.8401 | 0.8405 | 0.8401 | 0.8410 | 0.8407 | 0.8389 | 0.8425 | 0.8415 | 0.8386 | 0.8404 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8401 | 0.8405 | 0.8401 | 0.8410 | 0.8407 | 0.8389 | 0.8425 | 0.8415 | 0.8386 | 0.8404 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6703 | 0.0000 | 0.0000 | 0.0000 | 0.6703 | 1.0000 |
| 90 | 10 | 299,940 | 0.6663 | 0.1745 | 0.6267 | 0.2730 | 0.6707 | 0.9418 |
| 80 | 20 | 291,350 | 0.6621 | 0.3229 | 0.6287 | 0.4267 | 0.6705 | 0.8784 |
| 70 | 30 | 194,230 | 0.6588 | 0.4507 | 0.6287 | 0.5250 | 0.6717 | 0.8084 |
| 60 | 40 | 145,675 | 0.6534 | 0.5594 | 0.6287 | 0.5920 | 0.6699 | 0.7302 |
| 50 | 50 | 116,540 | 0.6507 | 0.6576 | 0.6287 | 0.6428 | 0.6726 | 0.6443 |
| 40 | 60 | 97,115 | 0.6460 | 0.7419 | 0.6287 | 0.6806 | 0.6719 | 0.5468 |
| 30 | 70 | 83,240 | 0.6414 | 0.8168 | 0.6287 | 0.7105 | 0.6710 | 0.4364 |
| 20 | 80 | 72,835 | 0.6369 | 0.8839 | 0.6286 | 0.7347 | 0.6697 | 0.3108 |
| 10 | 90 | 64,740 | 0.6339 | 0.9466 | 0.6287 | 0.7555 | 0.6807 | 0.1692 |
| 0 | 100 | 58,270 | 0.6287 | 1.0000 | 0.6287 | 0.7720 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8407 | 0.0000 | 0.0000 | 0.0000 | 0.8407 | 1.0000 |
| 90 | 10 | 299,940 | 0.8559 | 0.4090 | 0.9908 | 0.5790 | 0.8409 | 0.9988 |
| 80 | 20 | 291,350 | 0.8706 | 0.6085 | 0.9908 | 0.7539 | 0.8406 | 0.9973 |
| 70 | 30 | 194,230 | 0.8863 | 0.7283 | 0.9908 | 0.8395 | 0.8416 | 0.9953 |
| 60 | 40 | 145,675 | 0.9010 | 0.8061 | 0.9908 | 0.8890 | 0.8411 | 0.9928 |
| 50 | 50 | 116,540 | 0.9151 | 0.8606 | 0.9908 | 0.9211 | 0.8395 | 0.9892 |
| 40 | 60 | 97,115 | 0.9317 | 0.9045 | 0.9908 | 0.9457 | 0.8431 | 0.9839 |
| 30 | 70 | 83,240 | 0.9461 | 0.9359 | 0.9908 | 0.9626 | 0.8417 | 0.9751 |
| 20 | 80 | 72,835 | 0.9603 | 0.9609 | 0.9908 | 0.9756 | 0.8385 | 0.9580 |
| 10 | 90 | 64,740 | 0.9758 | 0.9825 | 0.9908 | 0.9866 | 0.8412 | 0.9104 |
| 0 | 100 | 58,270 | 0.9908 | 1.0000 | 0.9908 | 0.9954 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8401 | 0.0000 | 0.0000 | 0.0000 | 0.8401 | 1.0000 |
| 90 | 10 | 299,940 | 0.8555 | 0.4083 | 0.9906 | 0.5782 | 0.8405 | 0.9988 |
| 80 | 20 | 291,350 | 0.8702 | 0.6077 | 0.9906 | 0.7533 | 0.8401 | 0.9972 |
| 70 | 30 | 194,230 | 0.8859 | 0.7275 | 0.9906 | 0.8389 | 0.8410 | 0.9952 |
| 60 | 40 | 145,675 | 0.9007 | 0.8057 | 0.9906 | 0.8886 | 0.8407 | 0.9926 |
| 50 | 50 | 116,540 | 0.9147 | 0.8601 | 0.9906 | 0.9207 | 0.8389 | 0.9889 |
| 40 | 60 | 97,115 | 0.9313 | 0.9041 | 0.9906 | 0.9454 | 0.8425 | 0.9835 |
| 30 | 70 | 83,240 | 0.9459 | 0.9358 | 0.9906 | 0.9624 | 0.8415 | 0.9745 |
| 20 | 80 | 72,835 | 0.9602 | 0.9609 | 0.9906 | 0.9755 | 0.8386 | 0.9570 |
| 10 | 90 | 64,740 | 0.9756 | 0.9824 | 0.9906 | 0.9865 | 0.8404 | 0.9083 |
| 0 | 100 | 58,270 | 0.9906 | 1.0000 | 0.9906 | 0.9953 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8401 | 0.0000 | 0.0000 | 0.0000 | 0.8401 | 1.0000 |
| 90 | 10 | 299,940 | 0.8555 | 0.4083 | 0.9906 | 0.5782 | 0.8405 | 0.9988 |
| 80 | 20 | 291,350 | 0.8702 | 0.6077 | 0.9906 | 0.7533 | 0.8401 | 0.9972 |
| 70 | 30 | 194,230 | 0.8859 | 0.7275 | 0.9906 | 0.8389 | 0.8410 | 0.9952 |
| 60 | 40 | 145,675 | 0.9007 | 0.8057 | 0.9906 | 0.8886 | 0.8407 | 0.9926 |
| 50 | 50 | 116,540 | 0.9147 | 0.8601 | 0.9906 | 0.9207 | 0.8389 | 0.9889 |
| 40 | 60 | 97,115 | 0.9313 | 0.9041 | 0.9906 | 0.9454 | 0.8425 | 0.9835 |
| 30 | 70 | 83,240 | 0.9459 | 0.9358 | 0.9906 | 0.9624 | 0.8415 | 0.9745 |
| 20 | 80 | 72,835 | 0.9602 | 0.9609 | 0.9906 | 0.9755 | 0.8386 | 0.9570 |
| 10 | 90 | 64,740 | 0.9756 | 0.9824 | 0.9906 | 0.9865 | 0.8404 | 0.9083 |
| 0 | 100 | 58,270 | 0.9906 | 1.0000 | 0.9906 | 0.9953 | 0.0000 | 0.0000 |


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
0.15       0.6664   0.2734   0.6707   0.9419   0.6277   0.1748   <--
0.20       0.6664   0.2734   0.6707   0.9419   0.6277   0.1748  
0.25       0.6664   0.2734   0.6707   0.9419   0.6277   0.1748  
0.30       0.6664   0.2734   0.6707   0.9419   0.6277   0.1748  
0.35       0.6664   0.2734   0.6707   0.9419   0.6277   0.1748  
0.40       0.6664   0.2734   0.6707   0.9419   0.6277   0.1748  
0.45       0.6664   0.2734   0.6707   0.9419   0.6277   0.1748  
0.50       0.6664   0.2734   0.6707   0.9419   0.6277   0.1748  
0.55       0.6664   0.2734   0.6707   0.9419   0.6277   0.1748  
0.60       0.6664   0.2734   0.6707   0.9419   0.6277   0.1748  
0.65       0.6664   0.2734   0.6707   0.9419   0.6277   0.1748  
0.70       0.6664   0.2734   0.6707   0.9419   0.6277   0.1748  
0.75       0.6664   0.2734   0.6707   0.9419   0.6277   0.1748  
0.80       0.6664   0.2734   0.6707   0.9419   0.6277   0.1748  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6664, F1=0.2734, Normal Recall=0.6707, Normal Precision=0.9419, Attack Recall=0.6277, Attack Precision=0.1748

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
0.15       0.6621   0.4267   0.6704   0.8784   0.6287   0.3229   <--
0.20       0.6621   0.4267   0.6704   0.8784   0.6287   0.3229  
0.25       0.6621   0.4267   0.6704   0.8784   0.6287   0.3229  
0.30       0.6621   0.4267   0.6704   0.8784   0.6287   0.3229  
0.35       0.6621   0.4267   0.6704   0.8784   0.6287   0.3229  
0.40       0.6621   0.4267   0.6704   0.8784   0.6287   0.3229  
0.45       0.6621   0.4267   0.6704   0.8784   0.6287   0.3229  
0.50       0.6621   0.4267   0.6704   0.8784   0.6287   0.3229  
0.55       0.6621   0.4267   0.6704   0.8784   0.6287   0.3229  
0.60       0.6621   0.4267   0.6704   0.8784   0.6287   0.3229  
0.65       0.6621   0.4267   0.6704   0.8784   0.6287   0.3229  
0.70       0.6621   0.4267   0.6704   0.8784   0.6287   0.3229  
0.75       0.6621   0.4267   0.6704   0.8784   0.6287   0.3229  
0.80       0.6621   0.4267   0.6704   0.8784   0.6287   0.3229  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6621, F1=0.4267, Normal Recall=0.6704, Normal Precision=0.8784, Attack Recall=0.6287, Attack Precision=0.3229

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
0.15       0.6576   0.5242   0.6700   0.8081   0.6287   0.4495   <--
0.20       0.6576   0.5242   0.6700   0.8081   0.6287   0.4495  
0.25       0.6576   0.5242   0.6700   0.8081   0.6287   0.4495  
0.30       0.6576   0.5242   0.6700   0.8081   0.6287   0.4495  
0.35       0.6576   0.5242   0.6700   0.8081   0.6287   0.4495  
0.40       0.6576   0.5242   0.6700   0.8081   0.6287   0.4495  
0.45       0.6576   0.5242   0.6700   0.8081   0.6287   0.4495  
0.50       0.6576   0.5242   0.6700   0.8081   0.6287   0.4495  
0.55       0.6576   0.5242   0.6700   0.8081   0.6287   0.4495  
0.60       0.6576   0.5242   0.6700   0.8081   0.6287   0.4495  
0.65       0.6576   0.5242   0.6700   0.8081   0.6287   0.4495  
0.70       0.6576   0.5242   0.6700   0.8081   0.6287   0.4495  
0.75       0.6576   0.5242   0.6700   0.8081   0.6287   0.4495  
0.80       0.6576   0.5242   0.6700   0.8081   0.6287   0.4495  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6576, F1=0.5242, Normal Recall=0.6700, Normal Precision=0.8081, Attack Recall=0.6287, Attack Precision=0.4495

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
0.15       0.6543   0.5926   0.6714   0.7306   0.6287   0.5605   <--
0.20       0.6543   0.5926   0.6714   0.7306   0.6287   0.5605  
0.25       0.6543   0.5926   0.6714   0.7306   0.6287   0.5605  
0.30       0.6543   0.5926   0.6714   0.7306   0.6287   0.5605  
0.35       0.6543   0.5926   0.6714   0.7306   0.6287   0.5605  
0.40       0.6543   0.5926   0.6714   0.7306   0.6287   0.5605  
0.45       0.6543   0.5926   0.6714   0.7306   0.6287   0.5605  
0.50       0.6543   0.5926   0.6714   0.7306   0.6287   0.5605  
0.55       0.6543   0.5926   0.6714   0.7306   0.6287   0.5605  
0.60       0.6543   0.5926   0.6714   0.7306   0.6287   0.5605  
0.65       0.6543   0.5926   0.6714   0.7306   0.6287   0.5605  
0.70       0.6543   0.5926   0.6714   0.7306   0.6287   0.5605  
0.75       0.6543   0.5926   0.6714   0.7306   0.6287   0.5605  
0.80       0.6543   0.5926   0.6714   0.7306   0.6287   0.5605  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6543, F1=0.5926, Normal Recall=0.6714, Normal Precision=0.7306, Attack Recall=0.6287, Attack Precision=0.5605

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
0.15       0.6506   0.6427   0.6725   0.6443   0.6287   0.6575   <--
0.20       0.6506   0.6427   0.6725   0.6443   0.6287   0.6575  
0.25       0.6506   0.6427   0.6725   0.6443   0.6287   0.6575  
0.30       0.6506   0.6427   0.6725   0.6443   0.6287   0.6575  
0.35       0.6506   0.6427   0.6725   0.6443   0.6287   0.6575  
0.40       0.6506   0.6427   0.6725   0.6443   0.6287   0.6575  
0.45       0.6506   0.6427   0.6725   0.6443   0.6287   0.6575  
0.50       0.6506   0.6427   0.6725   0.6443   0.6287   0.6575  
0.55       0.6506   0.6427   0.6725   0.6443   0.6287   0.6575  
0.60       0.6506   0.6427   0.6725   0.6443   0.6287   0.6575  
0.65       0.6506   0.6427   0.6725   0.6443   0.6287   0.6575  
0.70       0.6506   0.6427   0.6725   0.6443   0.6287   0.6575  
0.75       0.6506   0.6427   0.6725   0.6443   0.6287   0.6575  
0.80       0.6506   0.6427   0.6725   0.6443   0.6287   0.6575  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6506, F1=0.6427, Normal Recall=0.6725, Normal Precision=0.6443, Attack Recall=0.6287, Attack Precision=0.6575

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
0.15       0.8560   0.5793   0.8409   0.9989   0.9914   0.4092   <--
0.20       0.8560   0.5793   0.8409   0.9989   0.9914   0.4092  
0.25       0.8560   0.5793   0.8409   0.9989   0.9914   0.4092  
0.30       0.8560   0.5793   0.8409   0.9989   0.9914   0.4092  
0.35       0.8560   0.5793   0.8409   0.9989   0.9914   0.4092  
0.40       0.8560   0.5793   0.8409   0.9989   0.9914   0.4092  
0.45       0.8560   0.5793   0.8409   0.9989   0.9914   0.4092  
0.50       0.8560   0.5793   0.8409   0.9989   0.9914   0.4092  
0.55       0.8560   0.5793   0.8409   0.9989   0.9914   0.4092  
0.60       0.8560   0.5793   0.8409   0.9989   0.9914   0.4092  
0.65       0.8560   0.5793   0.8409   0.9989   0.9914   0.4092  
0.70       0.8560   0.5793   0.8409   0.9989   0.9914   0.4092  
0.75       0.8560   0.5793   0.8409   0.9989   0.9914   0.4092  
0.80       0.8560   0.5793   0.8409   0.9989   0.9914   0.4092  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8560, F1=0.5793, Normal Recall=0.8409, Normal Precision=0.9989, Attack Recall=0.9914, Attack Precision=0.4092

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
0.15       0.8713   0.7548   0.8414   0.9973   0.9908   0.6096   <--
0.20       0.8713   0.7548   0.8414   0.9973   0.9908   0.6096  
0.25       0.8713   0.7548   0.8414   0.9973   0.9908   0.6096  
0.30       0.8713   0.7548   0.8414   0.9973   0.9908   0.6096  
0.35       0.8713   0.7548   0.8414   0.9973   0.9908   0.6096  
0.40       0.8713   0.7548   0.8414   0.9973   0.9908   0.6096  
0.45       0.8713   0.7548   0.8414   0.9973   0.9908   0.6096  
0.50       0.8713   0.7548   0.8414   0.9973   0.9908   0.6096  
0.55       0.8713   0.7548   0.8414   0.9973   0.9908   0.6096  
0.60       0.8713   0.7548   0.8414   0.9973   0.9908   0.6096  
0.65       0.8713   0.7548   0.8414   0.9973   0.9908   0.6096  
0.70       0.8713   0.7548   0.8414   0.9973   0.9908   0.6096  
0.75       0.8713   0.7548   0.8414   0.9973   0.9908   0.6096  
0.80       0.8713   0.7548   0.8414   0.9973   0.9908   0.6096  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8713, F1=0.7548, Normal Recall=0.8414, Normal Precision=0.9973, Attack Recall=0.9908, Attack Precision=0.6096

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
0.15       0.8863   0.8394   0.8415   0.9953   0.9908   0.7281   <--
0.20       0.8863   0.8394   0.8415   0.9953   0.9908   0.7281  
0.25       0.8863   0.8394   0.8415   0.9953   0.9908   0.7281  
0.30       0.8863   0.8394   0.8415   0.9953   0.9908   0.7281  
0.35       0.8863   0.8394   0.8415   0.9953   0.9908   0.7281  
0.40       0.8863   0.8394   0.8415   0.9953   0.9908   0.7281  
0.45       0.8863   0.8394   0.8415   0.9953   0.9908   0.7281  
0.50       0.8863   0.8394   0.8415   0.9953   0.9908   0.7281  
0.55       0.8863   0.8394   0.8415   0.9953   0.9908   0.7281  
0.60       0.8863   0.8394   0.8415   0.9953   0.9908   0.7281  
0.65       0.8863   0.8394   0.8415   0.9953   0.9908   0.7281  
0.70       0.8863   0.8394   0.8415   0.9953   0.9908   0.7281  
0.75       0.8863   0.8394   0.8415   0.9953   0.9908   0.7281  
0.80       0.8863   0.8394   0.8415   0.9953   0.9908   0.7281  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8863, F1=0.8394, Normal Recall=0.8415, Normal Precision=0.9953, Attack Recall=0.9908, Attack Precision=0.7281

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
0.15       0.9010   0.8890   0.8411   0.9928   0.9908   0.8061   <--
0.20       0.9010   0.8890   0.8411   0.9928   0.9908   0.8061  
0.25       0.9010   0.8890   0.8411   0.9928   0.9908   0.8061  
0.30       0.9010   0.8890   0.8411   0.9928   0.9908   0.8061  
0.35       0.9010   0.8890   0.8411   0.9928   0.9908   0.8061  
0.40       0.9010   0.8890   0.8411   0.9928   0.9908   0.8061  
0.45       0.9010   0.8890   0.8411   0.9928   0.9908   0.8061  
0.50       0.9010   0.8890   0.8411   0.9928   0.9908   0.8061  
0.55       0.9010   0.8890   0.8411   0.9928   0.9908   0.8061  
0.60       0.9010   0.8890   0.8411   0.9928   0.9908   0.8061  
0.65       0.9010   0.8890   0.8411   0.9928   0.9908   0.8061  
0.70       0.9010   0.8890   0.8411   0.9928   0.9908   0.8061  
0.75       0.9010   0.8890   0.8411   0.9928   0.9908   0.8061  
0.80       0.9010   0.8890   0.8411   0.9928   0.9908   0.8061  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9010, F1=0.8890, Normal Recall=0.8411, Normal Precision=0.9928, Attack Recall=0.9908, Attack Precision=0.8061

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
0.15       0.9156   0.9215   0.8403   0.9892   0.9908   0.8612   <--
0.20       0.9156   0.9215   0.8403   0.9892   0.9908   0.8612  
0.25       0.9156   0.9215   0.8403   0.9892   0.9908   0.8612  
0.30       0.9156   0.9215   0.8403   0.9892   0.9908   0.8612  
0.35       0.9156   0.9215   0.8403   0.9892   0.9908   0.8612  
0.40       0.9156   0.9215   0.8403   0.9892   0.9908   0.8612  
0.45       0.9156   0.9215   0.8403   0.9892   0.9908   0.8612  
0.50       0.9156   0.9215   0.8403   0.9892   0.9908   0.8612  
0.55       0.9156   0.9215   0.8403   0.9892   0.9908   0.8612  
0.60       0.9156   0.9215   0.8403   0.9892   0.9908   0.8612  
0.65       0.9156   0.9215   0.8403   0.9892   0.9908   0.8612  
0.70       0.9156   0.9215   0.8403   0.9892   0.9908   0.8612  
0.75       0.9156   0.9215   0.8403   0.9892   0.9908   0.8612  
0.80       0.9156   0.9215   0.8403   0.9892   0.9908   0.8612  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9156, F1=0.9215, Normal Recall=0.8403, Normal Precision=0.9892, Attack Recall=0.9908, Attack Precision=0.8612

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
0.15       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084   <--
0.20       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.25       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.30       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.35       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.40       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.45       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.50       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.55       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.60       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.65       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.70       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.75       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.80       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8556, F1=0.5785, Normal Recall=0.8405, Normal Precision=0.9988, Attack Recall=0.9912, Attack Precision=0.4084

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
0.15       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088   <--
0.20       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.25       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.30       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.35       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.40       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.45       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.50       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.55       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.60       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.65       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.70       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.75       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.80       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8708, F1=0.7541, Normal Recall=0.8409, Normal Precision=0.9972, Attack Recall=0.9906, Attack Precision=0.6088

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
0.15       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275   <--
0.20       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.25       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.30       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.35       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.40       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.45       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.50       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.55       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.60       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.65       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.70       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.75       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.80       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8859, F1=0.8389, Normal Recall=0.8410, Normal Precision=0.9952, Attack Recall=0.9906, Attack Precision=0.7275

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
0.15       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054   <--
0.20       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.25       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.30       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.35       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.40       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.45       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.50       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.55       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.60       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.65       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.70       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.75       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.80       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9005, F1=0.8885, Normal Recall=0.8405, Normal Precision=0.9926, Attack Recall=0.9906, Attack Precision=0.8054

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
0.15       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608   <--
0.20       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.25       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.30       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.35       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.40       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.45       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.50       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.55       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.60       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.65       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.70       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.75       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.80       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9152, F1=0.9211, Normal Recall=0.8398, Normal Precision=0.9889, Attack Recall=0.9906, Attack Precision=0.8608

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
0.15       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084   <--
0.20       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.25       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.30       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.35       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.40       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.45       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.50       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.55       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.60       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.65       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.70       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.75       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
0.80       0.8556   0.5785   0.8405   0.9988   0.9912   0.4084  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8556, F1=0.5785, Normal Recall=0.8405, Normal Precision=0.9988, Attack Recall=0.9912, Attack Precision=0.4084

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
0.15       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088   <--
0.20       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.25       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.30       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.35       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.40       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.45       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.50       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.55       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.60       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.65       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.70       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.75       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
0.80       0.8708   0.7541   0.8409   0.9972   0.9906   0.6088  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8708, F1=0.7541, Normal Recall=0.8409, Normal Precision=0.9972, Attack Recall=0.9906, Attack Precision=0.6088

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
0.15       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275   <--
0.20       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.25       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.30       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.35       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.40       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.45       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.50       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.55       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.60       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.65       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.70       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.75       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
0.80       0.8859   0.8389   0.8410   0.9952   0.9906   0.7275  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8859, F1=0.8389, Normal Recall=0.8410, Normal Precision=0.9952, Attack Recall=0.9906, Attack Precision=0.7275

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
0.15       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054   <--
0.20       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.25       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.30       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.35       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.40       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.45       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.50       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.55       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.60       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.65       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.70       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.75       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
0.80       0.9005   0.8885   0.8405   0.9926   0.9906   0.8054  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9005, F1=0.8885, Normal Recall=0.8405, Normal Precision=0.9926, Attack Recall=0.9906, Attack Precision=0.8054

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
0.15       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608   <--
0.20       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.25       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.30       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.35       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.40       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.45       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.50       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.55       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.60       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.65       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.70       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.75       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
0.80       0.9152   0.9211   0.8398   0.9889   0.9906   0.8608  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9152, F1=0.9211, Normal Recall=0.8398, Normal Precision=0.9889, Attack Recall=0.9906, Attack Precision=0.8608

```

