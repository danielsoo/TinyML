# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e2_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-15 00:19:04 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8629 | 0.8526 | 0.8423 | 0.8326 | 0.8213 | 0.8125 | 0.8022 | 0.7907 | 0.7804 | 0.7711 | 0.7606 |
| QAT+Prune only | 0.9725 | 0.9498 | 0.9277 | 0.9061 | 0.8837 | 0.8615 | 0.8399 | 0.8175 | 0.7959 | 0.7736 | 0.7517 |
| QAT+PTQ | 0.9715 | 0.9488 | 0.9269 | 0.9055 | 0.8833 | 0.8613 | 0.8398 | 0.8177 | 0.7962 | 0.7741 | 0.7523 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9715 | 0.9488 | 0.9269 | 0.9055 | 0.8833 | 0.8613 | 0.8398 | 0.8177 | 0.7962 | 0.7741 | 0.7523 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.5071 | 0.6586 | 0.7316 | 0.7730 | 0.8022 | 0.8219 | 0.8357 | 0.8471 | 0.8568 | 0.8640 |
| QAT+Prune only | 0.0000 | 0.7494 | 0.8062 | 0.8276 | 0.8380 | 0.8445 | 0.8492 | 0.8522 | 0.8549 | 0.8567 | 0.8582 |
| QAT+PTQ | 0.0000 | 0.7459 | 0.8046 | 0.8269 | 0.8376 | 0.8443 | 0.8493 | 0.8524 | 0.8552 | 0.8570 | 0.8586 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.7459 | 0.8046 | 0.8269 | 0.8376 | 0.8443 | 0.8493 | 0.8524 | 0.8552 | 0.8570 | 0.8586 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8629 | 0.8631 | 0.8627 | 0.8635 | 0.8618 | 0.8644 | 0.8646 | 0.8609 | 0.8595 | 0.8658 | 0.0000 |
| QAT+Prune only | 0.9725 | 0.9718 | 0.9717 | 0.9722 | 0.9718 | 0.9714 | 0.9721 | 0.9712 | 0.9726 | 0.9710 | 0.0000 |
| QAT+PTQ | 0.9715 | 0.9707 | 0.9706 | 0.9712 | 0.9706 | 0.9703 | 0.9712 | 0.9704 | 0.9719 | 0.9703 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9715 | 0.9707 | 0.9706 | 0.9712 | 0.9706 | 0.9703 | 0.9712 | 0.9704 | 0.9719 | 0.9703 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8629 | 0.0000 | 0.0000 | 0.0000 | 0.8629 | 1.0000 |
| 90 | 10 | 299,940 | 0.8526 | 0.3810 | 0.7582 | 0.5071 | 0.8631 | 0.9698 |
| 80 | 20 | 291,350 | 0.8423 | 0.5808 | 0.7606 | 0.6586 | 0.8627 | 0.9351 |
| 70 | 30 | 194,230 | 0.8326 | 0.7048 | 0.7606 | 0.7316 | 0.8635 | 0.8938 |
| 60 | 40 | 145,675 | 0.8213 | 0.7858 | 0.7606 | 0.7730 | 0.8618 | 0.8437 |
| 50 | 50 | 116,540 | 0.8125 | 0.8487 | 0.7606 | 0.8022 | 0.8644 | 0.7831 |
| 40 | 60 | 97,115 | 0.8022 | 0.8939 | 0.7606 | 0.8219 | 0.8646 | 0.7065 |
| 30 | 70 | 83,240 | 0.7907 | 0.9273 | 0.7606 | 0.8357 | 0.8609 | 0.6065 |
| 20 | 80 | 72,835 | 0.7804 | 0.9559 | 0.7606 | 0.8471 | 0.8595 | 0.4730 |
| 10 | 90 | 64,740 | 0.7711 | 0.9808 | 0.7606 | 0.8568 | 0.8658 | 0.2866 |
| 0 | 100 | 58,270 | 0.7606 | 1.0000 | 0.7606 | 0.8640 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9725 | 0.0000 | 0.0000 | 0.0000 | 0.9725 | 1.0000 |
| 90 | 10 | 299,940 | 0.9498 | 0.7475 | 0.7513 | 0.7494 | 0.9718 | 0.9724 |
| 80 | 20 | 291,350 | 0.9277 | 0.8693 | 0.7517 | 0.8062 | 0.9717 | 0.9400 |
| 70 | 30 | 194,230 | 0.9061 | 0.9207 | 0.7517 | 0.8276 | 0.9722 | 0.9013 |
| 60 | 40 | 145,675 | 0.8837 | 0.9466 | 0.7517 | 0.8380 | 0.9718 | 0.8544 |
| 50 | 50 | 116,540 | 0.8615 | 0.9634 | 0.7517 | 0.8445 | 0.9714 | 0.7964 |
| 40 | 60 | 97,115 | 0.8399 | 0.9759 | 0.7517 | 0.8492 | 0.9721 | 0.7230 |
| 30 | 70 | 83,240 | 0.8175 | 0.9838 | 0.7517 | 0.8522 | 0.9712 | 0.6263 |
| 20 | 80 | 72,835 | 0.7959 | 0.9910 | 0.7517 | 0.8549 | 0.9726 | 0.4948 |
| 10 | 90 | 64,740 | 0.7736 | 0.9957 | 0.7517 | 0.8567 | 0.9710 | 0.3029 |
| 0 | 100 | 58,270 | 0.7517 | 1.0000 | 0.7517 | 0.8582 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9715 | 0.0000 | 0.0000 | 0.0000 | 0.9715 | 1.0000 |
| 90 | 10 | 299,940 | 0.9488 | 0.7402 | 0.7517 | 0.7459 | 0.9707 | 0.9724 |
| 80 | 20 | 291,350 | 0.9269 | 0.8648 | 0.7523 | 0.8046 | 0.9706 | 0.9400 |
| 70 | 30 | 194,230 | 0.9055 | 0.9180 | 0.7523 | 0.8269 | 0.9712 | 0.9015 |
| 60 | 40 | 145,675 | 0.8833 | 0.9447 | 0.7523 | 0.8376 | 0.9706 | 0.8546 |
| 50 | 50 | 116,540 | 0.8613 | 0.9620 | 0.7523 | 0.8443 | 0.9703 | 0.7966 |
| 40 | 60 | 97,115 | 0.8398 | 0.9751 | 0.7523 | 0.8493 | 0.9712 | 0.7233 |
| 30 | 70 | 83,240 | 0.8177 | 0.9834 | 0.7523 | 0.8524 | 0.9704 | 0.6267 |
| 20 | 80 | 72,835 | 0.7962 | 0.9908 | 0.7523 | 0.8552 | 0.9719 | 0.4952 |
| 10 | 90 | 64,740 | 0.7741 | 0.9956 | 0.7523 | 0.8570 | 0.9703 | 0.3032 |
| 0 | 100 | 58,270 | 0.7523 | 1.0000 | 0.7523 | 0.8586 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9715 | 0.0000 | 0.0000 | 0.0000 | 0.9715 | 1.0000 |
| 90 | 10 | 299,940 | 0.9488 | 0.7402 | 0.7517 | 0.7459 | 0.9707 | 0.9724 |
| 80 | 20 | 291,350 | 0.9269 | 0.8648 | 0.7523 | 0.8046 | 0.9706 | 0.9400 |
| 70 | 30 | 194,230 | 0.9055 | 0.9180 | 0.7523 | 0.8269 | 0.9712 | 0.9015 |
| 60 | 40 | 145,675 | 0.8833 | 0.9447 | 0.7523 | 0.8376 | 0.9706 | 0.8546 |
| 50 | 50 | 116,540 | 0.8613 | 0.9620 | 0.7523 | 0.8443 | 0.9703 | 0.7966 |
| 40 | 60 | 97,115 | 0.8398 | 0.9751 | 0.7523 | 0.8493 | 0.9712 | 0.7233 |
| 30 | 70 | 83,240 | 0.8177 | 0.9834 | 0.7523 | 0.8524 | 0.9704 | 0.6267 |
| 20 | 80 | 72,835 | 0.7962 | 0.9908 | 0.7523 | 0.8552 | 0.9719 | 0.4952 |
| 10 | 90 | 64,740 | 0.7741 | 0.9956 | 0.7523 | 0.8570 | 0.9703 | 0.3032 |
| 0 | 100 | 58,270 | 0.7523 | 1.0000 | 0.7523 | 0.8586 | 0.0000 | 0.0000 |


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
0.15       0.8528   0.5080   0.8631   0.9700   0.7600   0.3815   <--
0.20       0.8528   0.5080   0.8631   0.9700   0.7600   0.3815  
0.25       0.8528   0.5080   0.8631   0.9700   0.7600   0.3815  
0.30       0.8528   0.5080   0.8631   0.9700   0.7600   0.3815  
0.35       0.8528   0.5080   0.8631   0.9700   0.7600   0.3815  
0.40       0.8528   0.5080   0.8631   0.9700   0.7600   0.3815  
0.45       0.8528   0.5080   0.8631   0.9700   0.7600   0.3815  
0.50       0.8528   0.5080   0.8631   0.9700   0.7600   0.3815  
0.55       0.8528   0.5080   0.8631   0.9700   0.7600   0.3815  
0.60       0.8528   0.5080   0.8631   0.9700   0.7600   0.3815  
0.65       0.8528   0.5080   0.8631   0.9700   0.7600   0.3815  
0.70       0.8528   0.5080   0.8631   0.9700   0.7600   0.3815  
0.75       0.8528   0.5080   0.8631   0.9700   0.7600   0.3815  
0.80       0.8528   0.5080   0.8631   0.9700   0.7600   0.3815  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8528, F1=0.5080, Normal Recall=0.8631, Normal Precision=0.9700, Attack Recall=0.7600, Attack Precision=0.3815

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
0.15       0.8426   0.6590   0.8631   0.9352   0.7606   0.5814   <--
0.20       0.8426   0.6590   0.8631   0.9352   0.7606   0.5814  
0.25       0.8426   0.6590   0.8631   0.9352   0.7606   0.5814  
0.30       0.8426   0.6590   0.8631   0.9352   0.7606   0.5814  
0.35       0.8426   0.6590   0.8631   0.9352   0.7606   0.5814  
0.40       0.8426   0.6590   0.8631   0.9352   0.7606   0.5814  
0.45       0.8426   0.6590   0.8631   0.9352   0.7606   0.5814  
0.50       0.8426   0.6590   0.8631   0.9352   0.7606   0.5814  
0.55       0.8426   0.6590   0.8631   0.9352   0.7606   0.5814  
0.60       0.8426   0.6590   0.8631   0.9352   0.7606   0.5814  
0.65       0.8426   0.6590   0.8631   0.9352   0.7606   0.5814  
0.70       0.8426   0.6590   0.8631   0.9352   0.7606   0.5814  
0.75       0.8426   0.6590   0.8631   0.9352   0.7606   0.5814  
0.80       0.8426   0.6590   0.8631   0.9352   0.7606   0.5814  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8426, F1=0.6590, Normal Recall=0.8631, Normal Precision=0.9352, Attack Recall=0.7606, Attack Precision=0.5814

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
0.15       0.8328   0.7319   0.8638   0.8938   0.7606   0.7052   <--
0.20       0.8328   0.7319   0.8638   0.8938   0.7606   0.7052  
0.25       0.8328   0.7319   0.8638   0.8938   0.7606   0.7052  
0.30       0.8328   0.7319   0.8638   0.8938   0.7606   0.7052  
0.35       0.8328   0.7319   0.8638   0.8938   0.7606   0.7052  
0.40       0.8328   0.7319   0.8638   0.8938   0.7606   0.7052  
0.45       0.8328   0.7319   0.8638   0.8938   0.7606   0.7052  
0.50       0.8328   0.7319   0.8638   0.8938   0.7606   0.7052  
0.55       0.8328   0.7319   0.8638   0.8938   0.7606   0.7052  
0.60       0.8328   0.7319   0.8638   0.8938   0.7606   0.7052  
0.65       0.8328   0.7319   0.8638   0.8938   0.7606   0.7052  
0.70       0.8328   0.7319   0.8638   0.8938   0.7606   0.7052  
0.75       0.8328   0.7319   0.8638   0.8938   0.7606   0.7052  
0.80       0.8328   0.7319   0.8638   0.8938   0.7606   0.7052  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8328, F1=0.7319, Normal Recall=0.8638, Normal Precision=0.8938, Attack Recall=0.7606, Attack Precision=0.7052

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
0.15       0.8220   0.7737   0.8630   0.8439   0.7606   0.7872   <--
0.20       0.8220   0.7737   0.8630   0.8439   0.7606   0.7872  
0.25       0.8220   0.7737   0.8630   0.8439   0.7606   0.7872  
0.30       0.8220   0.7737   0.8630   0.8439   0.7606   0.7872  
0.35       0.8220   0.7737   0.8630   0.8439   0.7606   0.7872  
0.40       0.8220   0.7737   0.8630   0.8439   0.7606   0.7872  
0.45       0.8220   0.7737   0.8630   0.8439   0.7606   0.7872  
0.50       0.8220   0.7737   0.8630   0.8439   0.7606   0.7872  
0.55       0.8220   0.7737   0.8630   0.8439   0.7606   0.7872  
0.60       0.8220   0.7737   0.8630   0.8439   0.7606   0.7872  
0.65       0.8220   0.7737   0.8630   0.8439   0.7606   0.7872  
0.70       0.8220   0.7737   0.8630   0.8439   0.7606   0.7872  
0.75       0.8220   0.7737   0.8630   0.8439   0.7606   0.7872  
0.80       0.8220   0.7737   0.8630   0.8439   0.7606   0.7872  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8220, F1=0.7737, Normal Recall=0.8630, Normal Precision=0.8439, Attack Recall=0.7606, Attack Precision=0.7872

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
0.15       0.8116   0.8015   0.8627   0.7828   0.7606   0.8471   <--
0.20       0.8116   0.8015   0.8627   0.7828   0.7606   0.8471  
0.25       0.8116   0.8015   0.8627   0.7828   0.7606   0.8471  
0.30       0.8116   0.8015   0.8627   0.7828   0.7606   0.8471  
0.35       0.8116   0.8015   0.8627   0.7828   0.7606   0.8471  
0.40       0.8116   0.8015   0.8627   0.7828   0.7606   0.8471  
0.45       0.8116   0.8015   0.8627   0.7828   0.7606   0.8471  
0.50       0.8116   0.8015   0.8627   0.7828   0.7606   0.8471  
0.55       0.8116   0.8015   0.8627   0.7828   0.7606   0.8471  
0.60       0.8116   0.8015   0.8627   0.7828   0.7606   0.8471  
0.65       0.8116   0.8015   0.8627   0.7828   0.7606   0.8471  
0.70       0.8116   0.8015   0.8627   0.7828   0.7606   0.8471  
0.75       0.8116   0.8015   0.8627   0.7828   0.7606   0.8471  
0.80       0.8116   0.8015   0.8627   0.7828   0.7606   0.8471  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8116, F1=0.8015, Normal Recall=0.8627, Normal Precision=0.7828, Attack Recall=0.7606, Attack Precision=0.8471

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
0.15       0.9499   0.7504   0.9718   0.9725   0.7529   0.7479   <--
0.20       0.9499   0.7504   0.9718   0.9725   0.7529   0.7479  
0.25       0.9499   0.7504   0.9718   0.9725   0.7529   0.7479  
0.30       0.9499   0.7504   0.9718   0.9725   0.7529   0.7479  
0.35       0.9499   0.7504   0.9718   0.9725   0.7529   0.7479  
0.40       0.9499   0.7504   0.9718   0.9725   0.7529   0.7479  
0.45       0.9499   0.7504   0.9718   0.9725   0.7529   0.7479  
0.50       0.9499   0.7504   0.9718   0.9725   0.7529   0.7479  
0.55       0.9499   0.7504   0.9718   0.9725   0.7529   0.7479  
0.60       0.9499   0.7504   0.9718   0.9725   0.7529   0.7479  
0.65       0.9499   0.7504   0.9718   0.9725   0.7529   0.7479  
0.70       0.9499   0.7504   0.9718   0.9725   0.7529   0.7479  
0.75       0.9499   0.7504   0.9718   0.9725   0.7529   0.7479  
0.80       0.9499   0.7504   0.9718   0.9725   0.7529   0.7479  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9499, F1=0.7504, Normal Recall=0.9718, Normal Precision=0.9725, Attack Recall=0.7529, Attack Precision=0.7479

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
0.15       0.9280   0.8068   0.9721   0.9400   0.7517   0.8706   <--
0.20       0.9280   0.8068   0.9721   0.9400   0.7517   0.8706  
0.25       0.9280   0.8068   0.9721   0.9400   0.7517   0.8706  
0.30       0.9280   0.8068   0.9721   0.9400   0.7517   0.8706  
0.35       0.9280   0.8068   0.9721   0.9400   0.7517   0.8706  
0.40       0.9280   0.8068   0.9721   0.9400   0.7517   0.8706  
0.45       0.9280   0.8068   0.9721   0.9400   0.7517   0.8706  
0.50       0.9280   0.8068   0.9721   0.9400   0.7517   0.8706  
0.55       0.9280   0.8068   0.9721   0.9400   0.7517   0.8706  
0.60       0.9280   0.8068   0.9721   0.9400   0.7517   0.8706  
0.65       0.9280   0.8068   0.9721   0.9400   0.7517   0.8706  
0.70       0.9280   0.8068   0.9721   0.9400   0.7517   0.8706  
0.75       0.9280   0.8068   0.9721   0.9400   0.7517   0.8706  
0.80       0.9280   0.8068   0.9721   0.9400   0.7517   0.8706  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9280, F1=0.8068, Normal Recall=0.9721, Normal Precision=0.9400, Attack Recall=0.7517, Attack Precision=0.8706

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
0.15       0.9060   0.8276   0.9722   0.9013   0.7517   0.9205   <--
0.20       0.9060   0.8276   0.9722   0.9013   0.7517   0.9205  
0.25       0.9060   0.8276   0.9722   0.9013   0.7517   0.9205  
0.30       0.9060   0.8276   0.9722   0.9013   0.7517   0.9205  
0.35       0.9060   0.8276   0.9722   0.9013   0.7517   0.9205  
0.40       0.9060   0.8276   0.9722   0.9013   0.7517   0.9205  
0.45       0.9060   0.8276   0.9722   0.9013   0.7517   0.9205  
0.50       0.9060   0.8276   0.9722   0.9013   0.7517   0.9205  
0.55       0.9060   0.8276   0.9722   0.9013   0.7517   0.9205  
0.60       0.9060   0.8276   0.9722   0.9013   0.7517   0.9205  
0.65       0.9060   0.8276   0.9722   0.9013   0.7517   0.9205  
0.70       0.9060   0.8276   0.9722   0.9013   0.7517   0.9205  
0.75       0.9060   0.8276   0.9722   0.9013   0.7517   0.9205  
0.80       0.9060   0.8276   0.9722   0.9013   0.7517   0.9205  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9060, F1=0.8276, Normal Recall=0.9722, Normal Precision=0.9013, Attack Recall=0.7517, Attack Precision=0.9205

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
0.15       0.8843   0.8386   0.9727   0.8546   0.7517   0.9483   <--
0.20       0.8843   0.8386   0.9727   0.8546   0.7517   0.9483  
0.25       0.8843   0.8386   0.9727   0.8546   0.7517   0.9483  
0.30       0.8843   0.8386   0.9727   0.8546   0.7517   0.9483  
0.35       0.8843   0.8386   0.9727   0.8546   0.7517   0.9483  
0.40       0.8843   0.8386   0.9727   0.8546   0.7517   0.9483  
0.45       0.8843   0.8386   0.9727   0.8546   0.7517   0.9483  
0.50       0.8843   0.8386   0.9727   0.8546   0.7517   0.9483  
0.55       0.8843   0.8386   0.9727   0.8546   0.7517   0.9483  
0.60       0.8843   0.8386   0.9727   0.8546   0.7517   0.9483  
0.65       0.8843   0.8386   0.9727   0.8546   0.7517   0.9483  
0.70       0.8843   0.8386   0.9727   0.8546   0.7517   0.9483  
0.75       0.8843   0.8386   0.9727   0.8546   0.7517   0.9483  
0.80       0.8843   0.8386   0.9727   0.8546   0.7517   0.9483  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8843, F1=0.8386, Normal Recall=0.9727, Normal Precision=0.8546, Attack Recall=0.7517, Attack Precision=0.9483

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
0.15       0.8616   0.8446   0.9716   0.7965   0.7517   0.9636   <--
0.20       0.8616   0.8446   0.9716   0.7965   0.7517   0.9636  
0.25       0.8616   0.8446   0.9716   0.7965   0.7517   0.9636  
0.30       0.8616   0.8446   0.9716   0.7965   0.7517   0.9636  
0.35       0.8616   0.8446   0.9716   0.7965   0.7517   0.9636  
0.40       0.8616   0.8446   0.9716   0.7965   0.7517   0.9636  
0.45       0.8616   0.8446   0.9716   0.7965   0.7517   0.9636  
0.50       0.8616   0.8446   0.9716   0.7965   0.7517   0.9636  
0.55       0.8616   0.8446   0.9716   0.7965   0.7517   0.9636  
0.60       0.8616   0.8446   0.9716   0.7965   0.7517   0.9636  
0.65       0.8616   0.8446   0.9716   0.7965   0.7517   0.9636  
0.70       0.8616   0.8446   0.9716   0.7965   0.7517   0.9636  
0.75       0.8616   0.8446   0.9716   0.7965   0.7517   0.9636  
0.80       0.8616   0.8446   0.9716   0.7965   0.7517   0.9636  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8616, F1=0.8446, Normal Recall=0.9716, Normal Precision=0.7965, Attack Recall=0.7517, Attack Precision=0.9636

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
0.15       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407   <--
0.20       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.25       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.30       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.35       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.40       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.45       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.50       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.55       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.60       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.65       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.70       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.75       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.80       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9490, F1=0.7471, Normal Recall=0.9707, Normal Precision=0.9726, Attack Recall=0.7536, Attack Precision=0.7407

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
0.15       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661   <--
0.20       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.25       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.30       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.35       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.40       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.45       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.50       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.55       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.60       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.65       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.70       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.75       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.80       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9272, F1=0.8052, Normal Recall=0.9709, Normal Precision=0.9400, Attack Recall=0.7523, Attack Precision=0.8661

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
0.15       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178   <--
0.20       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.25       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.30       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.35       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.40       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.45       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.50       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.55       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.60       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.65       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.70       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.75       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.80       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9055, F1=0.8268, Normal Recall=0.9711, Normal Precision=0.9014, Attack Recall=0.7523, Attack Precision=0.9178

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
0.15       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467   <--
0.20       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.25       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.30       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.35       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.40       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.45       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.50       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.55       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.60       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.65       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.70       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.75       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.80       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8840, F1=0.8383, Normal Recall=0.9717, Normal Precision=0.8547, Attack Recall=0.7523, Attack Precision=0.9467

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
0.15       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625   <--
0.20       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.25       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.30       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.35       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.40       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.45       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.50       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.55       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.60       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.65       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.70       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.75       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.80       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8615, F1=0.8445, Normal Recall=0.9707, Normal Precision=0.7967, Attack Recall=0.7523, Attack Precision=0.9625

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
0.15       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407   <--
0.20       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.25       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.30       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.35       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.40       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.45       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.50       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.55       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.60       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.65       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.70       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.75       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
0.80       0.9490   0.7471   0.9707   0.9726   0.7536   0.7407  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9490, F1=0.7471, Normal Recall=0.9707, Normal Precision=0.9726, Attack Recall=0.7536, Attack Precision=0.7407

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
0.15       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661   <--
0.20       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.25       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.30       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.35       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.40       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.45       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.50       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.55       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.60       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.65       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.70       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.75       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
0.80       0.9272   0.8052   0.9709   0.9400   0.7523   0.8661  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9272, F1=0.8052, Normal Recall=0.9709, Normal Precision=0.9400, Attack Recall=0.7523, Attack Precision=0.8661

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
0.15       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178   <--
0.20       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.25       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.30       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.35       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.40       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.45       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.50       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.55       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.60       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.65       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.70       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.75       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
0.80       0.9055   0.8268   0.9711   0.9014   0.7523   0.9178  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9055, F1=0.8268, Normal Recall=0.9711, Normal Precision=0.9014, Attack Recall=0.7523, Attack Precision=0.9178

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
0.15       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467   <--
0.20       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.25       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.30       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.35       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.40       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.45       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.50       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.55       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.60       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.65       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.70       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.75       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
0.80       0.8840   0.8383   0.9717   0.8547   0.7523   0.9467  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8840, F1=0.8383, Normal Recall=0.9717, Normal Precision=0.8547, Attack Recall=0.7523, Attack Precision=0.9467

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
0.15       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625   <--
0.20       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.25       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.30       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.35       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.40       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.45       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.50       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.55       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.60       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.65       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.70       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.75       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
0.80       0.8615   0.8445   0.9707   0.7967   0.7523   0.9625  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8615, F1=0.8445, Normal Recall=0.9707, Normal Precision=0.7967, Attack Recall=0.7523, Attack Precision=0.9625

```

