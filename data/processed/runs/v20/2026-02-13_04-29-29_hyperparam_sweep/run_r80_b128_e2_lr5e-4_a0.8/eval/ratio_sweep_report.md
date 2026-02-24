# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e2_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-18 01:05:21 |

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
| Original (TFLite) | 0.9035 | 0.8870 | 0.8706 | 0.8555 | 0.8388 | 0.8224 | 0.8069 | 0.7919 | 0.7755 | 0.7594 | 0.7432 |
| QAT+Prune only | 0.8685 | 0.8823 | 0.8942 | 0.9073 | 0.9202 | 0.9312 | 0.9449 | 0.9573 | 0.9695 | 0.9817 | 0.9945 |
| QAT+PTQ | 0.8679 | 0.8817 | 0.8937 | 0.9069 | 0.9198 | 0.9308 | 0.9447 | 0.9569 | 0.9693 | 0.9816 | 0.9944 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8679 | 0.8817 | 0.8937 | 0.9069 | 0.9198 | 0.9308 | 0.9447 | 0.9569 | 0.9693 | 0.9816 | 0.9944 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.5675 | 0.6967 | 0.7552 | 0.7867 | 0.8071 | 0.8220 | 0.8334 | 0.8412 | 0.8476 | 0.8527 |
| QAT+Prune only | 0.0000 | 0.6283 | 0.7899 | 0.8655 | 0.9089 | 0.9353 | 0.9559 | 0.9703 | 0.9812 | 0.9899 | 0.9972 |
| QAT+PTQ | 0.0000 | 0.6272 | 0.7891 | 0.8650 | 0.9084 | 0.9349 | 0.9557 | 0.9700 | 0.9810 | 0.9898 | 0.9972 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.6272 | 0.7891 | 0.8650 | 0.9084 | 0.9349 | 0.9557 | 0.9700 | 0.9810 | 0.9898 | 0.9972 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9035 | 0.9031 | 0.9024 | 0.9036 | 0.9025 | 0.9016 | 0.9025 | 0.9056 | 0.9046 | 0.9052 | 0.0000 |
| QAT+Prune only | 0.8685 | 0.8699 | 0.8691 | 0.8699 | 0.8707 | 0.8680 | 0.8706 | 0.8707 | 0.8697 | 0.8665 | 0.0000 |
| QAT+PTQ | 0.8679 | 0.8692 | 0.8685 | 0.8693 | 0.8701 | 0.8672 | 0.8701 | 0.8695 | 0.8687 | 0.8667 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8679 | 0.8692 | 0.8685 | 0.8693 | 0.8701 | 0.8672 | 0.8701 | 0.8695 | 0.8687 | 0.8667 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9035 | 0.0000 | 0.0000 | 0.0000 | 0.9035 | 1.0000 |
| 90 | 10 | 299,940 | 0.8870 | 0.4596 | 0.7414 | 0.5675 | 0.9031 | 0.9692 |
| 80 | 20 | 291,350 | 0.8706 | 0.6557 | 0.7432 | 0.6967 | 0.9024 | 0.9336 |
| 70 | 30 | 194,230 | 0.8555 | 0.7677 | 0.7432 | 0.7552 | 0.9036 | 0.8914 |
| 60 | 40 | 145,675 | 0.8388 | 0.8356 | 0.7432 | 0.7867 | 0.9025 | 0.8406 |
| 50 | 50 | 116,540 | 0.8224 | 0.8831 | 0.7432 | 0.8071 | 0.9016 | 0.7783 |
| 40 | 60 | 97,115 | 0.8069 | 0.9196 | 0.7432 | 0.8220 | 0.9025 | 0.7009 |
| 30 | 70 | 83,240 | 0.7919 | 0.9484 | 0.7432 | 0.8334 | 0.9056 | 0.6018 |
| 20 | 80 | 72,835 | 0.7755 | 0.9689 | 0.7432 | 0.8412 | 0.9046 | 0.4683 |
| 10 | 90 | 64,740 | 0.7594 | 0.9860 | 0.7432 | 0.8476 | 0.9052 | 0.2814 |
| 0 | 100 | 58,270 | 0.7432 | 1.0000 | 0.7432 | 0.8527 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8685 | 0.0000 | 0.0000 | 0.0000 | 0.8685 | 1.0000 |
| 90 | 10 | 299,940 | 0.8823 | 0.4592 | 0.9946 | 0.6283 | 0.8699 | 0.9993 |
| 80 | 20 | 291,350 | 0.8942 | 0.6552 | 0.9945 | 0.7899 | 0.8691 | 0.9984 |
| 70 | 30 | 194,230 | 0.9073 | 0.7662 | 0.9945 | 0.8655 | 0.8699 | 0.9973 |
| 60 | 40 | 145,675 | 0.9202 | 0.8368 | 0.9945 | 0.9089 | 0.8707 | 0.9958 |
| 50 | 50 | 116,540 | 0.9312 | 0.8828 | 0.9945 | 0.9353 | 0.8680 | 0.9937 |
| 40 | 60 | 97,115 | 0.9449 | 0.9202 | 0.9945 | 0.9559 | 0.8706 | 0.9906 |
| 30 | 70 | 83,240 | 0.9573 | 0.9472 | 0.9945 | 0.9703 | 0.8707 | 0.9855 |
| 20 | 80 | 72,835 | 0.9695 | 0.9683 | 0.9945 | 0.9812 | 0.8697 | 0.9753 |
| 10 | 90 | 64,740 | 0.9817 | 0.9853 | 0.9945 | 0.9899 | 0.8665 | 0.9459 |
| 0 | 100 | 58,270 | 0.9945 | 1.0000 | 0.9945 | 0.9972 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8679 | 0.0000 | 0.0000 | 0.0000 | 0.8679 | 1.0000 |
| 90 | 10 | 299,940 | 0.8817 | 0.4580 | 0.9947 | 0.6272 | 0.8692 | 0.9993 |
| 80 | 20 | 291,350 | 0.8937 | 0.6541 | 0.9944 | 0.7891 | 0.8685 | 0.9984 |
| 70 | 30 | 194,230 | 0.9069 | 0.7653 | 0.9944 | 0.8650 | 0.8693 | 0.9972 |
| 60 | 40 | 145,675 | 0.9198 | 0.8361 | 0.9944 | 0.9084 | 0.8701 | 0.9957 |
| 50 | 50 | 116,540 | 0.9308 | 0.8822 | 0.9944 | 0.9349 | 0.8672 | 0.9936 |
| 40 | 60 | 97,115 | 0.9447 | 0.9199 | 0.9944 | 0.9557 | 0.8701 | 0.9904 |
| 30 | 70 | 83,240 | 0.9569 | 0.9467 | 0.9944 | 0.9700 | 0.8695 | 0.9852 |
| 20 | 80 | 72,835 | 0.9693 | 0.9680 | 0.9944 | 0.9810 | 0.8687 | 0.9749 |
| 10 | 90 | 64,740 | 0.9816 | 0.9853 | 0.9944 | 0.9898 | 0.8667 | 0.9451 |
| 0 | 100 | 58,270 | 0.9944 | 1.0000 | 0.9944 | 0.9972 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8679 | 0.0000 | 0.0000 | 0.0000 | 0.8679 | 1.0000 |
| 90 | 10 | 299,940 | 0.8817 | 0.4580 | 0.9947 | 0.6272 | 0.8692 | 0.9993 |
| 80 | 20 | 291,350 | 0.8937 | 0.6541 | 0.9944 | 0.7891 | 0.8685 | 0.9984 |
| 70 | 30 | 194,230 | 0.9069 | 0.7653 | 0.9944 | 0.8650 | 0.8693 | 0.9972 |
| 60 | 40 | 145,675 | 0.9198 | 0.8361 | 0.9944 | 0.9084 | 0.8701 | 0.9957 |
| 50 | 50 | 116,540 | 0.9308 | 0.8822 | 0.9944 | 0.9349 | 0.8672 | 0.9936 |
| 40 | 60 | 97,115 | 0.9447 | 0.9199 | 0.9944 | 0.9557 | 0.8701 | 0.9904 |
| 30 | 70 | 83,240 | 0.9569 | 0.9467 | 0.9944 | 0.9700 | 0.8695 | 0.9852 |
| 20 | 80 | 72,835 | 0.9693 | 0.9680 | 0.9944 | 0.9810 | 0.8687 | 0.9749 |
| 10 | 90 | 64,740 | 0.9816 | 0.9853 | 0.9944 | 0.9898 | 0.8667 | 0.9451 |
| 0 | 100 | 58,270 | 0.9944 | 1.0000 | 0.9944 | 0.9972 | 0.0000 | 0.0000 |


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
0.15       0.8869   0.5672   0.9031   0.9691   0.7409   0.4595   <--
0.20       0.8869   0.5672   0.9031   0.9691   0.7409   0.4595  
0.25       0.8869   0.5672   0.9031   0.9691   0.7409   0.4595  
0.30       0.8869   0.5672   0.9031   0.9691   0.7409   0.4595  
0.35       0.8869   0.5672   0.9031   0.9691   0.7409   0.4595  
0.40       0.8869   0.5672   0.9031   0.9691   0.7409   0.4595  
0.45       0.8869   0.5672   0.9031   0.9691   0.7409   0.4595  
0.50       0.8869   0.5672   0.9031   0.9691   0.7409   0.4595  
0.55       0.8869   0.5672   0.9031   0.9691   0.7409   0.4595  
0.60       0.8869   0.5672   0.9031   0.9691   0.7409   0.4595  
0.65       0.8869   0.5672   0.9031   0.9691   0.7409   0.4595  
0.70       0.8869   0.5672   0.9031   0.9691   0.7409   0.4595  
0.75       0.8869   0.5672   0.9031   0.9691   0.7409   0.4595  
0.80       0.8869   0.5672   0.9031   0.9691   0.7409   0.4595  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8869, F1=0.5672, Normal Recall=0.9031, Normal Precision=0.9691, Attack Recall=0.7409, Attack Precision=0.4595

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
0.15       0.8711   0.6976   0.9031   0.9336   0.7432   0.6573   <--
0.20       0.8711   0.6976   0.9031   0.9336   0.7432   0.6573  
0.25       0.8711   0.6976   0.9031   0.9336   0.7432   0.6573  
0.30       0.8711   0.6976   0.9031   0.9336   0.7432   0.6573  
0.35       0.8711   0.6976   0.9031   0.9336   0.7432   0.6573  
0.40       0.8711   0.6976   0.9031   0.9336   0.7432   0.6573  
0.45       0.8711   0.6976   0.9031   0.9336   0.7432   0.6573  
0.50       0.8711   0.6976   0.9031   0.9336   0.7432   0.6573  
0.55       0.8711   0.6976   0.9031   0.9336   0.7432   0.6573  
0.60       0.8711   0.6976   0.9031   0.9336   0.7432   0.6573  
0.65       0.8711   0.6976   0.9031   0.9336   0.7432   0.6573  
0.70       0.8711   0.6976   0.9031   0.9336   0.7432   0.6573  
0.75       0.8711   0.6976   0.9031   0.9336   0.7432   0.6573  
0.80       0.8711   0.6976   0.9031   0.9336   0.7432   0.6573  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8711, F1=0.6976, Normal Recall=0.9031, Normal Precision=0.9336, Attack Recall=0.7432, Attack Precision=0.6573

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
0.15       0.8558   0.7557   0.9041   0.8915   0.7432   0.7685   <--
0.20       0.8558   0.7557   0.9041   0.8915   0.7432   0.7685  
0.25       0.8558   0.7557   0.9041   0.8915   0.7432   0.7685  
0.30       0.8558   0.7557   0.9041   0.8915   0.7432   0.7685  
0.35       0.8558   0.7557   0.9041   0.8915   0.7432   0.7685  
0.40       0.8558   0.7557   0.9041   0.8915   0.7432   0.7685  
0.45       0.8558   0.7557   0.9041   0.8915   0.7432   0.7685  
0.50       0.8558   0.7557   0.9041   0.8915   0.7432   0.7685  
0.55       0.8558   0.7557   0.9041   0.8915   0.7432   0.7685  
0.60       0.8558   0.7557   0.9041   0.8915   0.7432   0.7685  
0.65       0.8558   0.7557   0.9041   0.8915   0.7432   0.7685  
0.70       0.8558   0.7557   0.9041   0.8915   0.7432   0.7685  
0.75       0.8558   0.7557   0.9041   0.8915   0.7432   0.7685  
0.80       0.8558   0.7557   0.9041   0.8915   0.7432   0.7685  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8558, F1=0.7557, Normal Recall=0.9041, Normal Precision=0.8915, Attack Recall=0.7432, Attack Precision=0.7685

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
0.15       0.8392   0.7871   0.9032   0.8407   0.7432   0.8366   <--
0.20       0.8392   0.7871   0.9032   0.8407   0.7432   0.8366  
0.25       0.8392   0.7871   0.9032   0.8407   0.7432   0.8366  
0.30       0.8392   0.7871   0.9032   0.8407   0.7432   0.8366  
0.35       0.8392   0.7871   0.9032   0.8407   0.7432   0.8366  
0.40       0.8392   0.7871   0.9032   0.8407   0.7432   0.8366  
0.45       0.8392   0.7871   0.9032   0.8407   0.7432   0.8366  
0.50       0.8392   0.7871   0.9032   0.8407   0.7432   0.8366  
0.55       0.8392   0.7871   0.9032   0.8407   0.7432   0.8366  
0.60       0.8392   0.7871   0.9032   0.8407   0.7432   0.8366  
0.65       0.8392   0.7871   0.9032   0.8407   0.7432   0.8366  
0.70       0.8392   0.7871   0.9032   0.8407   0.7432   0.8366  
0.75       0.8392   0.7871   0.9032   0.8407   0.7432   0.8366  
0.80       0.8392   0.7871   0.9032   0.8407   0.7432   0.8366  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8392, F1=0.7871, Normal Recall=0.9032, Normal Precision=0.8407, Attack Recall=0.7432, Attack Precision=0.8366

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
0.15       0.8227   0.8074   0.9022   0.7784   0.7432   0.8837   <--
0.20       0.8227   0.8074   0.9022   0.7784   0.7432   0.8837  
0.25       0.8227   0.8074   0.9022   0.7784   0.7432   0.8837  
0.30       0.8227   0.8074   0.9022   0.7784   0.7432   0.8837  
0.35       0.8227   0.8074   0.9022   0.7784   0.7432   0.8837  
0.40       0.8227   0.8074   0.9022   0.7784   0.7432   0.8837  
0.45       0.8227   0.8074   0.9022   0.7784   0.7432   0.8837  
0.50       0.8227   0.8074   0.9022   0.7784   0.7432   0.8837  
0.55       0.8227   0.8074   0.9022   0.7784   0.7432   0.8837  
0.60       0.8227   0.8074   0.9022   0.7784   0.7432   0.8837  
0.65       0.8227   0.8074   0.9022   0.7784   0.7432   0.8837  
0.70       0.8227   0.8074   0.9022   0.7784   0.7432   0.8837  
0.75       0.8227   0.8074   0.9022   0.7784   0.7432   0.8837  
0.80       0.8227   0.8074   0.9022   0.7784   0.7432   0.8837  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8227, F1=0.8074, Normal Recall=0.9022, Normal Precision=0.7784, Attack Recall=0.7432, Attack Precision=0.8837

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
0.15       0.8823   0.6283   0.8699   0.9993   0.9947   0.4592   <--
0.20       0.8823   0.6283   0.8699   0.9993   0.9947   0.4592  
0.25       0.8823   0.6283   0.8699   0.9993   0.9947   0.4592  
0.30       0.8823   0.6283   0.8699   0.9993   0.9947   0.4592  
0.35       0.8823   0.6283   0.8699   0.9993   0.9947   0.4592  
0.40       0.8823   0.6283   0.8699   0.9993   0.9947   0.4592  
0.45       0.8823   0.6283   0.8699   0.9993   0.9947   0.4592  
0.50       0.8823   0.6283   0.8699   0.9993   0.9947   0.4592  
0.55       0.8823   0.6283   0.8699   0.9993   0.9947   0.4592  
0.60       0.8823   0.6283   0.8699   0.9993   0.9947   0.4592  
0.65       0.8823   0.6283   0.8699   0.9993   0.9947   0.4592  
0.70       0.8823   0.6283   0.8699   0.9993   0.9947   0.4592  
0.75       0.8823   0.6283   0.8699   0.9993   0.9947   0.4592  
0.80       0.8823   0.6283   0.8699   0.9993   0.9947   0.4592  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8823, F1=0.6283, Normal Recall=0.8699, Normal Precision=0.9993, Attack Recall=0.9947, Attack Precision=0.4592

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
0.15       0.8949   0.7911   0.8700   0.9984   0.9945   0.6567   <--
0.20       0.8949   0.7911   0.8700   0.9984   0.9945   0.6567  
0.25       0.8949   0.7911   0.8700   0.9984   0.9945   0.6567  
0.30       0.8949   0.7911   0.8700   0.9984   0.9945   0.6567  
0.35       0.8949   0.7911   0.8700   0.9984   0.9945   0.6567  
0.40       0.8949   0.7911   0.8700   0.9984   0.9945   0.6567  
0.45       0.8949   0.7911   0.8700   0.9984   0.9945   0.6567  
0.50       0.8949   0.7911   0.8700   0.9984   0.9945   0.6567  
0.55       0.8949   0.7911   0.8700   0.9984   0.9945   0.6567  
0.60       0.8949   0.7911   0.8700   0.9984   0.9945   0.6567  
0.65       0.8949   0.7911   0.8700   0.9984   0.9945   0.6567  
0.70       0.8949   0.7911   0.8700   0.9984   0.9945   0.6567  
0.75       0.8949   0.7911   0.8700   0.9984   0.9945   0.6567  
0.80       0.8949   0.7911   0.8700   0.9984   0.9945   0.6567  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8949, F1=0.7911, Normal Recall=0.8700, Normal Precision=0.9984, Attack Recall=0.9945, Attack Precision=0.6567

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
0.15       0.9069   0.8651   0.8694   0.9973   0.9945   0.7655   <--
0.20       0.9069   0.8651   0.8694   0.9973   0.9945   0.7655  
0.25       0.9069   0.8651   0.8694   0.9973   0.9945   0.7655  
0.30       0.9069   0.8651   0.8694   0.9973   0.9945   0.7655  
0.35       0.9069   0.8651   0.8694   0.9973   0.9945   0.7655  
0.40       0.9069   0.8651   0.8694   0.9973   0.9945   0.7655  
0.45       0.9069   0.8651   0.8694   0.9973   0.9945   0.7655  
0.50       0.9069   0.8651   0.8694   0.9973   0.9945   0.7655  
0.55       0.9069   0.8651   0.8694   0.9973   0.9945   0.7655  
0.60       0.9069   0.8651   0.8694   0.9973   0.9945   0.7655  
0.65       0.9069   0.8651   0.8694   0.9973   0.9945   0.7655  
0.70       0.9069   0.8651   0.8694   0.9973   0.9945   0.7655  
0.75       0.9069   0.8651   0.8694   0.9973   0.9945   0.7655  
0.80       0.9069   0.8651   0.8694   0.9973   0.9945   0.7655  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9069, F1=0.8651, Normal Recall=0.8694, Normal Precision=0.9973, Attack Recall=0.9945, Attack Precision=0.7655

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
0.15       0.9191   0.9077   0.8688   0.9958   0.9945   0.8348   <--
0.20       0.9191   0.9077   0.8688   0.9958   0.9945   0.8348  
0.25       0.9191   0.9077   0.8688   0.9958   0.9945   0.8348  
0.30       0.9191   0.9077   0.8688   0.9958   0.9945   0.8348  
0.35       0.9191   0.9077   0.8688   0.9958   0.9945   0.8348  
0.40       0.9191   0.9077   0.8688   0.9958   0.9945   0.8348  
0.45       0.9191   0.9077   0.8688   0.9958   0.9945   0.8348  
0.50       0.9191   0.9077   0.8688   0.9958   0.9945   0.8348  
0.55       0.9191   0.9077   0.8688   0.9958   0.9945   0.8348  
0.60       0.9191   0.9077   0.8688   0.9958   0.9945   0.8348  
0.65       0.9191   0.9077   0.8688   0.9958   0.9945   0.8348  
0.70       0.9191   0.9077   0.8688   0.9958   0.9945   0.8348  
0.75       0.9191   0.9077   0.8688   0.9958   0.9945   0.8348  
0.80       0.9191   0.9077   0.8688   0.9958   0.9945   0.8348  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9191, F1=0.9077, Normal Recall=0.8688, Normal Precision=0.9958, Attack Recall=0.9945, Attack Precision=0.8348

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
0.15       0.9316   0.9356   0.8686   0.9937   0.9945   0.8833   <--
0.20       0.9316   0.9356   0.8686   0.9937   0.9945   0.8833  
0.25       0.9316   0.9356   0.8686   0.9937   0.9945   0.8833  
0.30       0.9316   0.9356   0.8686   0.9937   0.9945   0.8833  
0.35       0.9316   0.9356   0.8686   0.9937   0.9945   0.8833  
0.40       0.9316   0.9356   0.8686   0.9937   0.9945   0.8833  
0.45       0.9316   0.9356   0.8686   0.9937   0.9945   0.8833  
0.50       0.9316   0.9356   0.8686   0.9937   0.9945   0.8833  
0.55       0.9316   0.9356   0.8686   0.9937   0.9945   0.8833  
0.60       0.9316   0.9356   0.8686   0.9937   0.9945   0.8833  
0.65       0.9316   0.9356   0.8686   0.9937   0.9945   0.8833  
0.70       0.9316   0.9356   0.8686   0.9937   0.9945   0.8833  
0.75       0.9316   0.9356   0.8686   0.9937   0.9945   0.8833  
0.80       0.9316   0.9356   0.8686   0.9937   0.9945   0.8833  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9316, F1=0.9356, Normal Recall=0.8686, Normal Precision=0.9937, Attack Recall=0.9945, Attack Precision=0.8833

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
0.15       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579   <--
0.20       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.25       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.30       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.35       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.40       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.45       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.50       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.55       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.60       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.65       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.70       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.75       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.80       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8817, F1=0.6271, Normal Recall=0.8692, Normal Precision=0.9993, Attack Recall=0.9945, Attack Precision=0.4579

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
0.15       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555   <--
0.20       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.25       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.30       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.35       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.40       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.45       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.50       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.55       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.60       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.65       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.70       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.75       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.80       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8944, F1=0.7902, Normal Recall=0.8694, Normal Precision=0.9984, Attack Recall=0.9944, Attack Precision=0.6555

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
0.15       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645   <--
0.20       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.25       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.30       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.35       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.40       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.45       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.50       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.55       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.60       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.65       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.70       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.75       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.80       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9064, F1=0.8644, Normal Recall=0.8687, Normal Precision=0.9972, Attack Recall=0.9944, Attack Precision=0.7645

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
0.15       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342   <--
0.20       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.25       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.30       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.35       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.40       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.45       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.50       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.55       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.60       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.65       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.70       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.75       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.80       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9187, F1=0.9073, Normal Recall=0.8683, Normal Precision=0.9957, Attack Recall=0.9944, Attack Precision=0.8342

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
0.15       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829   <--
0.20       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.25       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.30       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.35       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.40       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.45       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.50       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.55       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.60       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.65       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.70       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.75       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.80       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9312, F1=0.9353, Normal Recall=0.8681, Normal Precision=0.9936, Attack Recall=0.9944, Attack Precision=0.8829

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
0.15       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579   <--
0.20       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.25       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.30       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.35       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.40       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.45       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.50       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.55       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.60       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.65       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.70       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.75       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
0.80       0.8817   0.6271   0.8692   0.9993   0.9945   0.4579  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8817, F1=0.6271, Normal Recall=0.8692, Normal Precision=0.9993, Attack Recall=0.9945, Attack Precision=0.4579

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
0.15       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555   <--
0.20       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.25       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.30       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.35       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.40       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.45       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.50       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.55       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.60       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.65       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.70       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.75       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
0.80       0.8944   0.7902   0.8694   0.9984   0.9944   0.6555  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8944, F1=0.7902, Normal Recall=0.8694, Normal Precision=0.9984, Attack Recall=0.9944, Attack Precision=0.6555

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
0.15       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645   <--
0.20       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.25       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.30       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.35       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.40       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.45       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.50       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.55       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.60       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.65       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.70       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.75       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
0.80       0.9064   0.8644   0.8687   0.9972   0.9944   0.7645  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9064, F1=0.8644, Normal Recall=0.8687, Normal Precision=0.9972, Attack Recall=0.9944, Attack Precision=0.7645

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
0.15       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342   <--
0.20       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.25       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.30       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.35       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.40       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.45       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.50       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.55       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.60       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.65       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.70       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.75       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
0.80       0.9187   0.9073   0.8683   0.9957   0.9944   0.8342  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9187, F1=0.9073, Normal Recall=0.8683, Normal Precision=0.9957, Attack Recall=0.9944, Attack Precision=0.8342

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
0.15       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829   <--
0.20       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.25       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.30       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.35       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.40       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.45       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.50       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.55       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.60       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.65       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.70       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.75       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
0.80       0.9312   0.9353   0.8681   0.9936   0.9944   0.8829  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9312, F1=0.9353, Normal Recall=0.8681, Normal Precision=0.9936, Attack Recall=0.9944, Attack Precision=0.8829

```

