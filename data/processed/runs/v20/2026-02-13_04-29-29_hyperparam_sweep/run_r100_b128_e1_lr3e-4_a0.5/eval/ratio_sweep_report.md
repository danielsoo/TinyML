# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e1_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-20 21:56:52 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 1 |
| **Batch size** | 128 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5701 | 0.6125 | 0.6549 | 0.6979 | 0.7417 | 0.7816 | 0.8271 | 0.8683 | 0.9097 | 0.9538 | 0.9962 |
| QAT+Prune only | 0.8760 | 0.8511 | 0.8274 | 0.8038 | 0.7802 | 0.7558 | 0.7333 | 0.7083 | 0.6854 | 0.6614 | 0.6379 |
| QAT+PTQ | 0.8756 | 0.8505 | 0.8268 | 0.8031 | 0.7793 | 0.7548 | 0.7322 | 0.7069 | 0.6841 | 0.6599 | 0.6362 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8756 | 0.8505 | 0.8268 | 0.8031 | 0.7793 | 0.7548 | 0.7322 | 0.7069 | 0.6841 | 0.6599 | 0.6362 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3395 | 0.5359 | 0.6642 | 0.7552 | 0.8202 | 0.8736 | 0.9137 | 0.9464 | 0.9749 | 0.9981 |
| QAT+Prune only | 0.0000 | 0.4598 | 0.5965 | 0.6611 | 0.6989 | 0.7232 | 0.7417 | 0.7538 | 0.7644 | 0.7723 | 0.7789 |
| QAT+PTQ | 0.0000 | 0.4584 | 0.5950 | 0.6597 | 0.6976 | 0.7218 | 0.7403 | 0.7524 | 0.7632 | 0.7710 | 0.7777 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4584 | 0.5950 | 0.6597 | 0.6976 | 0.7218 | 0.7403 | 0.7524 | 0.7632 | 0.7710 | 0.7777 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5701 | 0.5698 | 0.5695 | 0.5700 | 0.5720 | 0.5670 | 0.5734 | 0.5699 | 0.5635 | 0.5726 | 0.0000 |
| QAT+Prune only | 0.8760 | 0.8752 | 0.8748 | 0.8749 | 0.8750 | 0.8738 | 0.8765 | 0.8724 | 0.8754 | 0.8730 | 0.0000 |
| QAT+PTQ | 0.8756 | 0.8748 | 0.8744 | 0.8746 | 0.8747 | 0.8733 | 0.8761 | 0.8718 | 0.8755 | 0.8727 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8756 | 0.8748 | 0.8744 | 0.8746 | 0.8747 | 0.8733 | 0.8761 | 0.8718 | 0.8755 | 0.8727 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5701 | 0.0000 | 0.0000 | 0.0000 | 0.5701 | 1.0000 |
| 90 | 10 | 299,940 | 0.6125 | 0.2046 | 0.9961 | 0.3395 | 0.5698 | 0.9992 |
| 80 | 20 | 291,350 | 0.6549 | 0.3665 | 0.9962 | 0.5359 | 0.5695 | 0.9983 |
| 70 | 30 | 194,230 | 0.6979 | 0.4982 | 0.9962 | 0.6642 | 0.5700 | 0.9972 |
| 60 | 40 | 145,675 | 0.7417 | 0.6081 | 0.9962 | 0.7552 | 0.5720 | 0.9956 |
| 50 | 50 | 116,540 | 0.7816 | 0.6970 | 0.9962 | 0.8202 | 0.5670 | 0.9934 |
| 40 | 60 | 97,115 | 0.8271 | 0.7779 | 0.9962 | 0.8736 | 0.5734 | 0.9902 |
| 30 | 70 | 83,240 | 0.8683 | 0.8439 | 0.9962 | 0.9137 | 0.5699 | 0.9847 |
| 20 | 80 | 72,835 | 0.9097 | 0.9013 | 0.9962 | 0.9464 | 0.5635 | 0.9738 |
| 10 | 90 | 64,740 | 0.9538 | 0.9545 | 0.9962 | 0.9749 | 0.5726 | 0.9437 |
| 0 | 100 | 58,270 | 0.9962 | 1.0000 | 0.9962 | 0.9981 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8760 | 0.0000 | 0.0000 | 0.0000 | 0.8760 | 1.0000 |
| 90 | 10 | 299,940 | 0.8511 | 0.3607 | 0.6339 | 0.4598 | 0.8752 | 0.9556 |
| 80 | 20 | 291,350 | 0.8274 | 0.5602 | 0.6379 | 0.5965 | 0.8748 | 0.9062 |
| 70 | 30 | 194,230 | 0.8038 | 0.6861 | 0.6379 | 0.6611 | 0.8749 | 0.8493 |
| 60 | 40 | 145,675 | 0.7802 | 0.7729 | 0.6379 | 0.6989 | 0.8750 | 0.7838 |
| 50 | 50 | 116,540 | 0.7558 | 0.8348 | 0.6379 | 0.7232 | 0.8738 | 0.7070 |
| 40 | 60 | 97,115 | 0.7333 | 0.8857 | 0.6379 | 0.7417 | 0.8765 | 0.6174 |
| 30 | 70 | 83,240 | 0.7083 | 0.9211 | 0.6379 | 0.7538 | 0.8724 | 0.5080 |
| 20 | 80 | 72,835 | 0.6854 | 0.9534 | 0.6379 | 0.7644 | 0.8754 | 0.3767 |
| 10 | 90 | 64,740 | 0.6614 | 0.9784 | 0.6379 | 0.7723 | 0.8730 | 0.2113 |
| 0 | 100 | 58,270 | 0.6379 | 1.0000 | 0.6379 | 0.7789 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8756 | 0.0000 | 0.0000 | 0.0000 | 0.8756 | 1.0000 |
| 90 | 10 | 299,940 | 0.8505 | 0.3594 | 0.6324 | 0.4584 | 0.8748 | 0.9554 |
| 80 | 20 | 291,350 | 0.8268 | 0.5588 | 0.6362 | 0.5950 | 0.8744 | 0.9058 |
| 70 | 30 | 194,230 | 0.8031 | 0.6849 | 0.6362 | 0.6597 | 0.8746 | 0.8487 |
| 60 | 40 | 145,675 | 0.7793 | 0.7720 | 0.6362 | 0.6976 | 0.8747 | 0.7829 |
| 50 | 50 | 116,540 | 0.7548 | 0.8339 | 0.6362 | 0.7218 | 0.8733 | 0.7060 |
| 40 | 60 | 97,115 | 0.7322 | 0.8851 | 0.6363 | 0.7403 | 0.8761 | 0.6162 |
| 30 | 70 | 83,240 | 0.7069 | 0.9205 | 0.6362 | 0.7524 | 0.8718 | 0.5067 |
| 20 | 80 | 72,835 | 0.6841 | 0.9534 | 0.6362 | 0.7632 | 0.8755 | 0.3757 |
| 10 | 90 | 64,740 | 0.6599 | 0.9783 | 0.6363 | 0.7710 | 0.8727 | 0.2105 |
| 0 | 100 | 58,270 | 0.6362 | 1.0000 | 0.6362 | 0.7777 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8756 | 0.0000 | 0.0000 | 0.0000 | 0.8756 | 1.0000 |
| 90 | 10 | 299,940 | 0.8505 | 0.3594 | 0.6324 | 0.4584 | 0.8748 | 0.9554 |
| 80 | 20 | 291,350 | 0.8268 | 0.5588 | 0.6362 | 0.5950 | 0.8744 | 0.9058 |
| 70 | 30 | 194,230 | 0.8031 | 0.6849 | 0.6362 | 0.6597 | 0.8746 | 0.8487 |
| 60 | 40 | 145,675 | 0.7793 | 0.7720 | 0.6362 | 0.6976 | 0.8747 | 0.7829 |
| 50 | 50 | 116,540 | 0.7548 | 0.8339 | 0.6362 | 0.7218 | 0.8733 | 0.7060 |
| 40 | 60 | 97,115 | 0.7322 | 0.8851 | 0.6363 | 0.7403 | 0.8761 | 0.6162 |
| 30 | 70 | 83,240 | 0.7069 | 0.9205 | 0.6362 | 0.7524 | 0.8718 | 0.5067 |
| 20 | 80 | 72,835 | 0.6841 | 0.9534 | 0.6362 | 0.7632 | 0.8755 | 0.3757 |
| 10 | 90 | 64,740 | 0.6599 | 0.9783 | 0.6363 | 0.7710 | 0.8727 | 0.2105 |
| 0 | 100 | 58,270 | 0.6362 | 1.0000 | 0.6362 | 0.7777 | 0.0000 | 0.0000 |


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
0.15       0.6125   0.3397   0.5698   0.9993   0.9966   0.2047   <--
0.20       0.6125   0.3397   0.5698   0.9993   0.9966   0.2047  
0.25       0.6125   0.3397   0.5698   0.9993   0.9966   0.2047  
0.30       0.6125   0.3397   0.5698   0.9993   0.9966   0.2047  
0.35       0.6125   0.3397   0.5698   0.9993   0.9966   0.2047  
0.40       0.6125   0.3397   0.5698   0.9993   0.9966   0.2047  
0.45       0.6125   0.3397   0.5698   0.9993   0.9966   0.2047  
0.50       0.6125   0.3397   0.5698   0.9993   0.9966   0.2047  
0.55       0.6125   0.3397   0.5698   0.9993   0.9966   0.2047  
0.60       0.6125   0.3397   0.5698   0.9993   0.9966   0.2047  
0.65       0.6125   0.3397   0.5698   0.9993   0.9966   0.2047  
0.70       0.6125   0.3397   0.5698   0.9993   0.9966   0.2047  
0.75       0.6125   0.3397   0.5698   0.9993   0.9966   0.2047  
0.80       0.6125   0.3397   0.5698   0.9993   0.9966   0.2047  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6125, F1=0.3397, Normal Recall=0.5698, Normal Precision=0.9993, Attack Recall=0.9966, Attack Precision=0.2047

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
0.15       0.6554   0.5362   0.5702   0.9983   0.9962   0.3669   <--
0.20       0.6554   0.5362   0.5702   0.9983   0.9962   0.3669  
0.25       0.6554   0.5362   0.5702   0.9983   0.9962   0.3669  
0.30       0.6554   0.5362   0.5702   0.9983   0.9962   0.3669  
0.35       0.6554   0.5362   0.5702   0.9983   0.9962   0.3669  
0.40       0.6554   0.5362   0.5702   0.9983   0.9962   0.3669  
0.45       0.6554   0.5362   0.5702   0.9983   0.9962   0.3669  
0.50       0.6554   0.5362   0.5702   0.9983   0.9962   0.3669  
0.55       0.6554   0.5362   0.5702   0.9983   0.9962   0.3669  
0.60       0.6554   0.5362   0.5702   0.9983   0.9962   0.3669  
0.65       0.6554   0.5362   0.5702   0.9983   0.9962   0.3669  
0.70       0.6554   0.5362   0.5702   0.9983   0.9962   0.3669  
0.75       0.6554   0.5362   0.5702   0.9983   0.9962   0.3669  
0.80       0.6554   0.5362   0.5702   0.9983   0.9962   0.3669  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6554, F1=0.5362, Normal Recall=0.5702, Normal Precision=0.9983, Attack Recall=0.9962, Attack Precision=0.3669

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
0.15       0.6991   0.6652   0.5718   0.9972   0.9962   0.4992   <--
0.20       0.6991   0.6652   0.5718   0.9972   0.9962   0.4992  
0.25       0.6991   0.6652   0.5718   0.9972   0.9962   0.4992  
0.30       0.6991   0.6652   0.5718   0.9972   0.9962   0.4992  
0.35       0.6991   0.6652   0.5718   0.9972   0.9962   0.4992  
0.40       0.6991   0.6652   0.5718   0.9972   0.9962   0.4992  
0.45       0.6991   0.6652   0.5718   0.9972   0.9962   0.4992  
0.50       0.6991   0.6652   0.5718   0.9972   0.9962   0.4992  
0.55       0.6991   0.6652   0.5718   0.9972   0.9962   0.4992  
0.60       0.6991   0.6652   0.5718   0.9972   0.9962   0.4992  
0.65       0.6991   0.6652   0.5718   0.9972   0.9962   0.4992  
0.70       0.6991   0.6652   0.5718   0.9972   0.9962   0.4992  
0.75       0.6991   0.6652   0.5718   0.9972   0.9962   0.4992  
0.80       0.6991   0.6652   0.5718   0.9972   0.9962   0.4992  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6991, F1=0.6652, Normal Recall=0.5718, Normal Precision=0.9972, Attack Recall=0.9962, Attack Precision=0.4992

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
0.15       0.7406   0.7545   0.5702   0.9956   0.9962   0.6071   <--
0.20       0.7406   0.7545   0.5702   0.9956   0.9962   0.6071  
0.25       0.7406   0.7545   0.5702   0.9956   0.9962   0.6071  
0.30       0.7406   0.7545   0.5702   0.9956   0.9962   0.6071  
0.35       0.7406   0.7545   0.5702   0.9956   0.9962   0.6071  
0.40       0.7406   0.7545   0.5702   0.9956   0.9962   0.6071  
0.45       0.7406   0.7545   0.5702   0.9956   0.9962   0.6071  
0.50       0.7406   0.7545   0.5702   0.9956   0.9962   0.6071  
0.55       0.7406   0.7545   0.5702   0.9956   0.9962   0.6071  
0.60       0.7406   0.7545   0.5702   0.9956   0.9962   0.6071  
0.65       0.7406   0.7545   0.5702   0.9956   0.9962   0.6071  
0.70       0.7406   0.7545   0.5702   0.9956   0.9962   0.6071  
0.75       0.7406   0.7545   0.5702   0.9956   0.9962   0.6071  
0.80       0.7406   0.7545   0.5702   0.9956   0.9962   0.6071  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7406, F1=0.7545, Normal Recall=0.5702, Normal Precision=0.9956, Attack Recall=0.9962, Attack Precision=0.6071

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
0.15       0.7831   0.8212   0.5700   0.9934   0.9962   0.6985   <--
0.20       0.7831   0.8212   0.5700   0.9934   0.9962   0.6985  
0.25       0.7831   0.8212   0.5700   0.9934   0.9962   0.6985  
0.30       0.7831   0.8212   0.5700   0.9934   0.9962   0.6985  
0.35       0.7831   0.8212   0.5700   0.9934   0.9962   0.6985  
0.40       0.7831   0.8212   0.5700   0.9934   0.9962   0.6985  
0.45       0.7831   0.8212   0.5700   0.9934   0.9962   0.6985  
0.50       0.7831   0.8212   0.5700   0.9934   0.9962   0.6985  
0.55       0.7831   0.8212   0.5700   0.9934   0.9962   0.6985  
0.60       0.7831   0.8212   0.5700   0.9934   0.9962   0.6985  
0.65       0.7831   0.8212   0.5700   0.9934   0.9962   0.6985  
0.70       0.7831   0.8212   0.5700   0.9934   0.9962   0.6985  
0.75       0.7831   0.8212   0.5700   0.9934   0.9962   0.6985  
0.80       0.7831   0.8212   0.5700   0.9934   0.9962   0.6985  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7831, F1=0.8212, Normal Recall=0.5700, Normal Precision=0.9934, Attack Recall=0.9962, Attack Precision=0.6985

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
0.15       0.8515   0.4621   0.8752   0.9561   0.6380   0.3622   <--
0.20       0.8515   0.4621   0.8752   0.9561   0.6380   0.3622  
0.25       0.8515   0.4621   0.8752   0.9561   0.6380   0.3622  
0.30       0.8515   0.4621   0.8752   0.9561   0.6380   0.3622  
0.35       0.8515   0.4621   0.8752   0.9561   0.6380   0.3622  
0.40       0.8515   0.4621   0.8752   0.9561   0.6380   0.3622  
0.45       0.8515   0.4621   0.8752   0.9561   0.6380   0.3622  
0.50       0.8515   0.4621   0.8752   0.9561   0.6380   0.3622  
0.55       0.8515   0.4621   0.8752   0.9561   0.6380   0.3622  
0.60       0.8515   0.4621   0.8752   0.9561   0.6380   0.3622  
0.65       0.8515   0.4621   0.8752   0.9561   0.6380   0.3622  
0.70       0.8515   0.4621   0.8752   0.9561   0.6380   0.3622  
0.75       0.8515   0.4621   0.8752   0.9561   0.6380   0.3622  
0.80       0.8515   0.4621   0.8752   0.9561   0.6380   0.3622  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8515, F1=0.4621, Normal Recall=0.8752, Normal Precision=0.9561, Attack Recall=0.6380, Attack Precision=0.3622

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
0.15       0.8281   0.5975   0.8757   0.9063   0.6379   0.5619   <--
0.20       0.8281   0.5975   0.8757   0.9063   0.6379   0.5619  
0.25       0.8281   0.5975   0.8757   0.9063   0.6379   0.5619  
0.30       0.8281   0.5975   0.8757   0.9063   0.6379   0.5619  
0.35       0.8281   0.5975   0.8757   0.9063   0.6379   0.5619  
0.40       0.8281   0.5975   0.8757   0.9063   0.6379   0.5619  
0.45       0.8281   0.5975   0.8757   0.9063   0.6379   0.5619  
0.50       0.8281   0.5975   0.8757   0.9063   0.6379   0.5619  
0.55       0.8281   0.5975   0.8757   0.9063   0.6379   0.5619  
0.60       0.8281   0.5975   0.8757   0.9063   0.6379   0.5619  
0.65       0.8281   0.5975   0.8757   0.9063   0.6379   0.5619  
0.70       0.8281   0.5975   0.8757   0.9063   0.6379   0.5619  
0.75       0.8281   0.5975   0.8757   0.9063   0.6379   0.5619  
0.80       0.8281   0.5975   0.8757   0.9063   0.6379   0.5619  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8281, F1=0.5975, Normal Recall=0.8757, Normal Precision=0.9063, Attack Recall=0.6379, Attack Precision=0.5619

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
0.15       0.8042   0.6616   0.8755   0.8494   0.6379   0.6871   <--
0.20       0.8042   0.6616   0.8755   0.8494   0.6379   0.6871  
0.25       0.8042   0.6616   0.8755   0.8494   0.6379   0.6871  
0.30       0.8042   0.6616   0.8755   0.8494   0.6379   0.6871  
0.35       0.8042   0.6616   0.8755   0.8494   0.6379   0.6871  
0.40       0.8042   0.6616   0.8755   0.8494   0.6379   0.6871  
0.45       0.8042   0.6616   0.8755   0.8494   0.6379   0.6871  
0.50       0.8042   0.6616   0.8755   0.8494   0.6379   0.6871  
0.55       0.8042   0.6616   0.8755   0.8494   0.6379   0.6871  
0.60       0.8042   0.6616   0.8755   0.8494   0.6379   0.6871  
0.65       0.8042   0.6616   0.8755   0.8494   0.6379   0.6871  
0.70       0.8042   0.6616   0.8755   0.8494   0.6379   0.6871  
0.75       0.8042   0.6616   0.8755   0.8494   0.6379   0.6871  
0.80       0.8042   0.6616   0.8755   0.8494   0.6379   0.6871  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8042, F1=0.6616, Normal Recall=0.8755, Normal Precision=0.8494, Attack Recall=0.6379, Attack Precision=0.6871

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
0.15       0.7810   0.6997   0.8764   0.7840   0.6379   0.7748   <--
0.20       0.7810   0.6997   0.8764   0.7840   0.6379   0.7748  
0.25       0.7810   0.6997   0.8764   0.7840   0.6379   0.7748  
0.30       0.7810   0.6997   0.8764   0.7840   0.6379   0.7748  
0.35       0.7810   0.6997   0.8764   0.7840   0.6379   0.7748  
0.40       0.7810   0.6997   0.8764   0.7840   0.6379   0.7748  
0.45       0.7810   0.6997   0.8764   0.7840   0.6379   0.7748  
0.50       0.7810   0.6997   0.8764   0.7840   0.6379   0.7748  
0.55       0.7810   0.6997   0.8764   0.7840   0.6379   0.7748  
0.60       0.7810   0.6997   0.8764   0.7840   0.6379   0.7748  
0.65       0.7810   0.6997   0.8764   0.7840   0.6379   0.7748  
0.70       0.7810   0.6997   0.8764   0.7840   0.6379   0.7748  
0.75       0.7810   0.6997   0.8764   0.7840   0.6379   0.7748  
0.80       0.7810   0.6997   0.8764   0.7840   0.6379   0.7748  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7810, F1=0.6997, Normal Recall=0.8764, Normal Precision=0.7840, Attack Recall=0.6379, Attack Precision=0.7748

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
0.15       0.7566   0.7238   0.8752   0.7074   0.6379   0.8364   <--
0.20       0.7566   0.7238   0.8752   0.7074   0.6379   0.8364  
0.25       0.7566   0.7238   0.8752   0.7074   0.6379   0.8364  
0.30       0.7566   0.7238   0.8752   0.7074   0.6379   0.8364  
0.35       0.7566   0.7238   0.8752   0.7074   0.6379   0.8364  
0.40       0.7566   0.7238   0.8752   0.7074   0.6379   0.8364  
0.45       0.7566   0.7238   0.8752   0.7074   0.6379   0.8364  
0.50       0.7566   0.7238   0.8752   0.7074   0.6379   0.8364  
0.55       0.7566   0.7238   0.8752   0.7074   0.6379   0.8364  
0.60       0.7566   0.7238   0.8752   0.7074   0.6379   0.8364  
0.65       0.7566   0.7238   0.8752   0.7074   0.6379   0.8364  
0.70       0.7566   0.7238   0.8752   0.7074   0.6379   0.8364  
0.75       0.7566   0.7238   0.8752   0.7074   0.6379   0.8364  
0.80       0.7566   0.7238   0.8752   0.7074   0.6379   0.8364  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7566, F1=0.7238, Normal Recall=0.8752, Normal Precision=0.7074, Attack Recall=0.6379, Attack Precision=0.8364

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
0.15       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609   <--
0.20       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.25       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.30       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.35       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.40       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.45       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.50       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.55       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.60       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.65       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.70       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.75       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.80       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8509, F1=0.4605, Normal Recall=0.8748, Normal Precision=0.9558, Attack Recall=0.6363, Attack Precision=0.3609

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
0.15       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605   <--
0.20       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.25       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.30       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.35       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.40       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.45       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.50       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.55       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.60       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.65       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.70       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.75       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.80       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8275, F1=0.5960, Normal Recall=0.8753, Normal Precision=0.9059, Attack Recall=0.6362, Attack Precision=0.5605

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
0.15       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858   <--
0.20       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.25       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.30       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.35       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.40       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.45       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.50       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.55       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.60       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.65       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.70       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.75       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.80       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8034, F1=0.6601, Normal Recall=0.8751, Normal Precision=0.8488, Attack Recall=0.6363, Attack Precision=0.6858

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
0.15       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738   <--
0.20       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.25       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.30       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.35       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.40       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.45       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.50       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.55       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.60       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.65       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.70       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.75       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.80       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7801, F1=0.6983, Normal Recall=0.8760, Normal Precision=0.7832, Attack Recall=0.6362, Attack Precision=0.7738

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
0.15       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358   <--
0.20       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.25       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.30       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.35       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.40       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.45       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.50       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.55       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.60       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.65       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.70       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.75       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.80       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7556, F1=0.7225, Normal Recall=0.8750, Normal Precision=0.7063, Attack Recall=0.6362, Attack Precision=0.8358

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
0.15       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609   <--
0.20       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.25       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.30       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.35       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.40       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.45       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.50       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.55       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.60       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.65       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.70       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.75       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
0.80       0.8509   0.4605   0.8748   0.9558   0.6363   0.3609  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8509, F1=0.4605, Normal Recall=0.8748, Normal Precision=0.9558, Attack Recall=0.6363, Attack Precision=0.3609

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
0.15       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605   <--
0.20       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.25       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.30       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.35       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.40       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.45       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.50       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.55       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.60       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.65       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.70       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.75       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
0.80       0.8275   0.5960   0.8753   0.9059   0.6362   0.5605  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8275, F1=0.5960, Normal Recall=0.8753, Normal Precision=0.9059, Attack Recall=0.6362, Attack Precision=0.5605

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
0.15       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858   <--
0.20       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.25       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.30       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.35       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.40       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.45       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.50       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.55       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.60       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.65       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.70       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.75       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
0.80       0.8034   0.6601   0.8751   0.8488   0.6363   0.6858  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8034, F1=0.6601, Normal Recall=0.8751, Normal Precision=0.8488, Attack Recall=0.6363, Attack Precision=0.6858

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
0.15       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738   <--
0.20       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.25       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.30       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.35       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.40       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.45       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.50       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.55       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.60       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.65       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.70       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.75       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
0.80       0.7801   0.6983   0.8760   0.7832   0.6362   0.7738  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7801, F1=0.6983, Normal Recall=0.8760, Normal Precision=0.7832, Attack Recall=0.6362, Attack Precision=0.7738

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
0.15       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358   <--
0.20       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.25       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.30       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.35       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.40       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.45       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.50       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.55       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.60       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.65       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.70       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.75       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
0.80       0.7556   0.7225   0.8750   0.7063   0.6362   0.8358  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7556, F1=0.7225, Normal Recall=0.8750, Normal Precision=0.7063, Attack Recall=0.6362, Attack Precision=0.8358

```

