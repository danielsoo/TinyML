# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-12 14:27:59 |

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
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8965 | 0.9065 | 0.9156 | 0.9260 | 0.9349 | 0.9447 | 0.9543 | 0.9642 | 0.9733 | 0.9825 | 0.9923 |
| QAT+Prune only | 0.7750 | 0.7967 | 0.8186 | 0.8419 | 0.8642 | 0.8848 | 0.9090 | 0.9302 | 0.9528 | 0.9754 | 0.9981 |
| QAT+PTQ | 0.7753 | 0.7970 | 0.8189 | 0.8421 | 0.8645 | 0.8849 | 0.9092 | 0.9303 | 0.9529 | 0.9755 | 0.9981 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7753 | 0.7970 | 0.8189 | 0.8421 | 0.8645 | 0.8849 | 0.9092 | 0.9303 | 0.9529 | 0.9755 | 0.9981 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6799 | 0.8246 | 0.8894 | 0.9243 | 0.9472 | 0.9631 | 0.9749 | 0.9835 | 0.9903 | 0.9962 |
| QAT+Prune only | 0.0000 | 0.4955 | 0.6875 | 0.7911 | 0.8547 | 0.8965 | 0.9294 | 0.9524 | 0.9713 | 0.9865 | 0.9990 |
| QAT+PTQ | 0.0000 | 0.4958 | 0.6879 | 0.7914 | 0.8550 | 0.8966 | 0.9295 | 0.9525 | 0.9713 | 0.9865 | 0.9990 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4958 | 0.6879 | 0.7914 | 0.8550 | 0.8966 | 0.9295 | 0.9525 | 0.9713 | 0.9865 | 0.9990 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8965 | 0.8969 | 0.8964 | 0.8975 | 0.8967 | 0.8971 | 0.8973 | 0.8986 | 0.8973 | 0.8934 | 0.0000 |
| QAT+Prune only | 0.7750 | 0.7743 | 0.7737 | 0.7749 | 0.7750 | 0.7714 | 0.7755 | 0.7719 | 0.7717 | 0.7717 | 0.0000 |
| QAT+PTQ | 0.7753 | 0.7747 | 0.7741 | 0.7753 | 0.7755 | 0.7718 | 0.7758 | 0.7723 | 0.7721 | 0.7720 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7753 | 0.7747 | 0.7741 | 0.7753 | 0.7755 | 0.7718 | 0.7758 | 0.7723 | 0.7721 | 0.7720 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8965 | 0.0000 | 0.0000 | 0.0000 | 0.8965 | 1.0000 |
| 90 | 10 | 299,940 | 0.9065 | 0.5169 | 0.9929 | 0.6799 | 0.8969 | 0.9991 |
| 80 | 20 | 291,350 | 0.9156 | 0.7054 | 0.9923 | 0.8246 | 0.8964 | 0.9979 |
| 70 | 30 | 194,230 | 0.9260 | 0.8058 | 0.9923 | 0.8894 | 0.8975 | 0.9964 |
| 60 | 40 | 145,675 | 0.9349 | 0.8649 | 0.9923 | 0.9243 | 0.8967 | 0.9943 |
| 50 | 50 | 116,540 | 0.9447 | 0.9061 | 0.9923 | 0.9472 | 0.8971 | 0.9915 |
| 40 | 60 | 97,115 | 0.9543 | 0.9354 | 0.9923 | 0.9631 | 0.8973 | 0.9874 |
| 30 | 70 | 83,240 | 0.9642 | 0.9581 | 0.9923 | 0.9749 | 0.8986 | 0.9805 |
| 20 | 80 | 72,835 | 0.9733 | 0.9748 | 0.9923 | 0.9835 | 0.8973 | 0.9670 |
| 10 | 90 | 64,740 | 0.9825 | 0.9882 | 0.9923 | 0.9903 | 0.8934 | 0.9284 |
| 0 | 100 | 58,270 | 0.9923 | 1.0000 | 0.9923 | 0.9962 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7750 | 0.0000 | 0.0000 | 0.0000 | 0.7750 | 1.0000 |
| 90 | 10 | 299,940 | 0.7967 | 0.3295 | 0.9981 | 0.4955 | 0.7743 | 0.9997 |
| 80 | 20 | 291,350 | 0.8186 | 0.5244 | 0.9981 | 0.6875 | 0.7737 | 0.9994 |
| 70 | 30 | 194,230 | 0.8419 | 0.6552 | 0.9981 | 0.7911 | 0.7749 | 0.9989 |
| 60 | 40 | 145,675 | 0.8642 | 0.7473 | 0.9981 | 0.8547 | 0.7750 | 0.9983 |
| 50 | 50 | 116,540 | 0.8848 | 0.8137 | 0.9981 | 0.8965 | 0.7714 | 0.9975 |
| 40 | 60 | 97,115 | 0.9090 | 0.8696 | 0.9981 | 0.9294 | 0.7755 | 0.9963 |
| 30 | 70 | 83,240 | 0.9302 | 0.9108 | 0.9981 | 0.9524 | 0.7719 | 0.9942 |
| 20 | 80 | 72,835 | 0.9528 | 0.9459 | 0.9981 | 0.9713 | 0.7717 | 0.9901 |
| 10 | 90 | 64,740 | 0.9754 | 0.9752 | 0.9981 | 0.9865 | 0.7717 | 0.9781 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9990 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7753 | 0.0000 | 0.0000 | 0.0000 | 0.7753 | 1.0000 |
| 90 | 10 | 299,940 | 0.7970 | 0.3299 | 0.9981 | 0.4958 | 0.7747 | 0.9997 |
| 80 | 20 | 291,350 | 0.8189 | 0.5248 | 0.9981 | 0.6879 | 0.7741 | 0.9994 |
| 70 | 30 | 194,230 | 0.8421 | 0.6556 | 0.9981 | 0.7914 | 0.7753 | 0.9989 |
| 60 | 40 | 145,675 | 0.8645 | 0.7477 | 0.9981 | 0.8550 | 0.7755 | 0.9984 |
| 50 | 50 | 116,540 | 0.8849 | 0.8139 | 0.9981 | 0.8966 | 0.7718 | 0.9975 |
| 40 | 60 | 97,115 | 0.9092 | 0.8698 | 0.9981 | 0.9295 | 0.7758 | 0.9963 |
| 30 | 70 | 83,240 | 0.9303 | 0.9109 | 0.9981 | 0.9525 | 0.7723 | 0.9942 |
| 20 | 80 | 72,835 | 0.9529 | 0.9460 | 0.9981 | 0.9713 | 0.7721 | 0.9901 |
| 10 | 90 | 64,740 | 0.9755 | 0.9752 | 0.9981 | 0.9865 | 0.7720 | 0.9781 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9990 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7753 | 0.0000 | 0.0000 | 0.0000 | 0.7753 | 1.0000 |
| 90 | 10 | 299,940 | 0.7970 | 0.3299 | 0.9981 | 0.4958 | 0.7747 | 0.9997 |
| 80 | 20 | 291,350 | 0.8189 | 0.5248 | 0.9981 | 0.6879 | 0.7741 | 0.9994 |
| 70 | 30 | 194,230 | 0.8421 | 0.6556 | 0.9981 | 0.7914 | 0.7753 | 0.9989 |
| 60 | 40 | 145,675 | 0.8645 | 0.7477 | 0.9981 | 0.8550 | 0.7755 | 0.9984 |
| 50 | 50 | 116,540 | 0.8849 | 0.8139 | 0.9981 | 0.8966 | 0.7718 | 0.9975 |
| 40 | 60 | 97,115 | 0.9092 | 0.8698 | 0.9981 | 0.9295 | 0.7758 | 0.9963 |
| 30 | 70 | 83,240 | 0.9303 | 0.9109 | 0.9981 | 0.9525 | 0.7723 | 0.9942 |
| 20 | 80 | 72,835 | 0.9529 | 0.9460 | 0.9981 | 0.9713 | 0.7721 | 0.9901 |
| 10 | 90 | 64,740 | 0.9755 | 0.9752 | 0.9981 | 0.9865 | 0.7720 | 0.9781 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9990 | 0.0000 | 0.0000 |


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
0.15       0.9065   0.6798   0.8969   0.9991   0.9926   0.5169   <--
0.20       0.9065   0.6798   0.8969   0.9991   0.9926   0.5169  
0.25       0.9065   0.6798   0.8969   0.9991   0.9926   0.5169  
0.30       0.9065   0.6798   0.8969   0.9991   0.9926   0.5169  
0.35       0.9065   0.6798   0.8969   0.9991   0.9926   0.5169  
0.40       0.9065   0.6798   0.8969   0.9991   0.9926   0.5169  
0.45       0.9065   0.6798   0.8969   0.9991   0.9926   0.5169  
0.50       0.9065   0.6798   0.8969   0.9991   0.9926   0.5169  
0.55       0.9065   0.6798   0.8969   0.9991   0.9926   0.5169  
0.60       0.9065   0.6798   0.8969   0.9991   0.9926   0.5169  
0.65       0.9065   0.6798   0.8969   0.9991   0.9926   0.5169  
0.70       0.9065   0.6798   0.8969   0.9991   0.9926   0.5169  
0.75       0.9065   0.6798   0.8969   0.9991   0.9926   0.5169  
0.80       0.9065   0.6798   0.8969   0.9991   0.9926   0.5169  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9065, F1=0.6798, Normal Recall=0.8969, Normal Precision=0.9991, Attack Recall=0.9926, Attack Precision=0.5169

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
0.15       0.9159   0.8252   0.8968   0.9979   0.9923   0.7063   <--
0.20       0.9159   0.8252   0.8968   0.9979   0.9923   0.7063  
0.25       0.9159   0.8252   0.8968   0.9979   0.9923   0.7063  
0.30       0.9159   0.8252   0.8968   0.9979   0.9923   0.7063  
0.35       0.9159   0.8252   0.8968   0.9979   0.9923   0.7063  
0.40       0.9159   0.8252   0.8968   0.9979   0.9923   0.7063  
0.45       0.9159   0.8252   0.8968   0.9979   0.9923   0.7063  
0.50       0.9159   0.8252   0.8968   0.9979   0.9923   0.7063  
0.55       0.9159   0.8252   0.8968   0.9979   0.9923   0.7063  
0.60       0.9159   0.8252   0.8968   0.9979   0.9923   0.7063  
0.65       0.9159   0.8252   0.8968   0.9979   0.9923   0.7063  
0.70       0.9159   0.8252   0.8968   0.9979   0.9923   0.7063  
0.75       0.9159   0.8252   0.8968   0.9979   0.9923   0.7063  
0.80       0.9159   0.8252   0.8968   0.9979   0.9923   0.7063  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9159, F1=0.8252, Normal Recall=0.8968, Normal Precision=0.9979, Attack Recall=0.9923, Attack Precision=0.7063

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
0.15       0.9257   0.8891   0.8972   0.9964   0.9923   0.8053   <--
0.20       0.9257   0.8891   0.8972   0.9964   0.9923   0.8053  
0.25       0.9257   0.8891   0.8972   0.9964   0.9923   0.8053  
0.30       0.9257   0.8891   0.8972   0.9964   0.9923   0.8053  
0.35       0.9257   0.8891   0.8972   0.9964   0.9923   0.8053  
0.40       0.9257   0.8891   0.8972   0.9964   0.9923   0.8053  
0.45       0.9257   0.8891   0.8972   0.9964   0.9923   0.8053  
0.50       0.9257   0.8891   0.8972   0.9964   0.9923   0.8053  
0.55       0.9257   0.8891   0.8972   0.9964   0.9923   0.8053  
0.60       0.9257   0.8891   0.8972   0.9964   0.9923   0.8053  
0.65       0.9257   0.8891   0.8972   0.9964   0.9923   0.8053  
0.70       0.9257   0.8891   0.8972   0.9964   0.9923   0.8053  
0.75       0.9257   0.8891   0.8972   0.9964   0.9923   0.8053  
0.80       0.9257   0.8891   0.8972   0.9964   0.9923   0.8053  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9257, F1=0.8891, Normal Recall=0.8972, Normal Precision=0.9964, Attack Recall=0.9923, Attack Precision=0.8053

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
0.15       0.9353   0.9247   0.8973   0.9943   0.9923   0.8656   <--
0.20       0.9353   0.9247   0.8973   0.9943   0.9923   0.8656  
0.25       0.9353   0.9247   0.8973   0.9943   0.9923   0.8656  
0.30       0.9353   0.9247   0.8973   0.9943   0.9923   0.8656  
0.35       0.9353   0.9247   0.8973   0.9943   0.9923   0.8656  
0.40       0.9353   0.9247   0.8973   0.9943   0.9923   0.8656  
0.45       0.9353   0.9247   0.8973   0.9943   0.9923   0.8656  
0.50       0.9353   0.9247   0.8973   0.9943   0.9923   0.8656  
0.55       0.9353   0.9247   0.8973   0.9943   0.9923   0.8656  
0.60       0.9353   0.9247   0.8973   0.9943   0.9923   0.8656  
0.65       0.9353   0.9247   0.8973   0.9943   0.9923   0.8656  
0.70       0.9353   0.9247   0.8973   0.9943   0.9923   0.8656  
0.75       0.9353   0.9247   0.8973   0.9943   0.9923   0.8656  
0.80       0.9353   0.9247   0.8973   0.9943   0.9923   0.8656  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9353, F1=0.9247, Normal Recall=0.8973, Normal Precision=0.9943, Attack Recall=0.9923, Attack Precision=0.8656

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
0.15       0.9449   0.9474   0.8974   0.9915   0.9923   0.9063   <--
0.20       0.9449   0.9474   0.8974   0.9915   0.9923   0.9063  
0.25       0.9449   0.9474   0.8974   0.9915   0.9923   0.9063  
0.30       0.9449   0.9474   0.8974   0.9915   0.9923   0.9063  
0.35       0.9449   0.9474   0.8974   0.9915   0.9923   0.9063  
0.40       0.9449   0.9474   0.8974   0.9915   0.9923   0.9063  
0.45       0.9449   0.9474   0.8974   0.9915   0.9923   0.9063  
0.50       0.9449   0.9474   0.8974   0.9915   0.9923   0.9063  
0.55       0.9449   0.9474   0.8974   0.9915   0.9923   0.9063  
0.60       0.9449   0.9474   0.8974   0.9915   0.9923   0.9063  
0.65       0.9449   0.9474   0.8974   0.9915   0.9923   0.9063  
0.70       0.9449   0.9474   0.8974   0.9915   0.9923   0.9063  
0.75       0.9449   0.9474   0.8974   0.9915   0.9923   0.9063  
0.80       0.9449   0.9474   0.8974   0.9915   0.9923   0.9063  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9449, F1=0.9474, Normal Recall=0.8974, Normal Precision=0.9915, Attack Recall=0.9923, Attack Precision=0.9063

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
0.15       0.7967   0.4955   0.7743   0.9998   0.9983   0.3296   <--
0.20       0.7967   0.4955   0.7743   0.9998   0.9983   0.3296  
0.25       0.7967   0.4955   0.7743   0.9998   0.9983   0.3296  
0.30       0.7967   0.4955   0.7743   0.9998   0.9983   0.3296  
0.35       0.7967   0.4955   0.7743   0.9998   0.9983   0.3296  
0.40       0.7967   0.4955   0.7743   0.9998   0.9983   0.3296  
0.45       0.7967   0.4955   0.7743   0.9998   0.9983   0.3296  
0.50       0.7967   0.4955   0.7743   0.9998   0.9983   0.3296  
0.55       0.7967   0.4955   0.7743   0.9998   0.9983   0.3296  
0.60       0.7967   0.4955   0.7743   0.9998   0.9983   0.3296  
0.65       0.7967   0.4955   0.7743   0.9998   0.9983   0.3296  
0.70       0.7967   0.4955   0.7743   0.9998   0.9983   0.3296  
0.75       0.7967   0.4955   0.7743   0.9998   0.9983   0.3296  
0.80       0.7967   0.4955   0.7743   0.9998   0.9983   0.3296  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7967, F1=0.4955, Normal Recall=0.7743, Normal Precision=0.9998, Attack Recall=0.9983, Attack Precision=0.3296

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
0.15       0.8197   0.6889   0.7752   0.9994   0.9981   0.5260   <--
0.20       0.8197   0.6889   0.7752   0.9994   0.9981   0.5260  
0.25       0.8197   0.6889   0.7752   0.9994   0.9981   0.5260  
0.30       0.8197   0.6889   0.7752   0.9994   0.9981   0.5260  
0.35       0.8197   0.6889   0.7752   0.9994   0.9981   0.5260  
0.40       0.8197   0.6889   0.7752   0.9994   0.9981   0.5260  
0.45       0.8197   0.6889   0.7752   0.9994   0.9981   0.5260  
0.50       0.8197   0.6889   0.7752   0.9994   0.9981   0.5260  
0.55       0.8197   0.6889   0.7752   0.9994   0.9981   0.5260  
0.60       0.8197   0.6889   0.7752   0.9994   0.9981   0.5260  
0.65       0.8197   0.6889   0.7752   0.9994   0.9981   0.5260  
0.70       0.8197   0.6889   0.7752   0.9994   0.9981   0.5260  
0.75       0.8197   0.6889   0.7752   0.9994   0.9981   0.5260  
0.80       0.8197   0.6889   0.7752   0.9994   0.9981   0.5260  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8197, F1=0.6889, Normal Recall=0.7752, Normal Precision=0.9994, Attack Recall=0.9981, Attack Precision=0.5260

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
0.15       0.8421   0.7913   0.7752   0.9989   0.9981   0.6555   <--
0.20       0.8421   0.7913   0.7752   0.9989   0.9981   0.6555  
0.25       0.8421   0.7913   0.7752   0.9989   0.9981   0.6555  
0.30       0.8421   0.7913   0.7752   0.9989   0.9981   0.6555  
0.35       0.8421   0.7913   0.7752   0.9989   0.9981   0.6555  
0.40       0.8421   0.7913   0.7752   0.9989   0.9981   0.6555  
0.45       0.8421   0.7913   0.7752   0.9989   0.9981   0.6555  
0.50       0.8421   0.7913   0.7752   0.9989   0.9981   0.6555  
0.55       0.8421   0.7913   0.7752   0.9989   0.9981   0.6555  
0.60       0.8421   0.7913   0.7752   0.9989   0.9981   0.6555  
0.65       0.8421   0.7913   0.7752   0.9989   0.9981   0.6555  
0.70       0.8421   0.7913   0.7752   0.9989   0.9981   0.6555  
0.75       0.8421   0.7913   0.7752   0.9989   0.9981   0.6555  
0.80       0.8421   0.7913   0.7752   0.9989   0.9981   0.6555  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8421, F1=0.7913, Normal Recall=0.7752, Normal Precision=0.9989, Attack Recall=0.9981, Attack Precision=0.6555

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
0.15       0.8642   0.8546   0.7749   0.9983   0.9981   0.7472   <--
0.20       0.8642   0.8546   0.7749   0.9983   0.9981   0.7472  
0.25       0.8642   0.8546   0.7749   0.9983   0.9981   0.7472  
0.30       0.8642   0.8546   0.7749   0.9983   0.9981   0.7472  
0.35       0.8642   0.8546   0.7749   0.9983   0.9981   0.7472  
0.40       0.8642   0.8546   0.7749   0.9983   0.9981   0.7472  
0.45       0.8642   0.8546   0.7749   0.9983   0.9981   0.7472  
0.50       0.8642   0.8546   0.7749   0.9983   0.9981   0.7472  
0.55       0.8642   0.8546   0.7749   0.9983   0.9981   0.7472  
0.60       0.8642   0.8546   0.7749   0.9983   0.9981   0.7472  
0.65       0.8642   0.8546   0.7749   0.9983   0.9981   0.7472  
0.70       0.8642   0.8546   0.7749   0.9983   0.9981   0.7472  
0.75       0.8642   0.8546   0.7749   0.9983   0.9981   0.7472  
0.80       0.8642   0.8546   0.7749   0.9983   0.9981   0.7472  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8642, F1=0.8546, Normal Recall=0.7749, Normal Precision=0.9983, Attack Recall=0.9981, Attack Precision=0.7472

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
0.15       0.8857   0.8972   0.7732   0.9975   0.9981   0.8149   <--
0.20       0.8857   0.8972   0.7732   0.9975   0.9981   0.8149  
0.25       0.8857   0.8972   0.7732   0.9975   0.9981   0.8149  
0.30       0.8857   0.8972   0.7732   0.9975   0.9981   0.8149  
0.35       0.8857   0.8972   0.7732   0.9975   0.9981   0.8149  
0.40       0.8857   0.8972   0.7732   0.9975   0.9981   0.8149  
0.45       0.8857   0.8972   0.7732   0.9975   0.9981   0.8149  
0.50       0.8857   0.8972   0.7732   0.9975   0.9981   0.8149  
0.55       0.8857   0.8972   0.7732   0.9975   0.9981   0.8149  
0.60       0.8857   0.8972   0.7732   0.9975   0.9981   0.8149  
0.65       0.8857   0.8972   0.7732   0.9975   0.9981   0.8149  
0.70       0.8857   0.8972   0.7732   0.9975   0.9981   0.8149  
0.75       0.8857   0.8972   0.7732   0.9975   0.9981   0.8149  
0.80       0.8857   0.8972   0.7732   0.9975   0.9981   0.8149  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8857, F1=0.8972, Normal Recall=0.7732, Normal Precision=0.9975, Attack Recall=0.9981, Attack Precision=0.8149

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
0.15       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299   <--
0.20       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.25       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.30       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.35       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.40       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.45       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.50       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.55       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.60       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.65       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.70       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.75       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.80       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7971, F1=0.4959, Normal Recall=0.7747, Normal Precision=0.9998, Attack Recall=0.9983, Attack Precision=0.3299

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
0.15       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264   <--
0.20       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.25       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.30       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.35       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.40       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.45       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.50       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.55       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.60       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.65       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.70       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.75       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.80       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8200, F1=0.6893, Normal Recall=0.7755, Normal Precision=0.9994, Attack Recall=0.9981, Attack Precision=0.5264

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
0.15       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559   <--
0.20       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.25       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.30       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.35       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.40       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.45       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.50       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.55       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.60       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.65       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.70       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.75       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.80       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8423, F1=0.7916, Normal Recall=0.7756, Normal Precision=0.9989, Attack Recall=0.9981, Attack Precision=0.6559

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
0.15       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475   <--
0.20       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.25       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.30       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.35       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.40       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.45       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.50       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.55       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.60       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.65       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.70       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.75       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.80       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8644, F1=0.8548, Normal Recall=0.7752, Normal Precision=0.9983, Attack Recall=0.9981, Attack Precision=0.7475

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
0.15       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151   <--
0.20       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.25       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.30       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.35       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.40       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.45       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.50       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.55       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.60       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.65       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.70       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.75       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.80       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8858, F1=0.8973, Normal Recall=0.7735, Normal Precision=0.9975, Attack Recall=0.9981, Attack Precision=0.8151

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
0.15       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299   <--
0.20       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.25       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.30       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.35       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.40       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.45       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.50       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.55       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.60       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.65       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.70       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.75       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
0.80       0.7971   0.4959   0.7747   0.9998   0.9983   0.3299  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7971, F1=0.4959, Normal Recall=0.7747, Normal Precision=0.9998, Attack Recall=0.9983, Attack Precision=0.3299

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
0.15       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264   <--
0.20       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.25       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.30       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.35       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.40       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.45       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.50       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.55       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.60       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.65       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.70       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.75       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
0.80       0.8200   0.6893   0.7755   0.9994   0.9981   0.5264  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8200, F1=0.6893, Normal Recall=0.7755, Normal Precision=0.9994, Attack Recall=0.9981, Attack Precision=0.5264

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
0.15       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559   <--
0.20       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.25       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.30       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.35       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.40       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.45       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.50       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.55       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.60       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.65       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.70       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.75       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
0.80       0.8423   0.7916   0.7756   0.9989   0.9981   0.6559  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8423, F1=0.7916, Normal Recall=0.7756, Normal Precision=0.9989, Attack Recall=0.9981, Attack Precision=0.6559

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
0.15       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475   <--
0.20       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.25       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.30       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.35       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.40       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.45       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.50       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.55       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.60       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.65       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.70       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.75       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
0.80       0.8644   0.8548   0.7752   0.9983   0.9981   0.7475  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8644, F1=0.8548, Normal Recall=0.7752, Normal Precision=0.9983, Attack Recall=0.9981, Attack Precision=0.7475

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
0.15       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151   <--
0.20       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.25       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.30       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.35       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.40       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.45       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.50       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.55       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.60       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.65       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.70       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.75       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
0.80       0.8858   0.8973   0.7735   0.9975   0.9981   0.8151  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8858, F1=0.8973, Normal Recall=0.7735, Normal Precision=0.9975, Attack Recall=0.9981, Attack Precision=0.8151

```

