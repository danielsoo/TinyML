# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e1_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-21 01:44:16 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8814 | 0.8841 | 0.8853 | 0.8878 | 0.8886 | 0.8908 | 0.8924 | 0.8945 | 0.8959 | 0.8975 | 0.8994 |
| QAT+Prune only | 0.4451 | 0.4823 | 0.5192 | 0.5564 | 0.5932 | 0.6296 | 0.6657 | 0.7035 | 0.7404 | 0.7771 | 0.8136 |
| QAT+PTQ | 0.4438 | 0.4816 | 0.5192 | 0.5571 | 0.5945 | 0.6316 | 0.6686 | 0.7071 | 0.7447 | 0.7820 | 0.8193 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.4438 | 0.4816 | 0.5192 | 0.5571 | 0.5945 | 0.6316 | 0.6686 | 0.7071 | 0.7447 | 0.7820 | 0.8193 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6080 | 0.7582 | 0.8278 | 0.8659 | 0.8917 | 0.9093 | 0.9227 | 0.9326 | 0.9404 | 0.9470 |
| QAT+Prune only | 0.0000 | 0.2393 | 0.4037 | 0.5239 | 0.6154 | 0.6871 | 0.7449 | 0.7935 | 0.8338 | 0.8679 | 0.8972 |
| QAT+PTQ | 0.0000 | 0.2403 | 0.4053 | 0.5260 | 0.6178 | 0.6898 | 0.7479 | 0.7966 | 0.8370 | 0.8712 | 0.9007 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2403 | 0.4053 | 0.5260 | 0.6178 | 0.6898 | 0.7479 | 0.7966 | 0.8370 | 0.8712 | 0.9007 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8814 | 0.8824 | 0.8817 | 0.8828 | 0.8814 | 0.8822 | 0.8819 | 0.8830 | 0.8823 | 0.8801 | 0.0000 |
| QAT+Prune only | 0.4451 | 0.4455 | 0.4456 | 0.4462 | 0.4462 | 0.4456 | 0.4439 | 0.4466 | 0.4478 | 0.4487 | 0.0000 |
| QAT+PTQ | 0.4438 | 0.4440 | 0.4441 | 0.4447 | 0.4447 | 0.4439 | 0.4425 | 0.4451 | 0.4461 | 0.4461 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.4438 | 0.4440 | 0.4441 | 0.4447 | 0.4447 | 0.4439 | 0.4425 | 0.4451 | 0.4461 | 0.4461 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8814 | 0.0000 | 0.0000 | 0.0000 | 0.8814 | 1.0000 |
| 90 | 10 | 299,940 | 0.8841 | 0.4593 | 0.8990 | 0.6080 | 0.8824 | 0.9874 |
| 80 | 20 | 291,350 | 0.8853 | 0.6553 | 0.8994 | 0.7582 | 0.8817 | 0.9723 |
| 70 | 30 | 194,230 | 0.8878 | 0.7668 | 0.8994 | 0.8278 | 0.8828 | 0.9534 |
| 60 | 40 | 145,675 | 0.8886 | 0.8349 | 0.8994 | 0.8659 | 0.8814 | 0.9293 |
| 50 | 50 | 116,540 | 0.8908 | 0.8842 | 0.8994 | 0.8917 | 0.8822 | 0.8976 |
| 40 | 60 | 97,115 | 0.8924 | 0.9195 | 0.8994 | 0.9093 | 0.8819 | 0.8538 |
| 30 | 70 | 83,240 | 0.8945 | 0.9472 | 0.8994 | 0.9227 | 0.8830 | 0.7899 |
| 20 | 80 | 72,835 | 0.8959 | 0.9683 | 0.8994 | 0.9326 | 0.8823 | 0.6867 |
| 10 | 90 | 64,740 | 0.8975 | 0.9854 | 0.8994 | 0.9404 | 0.8801 | 0.4929 |
| 0 | 100 | 58,270 | 0.8994 | 1.0000 | 0.8994 | 0.9470 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4451 | 0.0000 | 0.0000 | 0.0000 | 0.4451 | 1.0000 |
| 90 | 10 | 299,940 | 0.4823 | 0.1403 | 0.8141 | 0.2393 | 0.4455 | 0.9557 |
| 80 | 20 | 291,350 | 0.5192 | 0.2684 | 0.8136 | 0.4037 | 0.4456 | 0.9053 |
| 70 | 30 | 194,230 | 0.5564 | 0.3864 | 0.8136 | 0.5239 | 0.4462 | 0.8482 |
| 60 | 40 | 145,675 | 0.5932 | 0.4948 | 0.8136 | 0.6154 | 0.4462 | 0.7822 |
| 50 | 50 | 116,540 | 0.6296 | 0.5947 | 0.8136 | 0.6871 | 0.4456 | 0.7050 |
| 40 | 60 | 97,115 | 0.6657 | 0.6870 | 0.8136 | 0.7449 | 0.4439 | 0.6135 |
| 30 | 70 | 83,240 | 0.7035 | 0.7743 | 0.8136 | 0.7935 | 0.4466 | 0.5066 |
| 20 | 80 | 72,835 | 0.7404 | 0.8549 | 0.8136 | 0.8338 | 0.4478 | 0.3752 |
| 10 | 90 | 64,740 | 0.7771 | 0.9300 | 0.8136 | 0.8679 | 0.4487 | 0.2110 |
| 0 | 100 | 58,270 | 0.8136 | 1.0000 | 0.8136 | 0.8972 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4438 | 0.0000 | 0.0000 | 0.0000 | 0.4438 | 1.0000 |
| 90 | 10 | 299,940 | 0.4816 | 0.1408 | 0.8199 | 0.2403 | 0.4440 | 0.9569 |
| 80 | 20 | 291,350 | 0.5192 | 0.2693 | 0.8193 | 0.4053 | 0.4441 | 0.9077 |
| 70 | 30 | 194,230 | 0.5571 | 0.3874 | 0.8193 | 0.5260 | 0.4447 | 0.8517 |
| 60 | 40 | 145,675 | 0.5945 | 0.4959 | 0.8193 | 0.6178 | 0.4447 | 0.7869 |
| 50 | 50 | 116,540 | 0.6316 | 0.5957 | 0.8193 | 0.6898 | 0.4439 | 0.7107 |
| 40 | 60 | 97,115 | 0.6686 | 0.6879 | 0.8193 | 0.7479 | 0.4425 | 0.6202 |
| 30 | 70 | 83,240 | 0.7071 | 0.7751 | 0.8193 | 0.7966 | 0.4451 | 0.5136 |
| 20 | 80 | 72,835 | 0.7447 | 0.8554 | 0.8193 | 0.8370 | 0.4461 | 0.3817 |
| 10 | 90 | 64,740 | 0.7820 | 0.9301 | 0.8193 | 0.8712 | 0.4461 | 0.2153 |
| 0 | 100 | 58,270 | 0.8193 | 1.0000 | 0.8193 | 0.9007 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.4438 | 0.0000 | 0.0000 | 0.0000 | 0.4438 | 1.0000 |
| 90 | 10 | 299,940 | 0.4816 | 0.1408 | 0.8199 | 0.2403 | 0.4440 | 0.9569 |
| 80 | 20 | 291,350 | 0.5192 | 0.2693 | 0.8193 | 0.4053 | 0.4441 | 0.9077 |
| 70 | 30 | 194,230 | 0.5571 | 0.3874 | 0.8193 | 0.5260 | 0.4447 | 0.8517 |
| 60 | 40 | 145,675 | 0.5945 | 0.4959 | 0.8193 | 0.6178 | 0.4447 | 0.7869 |
| 50 | 50 | 116,540 | 0.6316 | 0.5957 | 0.8193 | 0.6898 | 0.4439 | 0.7107 |
| 40 | 60 | 97,115 | 0.6686 | 0.6879 | 0.8193 | 0.7479 | 0.4425 | 0.6202 |
| 30 | 70 | 83,240 | 0.7071 | 0.7751 | 0.8193 | 0.7966 | 0.4451 | 0.5136 |
| 20 | 80 | 72,835 | 0.7447 | 0.8554 | 0.8193 | 0.8370 | 0.4461 | 0.3817 |
| 10 | 90 | 64,740 | 0.7820 | 0.9301 | 0.8193 | 0.8712 | 0.4461 | 0.2153 |
| 0 | 100 | 58,270 | 0.8193 | 1.0000 | 0.8193 | 0.9007 | 0.0000 | 0.0000 |


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
0.15       0.8841   0.6084   0.8824   0.9876   0.8999   0.4595   <--
0.20       0.8841   0.6084   0.8824   0.9876   0.8999   0.4595  
0.25       0.8841   0.6084   0.8824   0.9876   0.8999   0.4595  
0.30       0.8841   0.6084   0.8824   0.9876   0.8999   0.4595  
0.35       0.8841   0.6084   0.8824   0.9876   0.8999   0.4595  
0.40       0.8841   0.6084   0.8824   0.9876   0.8999   0.4595  
0.45       0.8841   0.6084   0.8824   0.9876   0.8999   0.4595  
0.50       0.8841   0.6084   0.8824   0.9876   0.8999   0.4595  
0.55       0.8841   0.6084   0.8824   0.9876   0.8999   0.4595  
0.60       0.8841   0.6084   0.8824   0.9876   0.8999   0.4595  
0.65       0.8841   0.6084   0.8824   0.9876   0.8999   0.4595  
0.70       0.8841   0.6084   0.8824   0.9876   0.8999   0.4595  
0.75       0.8841   0.6084   0.8824   0.9876   0.8999   0.4595  
0.80       0.8841   0.6084   0.8824   0.9876   0.8999   0.4595  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8841, F1=0.6084, Normal Recall=0.8824, Normal Precision=0.9876, Attack Recall=0.8999, Attack Precision=0.4595

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
0.15       0.8859   0.7592   0.8825   0.9723   0.8994   0.6568   <--
0.20       0.8859   0.7592   0.8825   0.9723   0.8994   0.6568  
0.25       0.8859   0.7592   0.8825   0.9723   0.8994   0.6568  
0.30       0.8859   0.7592   0.8825   0.9723   0.8994   0.6568  
0.35       0.8859   0.7592   0.8825   0.9723   0.8994   0.6568  
0.40       0.8859   0.7592   0.8825   0.9723   0.8994   0.6568  
0.45       0.8859   0.7592   0.8825   0.9723   0.8994   0.6568  
0.50       0.8859   0.7592   0.8825   0.9723   0.8994   0.6568  
0.55       0.8859   0.7592   0.8825   0.9723   0.8994   0.6568  
0.60       0.8859   0.7592   0.8825   0.9723   0.8994   0.6568  
0.65       0.8859   0.7592   0.8825   0.9723   0.8994   0.6568  
0.70       0.8859   0.7592   0.8825   0.9723   0.8994   0.6568  
0.75       0.8859   0.7592   0.8825   0.9723   0.8994   0.6568  
0.80       0.8859   0.7592   0.8825   0.9723   0.8994   0.6568  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8859, F1=0.7592, Normal Recall=0.8825, Normal Precision=0.9723, Attack Recall=0.8994, Attack Precision=0.6568

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
0.15       0.8873   0.8273   0.8822   0.9534   0.8994   0.7659   <--
0.20       0.8873   0.8273   0.8822   0.9534   0.8994   0.7659  
0.25       0.8873   0.8273   0.8822   0.9534   0.8994   0.7659  
0.30       0.8873   0.8273   0.8822   0.9534   0.8994   0.7659  
0.35       0.8873   0.8273   0.8822   0.9534   0.8994   0.7659  
0.40       0.8873   0.8273   0.8822   0.9534   0.8994   0.7659  
0.45       0.8873   0.8273   0.8822   0.9534   0.8994   0.7659  
0.50       0.8873   0.8273   0.8822   0.9534   0.8994   0.7659  
0.55       0.8873   0.8273   0.8822   0.9534   0.8994   0.7659  
0.60       0.8873   0.8273   0.8822   0.9534   0.8994   0.7659  
0.65       0.8873   0.8273   0.8822   0.9534   0.8994   0.7659  
0.70       0.8873   0.8273   0.8822   0.9534   0.8994   0.7659  
0.75       0.8873   0.8273   0.8822   0.9534   0.8994   0.7659  
0.80       0.8873   0.8273   0.8822   0.9534   0.8994   0.7659  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8873, F1=0.8273, Normal Recall=0.8822, Normal Precision=0.9534, Attack Recall=0.8994, Attack Precision=0.7659

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
0.15       0.8887   0.8661   0.8816   0.9293   0.8994   0.8351   <--
0.20       0.8887   0.8661   0.8816   0.9293   0.8994   0.8351  
0.25       0.8887   0.8661   0.8816   0.9293   0.8994   0.8351  
0.30       0.8887   0.8661   0.8816   0.9293   0.8994   0.8351  
0.35       0.8887   0.8661   0.8816   0.9293   0.8994   0.8351  
0.40       0.8887   0.8661   0.8816   0.9293   0.8994   0.8351  
0.45       0.8887   0.8661   0.8816   0.9293   0.8994   0.8351  
0.50       0.8887   0.8661   0.8816   0.9293   0.8994   0.8351  
0.55       0.8887   0.8661   0.8816   0.9293   0.8994   0.8351  
0.60       0.8887   0.8661   0.8816   0.9293   0.8994   0.8351  
0.65       0.8887   0.8661   0.8816   0.9293   0.8994   0.8351  
0.70       0.8887   0.8661   0.8816   0.9293   0.8994   0.8351  
0.75       0.8887   0.8661   0.8816   0.9293   0.8994   0.8351  
0.80       0.8887   0.8661   0.8816   0.9293   0.8994   0.8351  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8887, F1=0.8661, Normal Recall=0.8816, Normal Precision=0.9293, Attack Recall=0.8994, Attack Precision=0.8351

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
0.15       0.8902   0.8912   0.8811   0.8975   0.8994   0.8832   <--
0.20       0.8902   0.8912   0.8811   0.8975   0.8994   0.8832  
0.25       0.8902   0.8912   0.8811   0.8975   0.8994   0.8832  
0.30       0.8902   0.8912   0.8811   0.8975   0.8994   0.8832  
0.35       0.8902   0.8912   0.8811   0.8975   0.8994   0.8832  
0.40       0.8902   0.8912   0.8811   0.8975   0.8994   0.8832  
0.45       0.8902   0.8912   0.8811   0.8975   0.8994   0.8832  
0.50       0.8902   0.8912   0.8811   0.8975   0.8994   0.8832  
0.55       0.8902   0.8912   0.8811   0.8975   0.8994   0.8832  
0.60       0.8902   0.8912   0.8811   0.8975   0.8994   0.8832  
0.65       0.8902   0.8912   0.8811   0.8975   0.8994   0.8832  
0.70       0.8902   0.8912   0.8811   0.8975   0.8994   0.8832  
0.75       0.8902   0.8912   0.8811   0.8975   0.8994   0.8832  
0.80       0.8902   0.8912   0.8811   0.8975   0.8994   0.8832  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8902, F1=0.8912, Normal Recall=0.8811, Normal Precision=0.8975, Attack Recall=0.8994, Attack Precision=0.8832

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
0.15       0.4824   0.2394   0.4455   0.9558   0.8148   0.1403   <--
0.20       0.4824   0.2394   0.4455   0.9558   0.8148   0.1403  
0.25       0.4824   0.2394   0.4455   0.9558   0.8148   0.1403  
0.30       0.4824   0.2394   0.4455   0.9558   0.8148   0.1403  
0.35       0.4824   0.2394   0.4455   0.9558   0.8148   0.1403  
0.40       0.4824   0.2394   0.4455   0.9558   0.8148   0.1403  
0.45       0.4824   0.2394   0.4455   0.9558   0.8148   0.1403  
0.50       0.4824   0.2394   0.4455   0.9558   0.8148   0.1403  
0.55       0.4824   0.2394   0.4455   0.9558   0.8148   0.1403  
0.60       0.4824   0.2394   0.4455   0.9558   0.8148   0.1403  
0.65       0.4824   0.2394   0.4455   0.9558   0.8148   0.1403  
0.70       0.4824   0.2394   0.4455   0.9558   0.8148   0.1403  
0.75       0.4824   0.2394   0.4455   0.9558   0.8148   0.1403  
0.80       0.4824   0.2394   0.4455   0.9558   0.8148   0.1403  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4824, F1=0.2394, Normal Recall=0.4455, Normal Precision=0.9558, Attack Recall=0.8148, Attack Precision=0.1403

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
0.15       0.5189   0.4035   0.4452   0.9053   0.8136   0.2683   <--
0.20       0.5189   0.4035   0.4452   0.9053   0.8136   0.2683  
0.25       0.5189   0.4035   0.4452   0.9053   0.8136   0.2683  
0.30       0.5189   0.4035   0.4452   0.9053   0.8136   0.2683  
0.35       0.5189   0.4035   0.4452   0.9053   0.8136   0.2683  
0.40       0.5189   0.4035   0.4452   0.9053   0.8136   0.2683  
0.45       0.5189   0.4035   0.4452   0.9053   0.8136   0.2683  
0.50       0.5189   0.4035   0.4452   0.9053   0.8136   0.2683  
0.55       0.5189   0.4035   0.4452   0.9053   0.8136   0.2683  
0.60       0.5189   0.4035   0.4452   0.9053   0.8136   0.2683  
0.65       0.5189   0.4035   0.4452   0.9053   0.8136   0.2683  
0.70       0.5189   0.4035   0.4452   0.9053   0.8136   0.2683  
0.75       0.5189   0.4035   0.4452   0.9053   0.8136   0.2683  
0.80       0.5189   0.4035   0.4452   0.9053   0.8136   0.2683  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5189, F1=0.4035, Normal Recall=0.4452, Normal Precision=0.9053, Attack Recall=0.8136, Attack Precision=0.2683

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
0.15       0.5558   0.5236   0.4453   0.8479   0.8136   0.3860   <--
0.20       0.5558   0.5236   0.4453   0.8479   0.8136   0.3860  
0.25       0.5558   0.5236   0.4453   0.8479   0.8136   0.3860  
0.30       0.5558   0.5236   0.4453   0.8479   0.8136   0.3860  
0.35       0.5558   0.5236   0.4453   0.8479   0.8136   0.3860  
0.40       0.5558   0.5236   0.4453   0.8479   0.8136   0.3860  
0.45       0.5558   0.5236   0.4453   0.8479   0.8136   0.3860  
0.50       0.5558   0.5236   0.4453   0.8479   0.8136   0.3860  
0.55       0.5558   0.5236   0.4453   0.8479   0.8136   0.3860  
0.60       0.5558   0.5236   0.4453   0.8479   0.8136   0.3860  
0.65       0.5558   0.5236   0.4453   0.8479   0.8136   0.3860  
0.70       0.5558   0.5236   0.4453   0.8479   0.8136   0.3860  
0.75       0.5558   0.5236   0.4453   0.8479   0.8136   0.3860  
0.80       0.5558   0.5236   0.4453   0.8479   0.8136   0.3860  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5558, F1=0.5236, Normal Recall=0.4453, Normal Precision=0.8479, Attack Recall=0.8136, Attack Precision=0.3860

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
0.15       0.5923   0.6148   0.4447   0.7816   0.8136   0.4941   <--
0.20       0.5923   0.6148   0.4447   0.7816   0.8136   0.4941  
0.25       0.5923   0.6148   0.4447   0.7816   0.8136   0.4941  
0.30       0.5923   0.6148   0.4447   0.7816   0.8136   0.4941  
0.35       0.5923   0.6148   0.4447   0.7816   0.8136   0.4941  
0.40       0.5923   0.6148   0.4447   0.7816   0.8136   0.4941  
0.45       0.5923   0.6148   0.4447   0.7816   0.8136   0.4941  
0.50       0.5923   0.6148   0.4447   0.7816   0.8136   0.4941  
0.55       0.5923   0.6148   0.4447   0.7816   0.8136   0.4941  
0.60       0.5923   0.6148   0.4447   0.7816   0.8136   0.4941  
0.65       0.5923   0.6148   0.4447   0.7816   0.8136   0.4941  
0.70       0.5923   0.6148   0.4447   0.7816   0.8136   0.4941  
0.75       0.5923   0.6148   0.4447   0.7816   0.8136   0.4941  
0.80       0.5923   0.6148   0.4447   0.7816   0.8136   0.4941  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5923, F1=0.6148, Normal Recall=0.4447, Normal Precision=0.7816, Attack Recall=0.8136, Attack Precision=0.4941

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
0.15       0.6287   0.6867   0.4439   0.7042   0.8136   0.5940   <--
0.20       0.6287   0.6867   0.4439   0.7042   0.8136   0.5940  
0.25       0.6287   0.6867   0.4439   0.7042   0.8136   0.5940  
0.30       0.6287   0.6867   0.4439   0.7042   0.8136   0.5940  
0.35       0.6287   0.6867   0.4439   0.7042   0.8136   0.5940  
0.40       0.6287   0.6867   0.4439   0.7042   0.8136   0.5940  
0.45       0.6287   0.6867   0.4439   0.7042   0.8136   0.5940  
0.50       0.6287   0.6867   0.4439   0.7042   0.8136   0.5940  
0.55       0.6287   0.6867   0.4439   0.7042   0.8136   0.5940  
0.60       0.6287   0.6867   0.4439   0.7042   0.8136   0.5940  
0.65       0.6287   0.6867   0.4439   0.7042   0.8136   0.5940  
0.70       0.6287   0.6867   0.4439   0.7042   0.8136   0.5940  
0.75       0.6287   0.6867   0.4439   0.7042   0.8136   0.5940  
0.80       0.6287   0.6867   0.4439   0.7042   0.8136   0.5940  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6287, F1=0.6867, Normal Recall=0.4439, Normal Precision=0.7042, Attack Recall=0.8136, Attack Precision=0.5940

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
0.15       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408   <--
0.20       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.25       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.30       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.35       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.40       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.45       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.50       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.55       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.60       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.65       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.70       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.75       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.80       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4816, F1=0.2404, Normal Recall=0.4440, Normal Precision=0.9569, Attack Recall=0.8202, Attack Precision=0.1408

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
0.15       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692   <--
0.20       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.25       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.30       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.35       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.40       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.45       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.50       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.55       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.60       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.65       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.70       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.75       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.80       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5190, F1=0.4052, Normal Recall=0.4439, Normal Precision=0.9076, Attack Recall=0.8193, Attack Precision=0.2692

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
0.15       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870   <--
0.20       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.25       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.30       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.35       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.40       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.45       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.50       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.55       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.60       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.65       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.70       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.75       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.80       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5565, F1=0.5257, Normal Recall=0.4439, Normal Precision=0.8515, Attack Recall=0.8193, Attack Precision=0.3870

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
0.15       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953   <--
0.20       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.25       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.30       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.35       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.40       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.45       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.50       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.55       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.60       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.65       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.70       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.75       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.80       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5937, F1=0.6173, Normal Recall=0.4433, Normal Precision=0.7863, Attack Recall=0.8193, Attack Precision=0.4953

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
0.15       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951   <--
0.20       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.25       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.30       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.35       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.40       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.45       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.50       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.55       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.60       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.65       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.70       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.75       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.80       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6309, F1=0.6894, Normal Recall=0.4424, Normal Precision=0.7100, Attack Recall=0.8193, Attack Precision=0.5951

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
0.15       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408   <--
0.20       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.25       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.30       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.35       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.40       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.45       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.50       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.55       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.60       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.65       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.70       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.75       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
0.80       0.4816   0.2404   0.4440   0.9569   0.8202   0.1408  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4816, F1=0.2404, Normal Recall=0.4440, Normal Precision=0.9569, Attack Recall=0.8202, Attack Precision=0.1408

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
0.15       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692   <--
0.20       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.25       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.30       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.35       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.40       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.45       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.50       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.55       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.60       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.65       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.70       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.75       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
0.80       0.5190   0.4052   0.4439   0.9076   0.8193   0.2692  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5190, F1=0.4052, Normal Recall=0.4439, Normal Precision=0.9076, Attack Recall=0.8193, Attack Precision=0.2692

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
0.15       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870   <--
0.20       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.25       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.30       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.35       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.40       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.45       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.50       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.55       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.60       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.65       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.70       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.75       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
0.80       0.5565   0.5257   0.4439   0.8515   0.8193   0.3870  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5565, F1=0.5257, Normal Recall=0.4439, Normal Precision=0.8515, Attack Recall=0.8193, Attack Precision=0.3870

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
0.15       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953   <--
0.20       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.25       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.30       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.35       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.40       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.45       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.50       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.55       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.60       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.65       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.70       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.75       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
0.80       0.5937   0.6173   0.4433   0.7863   0.8193   0.4953  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5937, F1=0.6173, Normal Recall=0.4433, Normal Precision=0.7863, Attack Recall=0.8193, Attack Precision=0.4953

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
0.15       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951   <--
0.20       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.25       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.30       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.35       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.40       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.45       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.50       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.55       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.60       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.65       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.70       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.75       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
0.80       0.6309   0.6894   0.4424   0.7100   0.8193   0.5951  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6309, F1=0.6894, Normal Recall=0.4424, Normal Precision=0.7100, Attack Recall=0.8193, Attack Precision=0.5951

```

