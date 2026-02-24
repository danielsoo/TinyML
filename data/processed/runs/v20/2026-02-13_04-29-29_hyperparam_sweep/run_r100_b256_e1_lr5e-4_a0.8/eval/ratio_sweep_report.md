# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e1_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-22 07:25:31 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8925 | 0.9028 | 0.9128 | 0.9236 | 0.9335 | 0.9434 | 0.9541 | 0.9647 | 0.9747 | 0.9847 | 0.9953 |
| QAT+Prune only | 0.6010 | 0.6407 | 0.6791 | 0.7175 | 0.7572 | 0.7956 | 0.8355 | 0.8737 | 0.9140 | 0.9508 | 0.9909 |
| QAT+PTQ | 0.5981 | 0.6383 | 0.6772 | 0.7162 | 0.7562 | 0.7950 | 0.8352 | 0.8741 | 0.9148 | 0.9521 | 0.9925 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.5981 | 0.6383 | 0.6772 | 0.7162 | 0.7562 | 0.7950 | 0.8352 | 0.8741 | 0.9148 | 0.9521 | 0.9925 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6720 | 0.8202 | 0.8866 | 0.9230 | 0.9462 | 0.9630 | 0.9753 | 0.9844 | 0.9915 | 0.9976 |
| QAT+Prune only | 0.0000 | 0.3555 | 0.5526 | 0.6779 | 0.7655 | 0.8290 | 0.8785 | 0.9166 | 0.9486 | 0.9732 | 0.9954 |
| QAT+PTQ | 0.0000 | 0.3544 | 0.5515 | 0.6773 | 0.7651 | 0.8288 | 0.8784 | 0.9169 | 0.9491 | 0.9739 | 0.9962 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3544 | 0.5515 | 0.6773 | 0.7651 | 0.8288 | 0.8784 | 0.9169 | 0.9491 | 0.9739 | 0.9962 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8925 | 0.8926 | 0.8921 | 0.8929 | 0.8924 | 0.8915 | 0.8923 | 0.8935 | 0.8923 | 0.8897 | 0.0000 |
| QAT+Prune only | 0.6010 | 0.6018 | 0.6012 | 0.6004 | 0.6014 | 0.6003 | 0.6025 | 0.6005 | 0.6068 | 0.5905 | 0.0000 |
| QAT+PTQ | 0.5981 | 0.5990 | 0.5984 | 0.5978 | 0.5987 | 0.5975 | 0.5992 | 0.5979 | 0.6038 | 0.5882 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.5981 | 0.5990 | 0.5984 | 0.5978 | 0.5987 | 0.5975 | 0.5992 | 0.5979 | 0.6038 | 0.5882 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8925 | 0.0000 | 0.0000 | 0.0000 | 0.8925 | 1.0000 |
| 90 | 10 | 299,940 | 0.9028 | 0.5073 | 0.9954 | 0.6720 | 0.8926 | 0.9994 |
| 80 | 20 | 291,350 | 0.9128 | 0.6976 | 0.9953 | 0.8202 | 0.8921 | 0.9987 |
| 70 | 30 | 194,230 | 0.9236 | 0.7993 | 0.9953 | 0.8866 | 0.8929 | 0.9977 |
| 60 | 40 | 145,675 | 0.9335 | 0.8604 | 0.9953 | 0.9230 | 0.8924 | 0.9965 |
| 50 | 50 | 116,540 | 0.9434 | 0.9017 | 0.9953 | 0.9462 | 0.8915 | 0.9947 |
| 40 | 60 | 97,115 | 0.9541 | 0.9327 | 0.9953 | 0.9630 | 0.8923 | 0.9921 |
| 30 | 70 | 83,240 | 0.9647 | 0.9561 | 0.9953 | 0.9753 | 0.8935 | 0.9878 |
| 20 | 80 | 72,835 | 0.9747 | 0.9737 | 0.9953 | 0.9844 | 0.8923 | 0.9793 |
| 10 | 90 | 64,740 | 0.9847 | 0.9878 | 0.9953 | 0.9915 | 0.8897 | 0.9544 |
| 0 | 100 | 58,270 | 0.9953 | 1.0000 | 0.9953 | 0.9976 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6010 | 0.0000 | 0.0000 | 0.0000 | 0.6010 | 1.0000 |
| 90 | 10 | 299,940 | 0.6407 | 0.2166 | 0.9910 | 0.3555 | 0.6018 | 0.9983 |
| 80 | 20 | 291,350 | 0.6791 | 0.3831 | 0.9909 | 0.5526 | 0.6012 | 0.9962 |
| 70 | 30 | 194,230 | 0.7175 | 0.5152 | 0.9909 | 0.6779 | 0.6004 | 0.9935 |
| 60 | 40 | 145,675 | 0.7572 | 0.6237 | 0.9909 | 0.7655 | 0.6014 | 0.9900 |
| 50 | 50 | 116,540 | 0.7956 | 0.7125 | 0.9909 | 0.8290 | 0.6003 | 0.9850 |
| 40 | 60 | 97,115 | 0.8355 | 0.7890 | 0.9909 | 0.8785 | 0.6025 | 0.9777 |
| 30 | 70 | 83,240 | 0.8737 | 0.8527 | 0.9909 | 0.9166 | 0.6005 | 0.9657 |
| 20 | 80 | 72,835 | 0.9140 | 0.9097 | 0.9909 | 0.9486 | 0.6068 | 0.9431 |
| 10 | 90 | 64,740 | 0.9508 | 0.9561 | 0.9909 | 0.9732 | 0.5905 | 0.8776 |
| 0 | 100 | 58,270 | 0.9909 | 1.0000 | 0.9909 | 0.9954 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5981 | 0.0000 | 0.0000 | 0.0000 | 0.5981 | 1.0000 |
| 90 | 10 | 299,940 | 0.6383 | 0.2157 | 0.9925 | 0.3544 | 0.5990 | 0.9986 |
| 80 | 20 | 291,350 | 0.6772 | 0.3819 | 0.9925 | 0.5515 | 0.5984 | 0.9969 |
| 70 | 30 | 194,230 | 0.7162 | 0.5140 | 0.9925 | 0.6773 | 0.5978 | 0.9947 |
| 60 | 40 | 145,675 | 0.7562 | 0.6225 | 0.9925 | 0.7651 | 0.5987 | 0.9917 |
| 50 | 50 | 116,540 | 0.7950 | 0.7115 | 0.9925 | 0.8288 | 0.5975 | 0.9876 |
| 40 | 60 | 97,115 | 0.8352 | 0.7879 | 0.9925 | 0.8784 | 0.5992 | 0.9816 |
| 30 | 70 | 83,240 | 0.8741 | 0.8521 | 0.9925 | 0.9169 | 0.5979 | 0.9716 |
| 20 | 80 | 72,835 | 0.9148 | 0.9093 | 0.9925 | 0.9491 | 0.6038 | 0.9528 |
| 10 | 90 | 64,740 | 0.9521 | 0.9559 | 0.9925 | 0.9739 | 0.5882 | 0.8973 |
| 0 | 100 | 58,270 | 0.9925 | 1.0000 | 0.9925 | 0.9962 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.5981 | 0.0000 | 0.0000 | 0.0000 | 0.5981 | 1.0000 |
| 90 | 10 | 299,940 | 0.6383 | 0.2157 | 0.9925 | 0.3544 | 0.5990 | 0.9986 |
| 80 | 20 | 291,350 | 0.6772 | 0.3819 | 0.9925 | 0.5515 | 0.5984 | 0.9969 |
| 70 | 30 | 194,230 | 0.7162 | 0.5140 | 0.9925 | 0.6773 | 0.5978 | 0.9947 |
| 60 | 40 | 145,675 | 0.7562 | 0.6225 | 0.9925 | 0.7651 | 0.5987 | 0.9917 |
| 50 | 50 | 116,540 | 0.7950 | 0.7115 | 0.9925 | 0.8288 | 0.5975 | 0.9876 |
| 40 | 60 | 97,115 | 0.8352 | 0.7879 | 0.9925 | 0.8784 | 0.5992 | 0.9816 |
| 30 | 70 | 83,240 | 0.8741 | 0.8521 | 0.9925 | 0.9169 | 0.5979 | 0.9716 |
| 20 | 80 | 72,835 | 0.9148 | 0.9093 | 0.9925 | 0.9491 | 0.6038 | 0.9528 |
| 10 | 90 | 64,740 | 0.9521 | 0.9559 | 0.9925 | 0.9739 | 0.5882 | 0.8973 |
| 0 | 100 | 58,270 | 0.9925 | 1.0000 | 0.9925 | 0.9962 | 0.0000 | 0.0000 |


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
0.15       0.9029   0.6722   0.8926   0.9995   0.9957   0.5073   <--
0.20       0.9029   0.6722   0.8926   0.9995   0.9957   0.5073  
0.25       0.9029   0.6722   0.8926   0.9995   0.9957   0.5073  
0.30       0.9029   0.6722   0.8926   0.9995   0.9957   0.5073  
0.35       0.9029   0.6722   0.8926   0.9995   0.9957   0.5073  
0.40       0.9029   0.6722   0.8926   0.9995   0.9957   0.5073  
0.45       0.9029   0.6722   0.8926   0.9995   0.9957   0.5073  
0.50       0.9029   0.6722   0.8926   0.9995   0.9957   0.5073  
0.55       0.9029   0.6722   0.8926   0.9995   0.9957   0.5073  
0.60       0.9029   0.6722   0.8926   0.9995   0.9957   0.5073  
0.65       0.9029   0.6722   0.8926   0.9995   0.9957   0.5073  
0.70       0.9029   0.6722   0.8926   0.9995   0.9957   0.5073  
0.75       0.9029   0.6722   0.8926   0.9995   0.9957   0.5073  
0.80       0.9029   0.6722   0.8926   0.9995   0.9957   0.5073  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9029, F1=0.6722, Normal Recall=0.8926, Normal Precision=0.9995, Attack Recall=0.9957, Attack Precision=0.5073

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
0.15       0.9134   0.8213   0.8929   0.9987   0.9953   0.6990   <--
0.20       0.9134   0.8213   0.8929   0.9987   0.9953   0.6990  
0.25       0.9134   0.8213   0.8929   0.9987   0.9953   0.6990  
0.30       0.9134   0.8213   0.8929   0.9987   0.9953   0.6990  
0.35       0.9134   0.8213   0.8929   0.9987   0.9953   0.6990  
0.40       0.9134   0.8213   0.8929   0.9987   0.9953   0.6990  
0.45       0.9134   0.8213   0.8929   0.9987   0.9953   0.6990  
0.50       0.9134   0.8213   0.8929   0.9987   0.9953   0.6990  
0.55       0.9134   0.8213   0.8929   0.9987   0.9953   0.6990  
0.60       0.9134   0.8213   0.8929   0.9987   0.9953   0.6990  
0.65       0.9134   0.8213   0.8929   0.9987   0.9953   0.6990  
0.70       0.9134   0.8213   0.8929   0.9987   0.9953   0.6990  
0.75       0.9134   0.8213   0.8929   0.9987   0.9953   0.6990  
0.80       0.9134   0.8213   0.8929   0.9987   0.9953   0.6990  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9134, F1=0.8213, Normal Recall=0.8929, Normal Precision=0.9987, Attack Recall=0.9953, Attack Precision=0.6990

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
0.15       0.9236   0.8865   0.8928   0.9977   0.9953   0.7992   <--
0.20       0.9236   0.8865   0.8928   0.9977   0.9953   0.7992  
0.25       0.9236   0.8865   0.8928   0.9977   0.9953   0.7992  
0.30       0.9236   0.8865   0.8928   0.9977   0.9953   0.7992  
0.35       0.9236   0.8865   0.8928   0.9977   0.9953   0.7992  
0.40       0.9236   0.8865   0.8928   0.9977   0.9953   0.7992  
0.45       0.9236   0.8865   0.8928   0.9977   0.9953   0.7992  
0.50       0.9236   0.8865   0.8928   0.9977   0.9953   0.7992  
0.55       0.9236   0.8865   0.8928   0.9977   0.9953   0.7992  
0.60       0.9236   0.8865   0.8928   0.9977   0.9953   0.7992  
0.65       0.9236   0.8865   0.8928   0.9977   0.9953   0.7992  
0.70       0.9236   0.8865   0.8928   0.9977   0.9953   0.7992  
0.75       0.9236   0.8865   0.8928   0.9977   0.9953   0.7992  
0.80       0.9236   0.8865   0.8928   0.9977   0.9953   0.7992  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9236, F1=0.8865, Normal Recall=0.8928, Normal Precision=0.9977, Attack Recall=0.9953, Attack Precision=0.7992

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
0.15       0.9338   0.9233   0.8929   0.9965   0.9953   0.8610   <--
0.20       0.9338   0.9233   0.8929   0.9965   0.9953   0.8610  
0.25       0.9338   0.9233   0.8929   0.9965   0.9953   0.8610  
0.30       0.9338   0.9233   0.8929   0.9965   0.9953   0.8610  
0.35       0.9338   0.9233   0.8929   0.9965   0.9953   0.8610  
0.40       0.9338   0.9233   0.8929   0.9965   0.9953   0.8610  
0.45       0.9338   0.9233   0.8929   0.9965   0.9953   0.8610  
0.50       0.9338   0.9233   0.8929   0.9965   0.9953   0.8610  
0.55       0.9338   0.9233   0.8929   0.9965   0.9953   0.8610  
0.60       0.9338   0.9233   0.8929   0.9965   0.9953   0.8610  
0.65       0.9338   0.9233   0.8929   0.9965   0.9953   0.8610  
0.70       0.9338   0.9233   0.8929   0.9965   0.9953   0.8610  
0.75       0.9338   0.9233   0.8929   0.9965   0.9953   0.8610  
0.80       0.9338   0.9233   0.8929   0.9965   0.9953   0.8610  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9338, F1=0.9233, Normal Recall=0.8929, Normal Precision=0.9965, Attack Recall=0.9953, Attack Precision=0.8610

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
0.15       0.9442   0.9469   0.8931   0.9947   0.9953   0.9030   <--
0.20       0.9442   0.9469   0.8931   0.9947   0.9953   0.9030  
0.25       0.9442   0.9469   0.8931   0.9947   0.9953   0.9030  
0.30       0.9442   0.9469   0.8931   0.9947   0.9953   0.9030  
0.35       0.9442   0.9469   0.8931   0.9947   0.9953   0.9030  
0.40       0.9442   0.9469   0.8931   0.9947   0.9953   0.9030  
0.45       0.9442   0.9469   0.8931   0.9947   0.9953   0.9030  
0.50       0.9442   0.9469   0.8931   0.9947   0.9953   0.9030  
0.55       0.9442   0.9469   0.8931   0.9947   0.9953   0.9030  
0.60       0.9442   0.9469   0.8931   0.9947   0.9953   0.9030  
0.65       0.9442   0.9469   0.8931   0.9947   0.9953   0.9030  
0.70       0.9442   0.9469   0.8931   0.9947   0.9953   0.9030  
0.75       0.9442   0.9469   0.8931   0.9947   0.9953   0.9030  
0.80       0.9442   0.9469   0.8931   0.9947   0.9953   0.9030  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9442, F1=0.9469, Normal Recall=0.8931, Normal Precision=0.9947, Attack Recall=0.9953, Attack Precision=0.9030

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
0.15       0.6407   0.3557   0.6018   0.9984   0.9915   0.2167   <--
0.20       0.6407   0.3557   0.6018   0.9984   0.9915   0.2167  
0.25       0.6407   0.3557   0.6018   0.9984   0.9915   0.2167  
0.30       0.6407   0.3557   0.6018   0.9984   0.9915   0.2167  
0.35       0.6407   0.3557   0.6018   0.9984   0.9915   0.2167  
0.40       0.6407   0.3557   0.6018   0.9984   0.9915   0.2167  
0.45       0.6407   0.3557   0.6018   0.9984   0.9915   0.2167  
0.50       0.6407   0.3557   0.6018   0.9984   0.9915   0.2167  
0.55       0.6407   0.3557   0.6018   0.9984   0.9915   0.2167  
0.60       0.6407   0.3557   0.6018   0.9984   0.9915   0.2167  
0.65       0.6407   0.3557   0.6018   0.9984   0.9915   0.2167  
0.70       0.6407   0.3557   0.6018   0.9984   0.9915   0.2167  
0.75       0.6407   0.3557   0.6018   0.9984   0.9915   0.2167  
0.80       0.6407   0.3557   0.6018   0.9984   0.9915   0.2167  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6407, F1=0.3557, Normal Recall=0.6018, Normal Precision=0.9984, Attack Recall=0.9915, Attack Precision=0.2167

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
0.15       0.6799   0.5532   0.6021   0.9962   0.9909   0.3837   <--
0.20       0.6799   0.5532   0.6021   0.9962   0.9909   0.3837  
0.25       0.6799   0.5532   0.6021   0.9962   0.9909   0.3837  
0.30       0.6799   0.5532   0.6021   0.9962   0.9909   0.3837  
0.35       0.6799   0.5532   0.6021   0.9962   0.9909   0.3837  
0.40       0.6799   0.5532   0.6021   0.9962   0.9909   0.3837  
0.45       0.6799   0.5532   0.6021   0.9962   0.9909   0.3837  
0.50       0.6799   0.5532   0.6021   0.9962   0.9909   0.3837  
0.55       0.6799   0.5532   0.6021   0.9962   0.9909   0.3837  
0.60       0.6799   0.5532   0.6021   0.9962   0.9909   0.3837  
0.65       0.6799   0.5532   0.6021   0.9962   0.9909   0.3837  
0.70       0.6799   0.5532   0.6021   0.9962   0.9909   0.3837  
0.75       0.6799   0.5532   0.6021   0.9962   0.9909   0.3837  
0.80       0.6799   0.5532   0.6021   0.9962   0.9909   0.3837  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6799, F1=0.5532, Normal Recall=0.6021, Normal Precision=0.9962, Attack Recall=0.9909, Attack Precision=0.3837

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
0.15       0.7182   0.6784   0.6014   0.9935   0.9909   0.5158   <--
0.20       0.7182   0.6784   0.6014   0.9935   0.9909   0.5158  
0.25       0.7182   0.6784   0.6014   0.9935   0.9909   0.5158  
0.30       0.7182   0.6784   0.6014   0.9935   0.9909   0.5158  
0.35       0.7182   0.6784   0.6014   0.9935   0.9909   0.5158  
0.40       0.7182   0.6784   0.6014   0.9935   0.9909   0.5158  
0.45       0.7182   0.6784   0.6014   0.9935   0.9909   0.5158  
0.50       0.7182   0.6784   0.6014   0.9935   0.9909   0.5158  
0.55       0.7182   0.6784   0.6014   0.9935   0.9909   0.5158  
0.60       0.7182   0.6784   0.6014   0.9935   0.9909   0.5158  
0.65       0.7182   0.6784   0.6014   0.9935   0.9909   0.5158  
0.70       0.7182   0.6784   0.6014   0.9935   0.9909   0.5158  
0.75       0.7182   0.6784   0.6014   0.9935   0.9909   0.5158  
0.80       0.7182   0.6784   0.6014   0.9935   0.9909   0.5158  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7182, F1=0.6784, Normal Recall=0.6014, Normal Precision=0.9935, Attack Recall=0.9909, Attack Precision=0.5158

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
0.15       0.7564   0.7649   0.6001   0.9899   0.9909   0.6229   <--
0.20       0.7564   0.7649   0.6001   0.9899   0.9909   0.6229  
0.25       0.7564   0.7649   0.6001   0.9899   0.9909   0.6229  
0.30       0.7564   0.7649   0.6001   0.9899   0.9909   0.6229  
0.35       0.7564   0.7649   0.6001   0.9899   0.9909   0.6229  
0.40       0.7564   0.7649   0.6001   0.9899   0.9909   0.6229  
0.45       0.7564   0.7649   0.6001   0.9899   0.9909   0.6229  
0.50       0.7564   0.7649   0.6001   0.9899   0.9909   0.6229  
0.55       0.7564   0.7649   0.6001   0.9899   0.9909   0.6229  
0.60       0.7564   0.7649   0.6001   0.9899   0.9909   0.6229  
0.65       0.7564   0.7649   0.6001   0.9899   0.9909   0.6229  
0.70       0.7564   0.7649   0.6001   0.9899   0.9909   0.6229  
0.75       0.7564   0.7649   0.6001   0.9899   0.9909   0.6229  
0.80       0.7564   0.7649   0.6001   0.9899   0.9909   0.6229  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7564, F1=0.7649, Normal Recall=0.6001, Normal Precision=0.9899, Attack Recall=0.9909, Attack Precision=0.6229

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
0.15       0.7945   0.8282   0.5981   0.9849   0.9909   0.7114   <--
0.20       0.7945   0.8282   0.5981   0.9849   0.9909   0.7114  
0.25       0.7945   0.8282   0.5981   0.9849   0.9909   0.7114  
0.30       0.7945   0.8282   0.5981   0.9849   0.9909   0.7114  
0.35       0.7945   0.8282   0.5981   0.9849   0.9909   0.7114  
0.40       0.7945   0.8282   0.5981   0.9849   0.9909   0.7114  
0.45       0.7945   0.8282   0.5981   0.9849   0.9909   0.7114  
0.50       0.7945   0.8282   0.5981   0.9849   0.9909   0.7114  
0.55       0.7945   0.8282   0.5981   0.9849   0.9909   0.7114  
0.60       0.7945   0.8282   0.5981   0.9849   0.9909   0.7114  
0.65       0.7945   0.8282   0.5981   0.9849   0.9909   0.7114  
0.70       0.7945   0.8282   0.5981   0.9849   0.9909   0.7114  
0.75       0.7945   0.8282   0.5981   0.9849   0.9909   0.7114  
0.80       0.7945   0.8282   0.5981   0.9849   0.9909   0.7114  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7945, F1=0.8282, Normal Recall=0.5981, Normal Precision=0.9849, Attack Recall=0.9909, Attack Precision=0.7114

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
0.15       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158   <--
0.20       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.25       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.30       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.35       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.40       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.45       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.50       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.55       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.60       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.65       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.70       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.75       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.80       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6384, F1=0.3545, Normal Recall=0.5990, Normal Precision=0.9987, Attack Recall=0.9930, Attack Precision=0.2158

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
0.15       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824   <--
0.20       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.25       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.30       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.35       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.40       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.45       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.50       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.55       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.60       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.65       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.70       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.75       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.80       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6780, F1=0.5521, Normal Recall=0.5993, Normal Precision=0.9969, Attack Recall=0.9925, Attack Precision=0.3824

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
0.15       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144   <--
0.20       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.25       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.30       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.35       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.40       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.45       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.50       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.55       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.60       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.65       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.70       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.75       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.80       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7167, F1=0.6776, Normal Recall=0.5984, Normal Precision=0.9947, Attack Recall=0.9925, Attack Precision=0.5144

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
0.15       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216   <--
0.20       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.25       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.30       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.35       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.40       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.45       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.50       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.55       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.60       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.65       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.70       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.75       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.80       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7553, F1=0.7644, Normal Recall=0.5972, Normal Precision=0.9917, Attack Recall=0.9925, Attack Precision=0.6216

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
0.15       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102   <--
0.20       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.25       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.30       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.35       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.40       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.45       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.50       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.55       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.60       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.65       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.70       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.75       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.80       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7937, F1=0.8279, Normal Recall=0.5949, Normal Precision=0.9876, Attack Recall=0.9925, Attack Precision=0.7102

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
0.15       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158   <--
0.20       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.25       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.30       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.35       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.40       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.45       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.50       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.55       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.60       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.65       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.70       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.75       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
0.80       0.6384   0.3545   0.5990   0.9987   0.9930   0.2158  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6384, F1=0.3545, Normal Recall=0.5990, Normal Precision=0.9987, Attack Recall=0.9930, Attack Precision=0.2158

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
0.15       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824   <--
0.20       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.25       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.30       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.35       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.40       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.45       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.50       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.55       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.60       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.65       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.70       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.75       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
0.80       0.6780   0.5521   0.5993   0.9969   0.9925   0.3824  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6780, F1=0.5521, Normal Recall=0.5993, Normal Precision=0.9969, Attack Recall=0.9925, Attack Precision=0.3824

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
0.15       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144   <--
0.20       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.25       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.30       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.35       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.40       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.45       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.50       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.55       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.60       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.65       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.70       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.75       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
0.80       0.7167   0.6776   0.5984   0.9947   0.9925   0.5144  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7167, F1=0.6776, Normal Recall=0.5984, Normal Precision=0.9947, Attack Recall=0.9925, Attack Precision=0.5144

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
0.15       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216   <--
0.20       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.25       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.30       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.35       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.40       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.45       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.50       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.55       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.60       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.65       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.70       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.75       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
0.80       0.7553   0.7644   0.5972   0.9917   0.9925   0.6216  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7553, F1=0.7644, Normal Recall=0.5972, Normal Precision=0.9917, Attack Recall=0.9925, Attack Precision=0.6216

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
0.15       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102   <--
0.20       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.25       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.30       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.35       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.40       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.45       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.50       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.55       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.60       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.65       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.70       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.75       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
0.80       0.7937   0.8279   0.5949   0.9876   0.9925   0.7102  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7937, F1=0.8279, Normal Recall=0.5949, Normal Precision=0.9876, Attack Recall=0.9925, Attack Precision=0.7102

```

