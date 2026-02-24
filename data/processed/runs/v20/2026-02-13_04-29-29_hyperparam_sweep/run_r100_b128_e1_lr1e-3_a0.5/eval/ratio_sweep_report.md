# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e1_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-21 05:48:23 |

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
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6199 | 0.6562 | 0.6934 | 0.7314 | 0.7684 | 0.8057 | 0.8430 | 0.8812 | 0.9175 | 0.9554 | 0.9933 |
| QAT+Prune only | 0.8295 | 0.8461 | 0.8616 | 0.8779 | 0.8938 | 0.9078 | 0.9251 | 0.9408 | 0.9560 | 0.9719 | 0.9881 |
| QAT+PTQ | 0.8278 | 0.8448 | 0.8604 | 0.8768 | 0.8930 | 0.9071 | 0.9247 | 0.9406 | 0.9558 | 0.9719 | 0.9883 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8278 | 0.8448 | 0.8604 | 0.8768 | 0.8930 | 0.9071 | 0.9247 | 0.9406 | 0.9558 | 0.9719 | 0.9883 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3663 | 0.5645 | 0.6893 | 0.7743 | 0.8364 | 0.8836 | 0.9213 | 0.9507 | 0.9756 | 0.9967 |
| QAT+Prune only | 0.0000 | 0.5622 | 0.7406 | 0.8292 | 0.8816 | 0.9146 | 0.9406 | 0.9590 | 0.9729 | 0.9844 | 0.9940 |
| QAT+PTQ | 0.0000 | 0.5601 | 0.7390 | 0.8279 | 0.8808 | 0.9141 | 0.9403 | 0.9588 | 0.9728 | 0.9844 | 0.9941 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5601 | 0.7390 | 0.8279 | 0.8808 | 0.9141 | 0.9403 | 0.9588 | 0.9728 | 0.9844 | 0.9941 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6199 | 0.6186 | 0.6184 | 0.6191 | 0.6184 | 0.6181 | 0.6174 | 0.6196 | 0.6142 | 0.6135 | 0.0000 |
| QAT+Prune only | 0.8295 | 0.8304 | 0.8299 | 0.8306 | 0.8310 | 0.8274 | 0.8305 | 0.8305 | 0.8273 | 0.8253 | 0.0000 |
| QAT+PTQ | 0.8278 | 0.8288 | 0.8284 | 0.8290 | 0.8295 | 0.8259 | 0.8294 | 0.8293 | 0.8257 | 0.8239 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8278 | 0.8288 | 0.8284 | 0.8290 | 0.8295 | 0.8259 | 0.8294 | 0.8293 | 0.8257 | 0.8239 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6199 | 0.0000 | 0.0000 | 0.0000 | 0.6199 | 1.0000 |
| 90 | 10 | 299,940 | 0.6562 | 0.2245 | 0.9938 | 0.3663 | 0.6186 | 0.9989 |
| 80 | 20 | 291,350 | 0.6934 | 0.3942 | 0.9933 | 0.5645 | 0.6184 | 0.9973 |
| 70 | 30 | 194,230 | 0.7314 | 0.5278 | 0.9933 | 0.6893 | 0.6191 | 0.9954 |
| 60 | 40 | 145,675 | 0.7684 | 0.6344 | 0.9933 | 0.7743 | 0.6184 | 0.9929 |
| 50 | 50 | 116,540 | 0.8057 | 0.7223 | 0.9933 | 0.8364 | 0.6181 | 0.9893 |
| 40 | 60 | 97,115 | 0.8430 | 0.7957 | 0.9933 | 0.8836 | 0.6174 | 0.9841 |
| 30 | 70 | 83,240 | 0.8812 | 0.8590 | 0.9933 | 0.9213 | 0.6196 | 0.9755 |
| 20 | 80 | 72,835 | 0.9175 | 0.9115 | 0.9933 | 0.9507 | 0.6142 | 0.9584 |
| 10 | 90 | 64,740 | 0.9554 | 0.9586 | 0.9933 | 0.9756 | 0.6135 | 0.9110 |
| 0 | 100 | 58,270 | 0.9933 | 1.0000 | 0.9933 | 0.9967 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8295 | 0.0000 | 0.0000 | 0.0000 | 0.8295 | 1.0000 |
| 90 | 10 | 299,940 | 0.8461 | 0.3929 | 0.9880 | 0.5622 | 0.8304 | 0.9984 |
| 80 | 20 | 291,350 | 0.8616 | 0.5922 | 0.9881 | 0.7406 | 0.8299 | 0.9964 |
| 70 | 30 | 194,230 | 0.8779 | 0.7143 | 0.9881 | 0.8292 | 0.8306 | 0.9939 |
| 60 | 40 | 145,675 | 0.8938 | 0.7958 | 0.9881 | 0.8816 | 0.8310 | 0.9906 |
| 50 | 50 | 116,540 | 0.9078 | 0.8513 | 0.9881 | 0.9146 | 0.8274 | 0.9859 |
| 40 | 60 | 97,115 | 0.9251 | 0.8974 | 0.9881 | 0.9406 | 0.8305 | 0.9790 |
| 30 | 70 | 83,240 | 0.9408 | 0.9315 | 0.9881 | 0.9590 | 0.8305 | 0.9678 |
| 20 | 80 | 72,835 | 0.9560 | 0.9581 | 0.9881 | 0.9729 | 0.8273 | 0.9458 |
| 10 | 90 | 64,740 | 0.9719 | 0.9807 | 0.9881 | 0.9844 | 0.8253 | 0.8855 |
| 0 | 100 | 58,270 | 0.9881 | 1.0000 | 0.9881 | 0.9940 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8278 | 0.0000 | 0.0000 | 0.0000 | 0.8278 | 1.0000 |
| 90 | 10 | 299,940 | 0.8448 | 0.3908 | 0.9881 | 0.5601 | 0.8288 | 0.9984 |
| 80 | 20 | 291,350 | 0.8604 | 0.5902 | 0.9883 | 0.7390 | 0.8284 | 0.9965 |
| 70 | 30 | 194,230 | 0.8768 | 0.7123 | 0.9883 | 0.8279 | 0.8290 | 0.9940 |
| 60 | 40 | 145,675 | 0.8930 | 0.7944 | 0.9883 | 0.8808 | 0.8295 | 0.9907 |
| 50 | 50 | 116,540 | 0.9071 | 0.8502 | 0.9883 | 0.9141 | 0.8259 | 0.9861 |
| 40 | 60 | 97,115 | 0.9247 | 0.8968 | 0.9883 | 0.9403 | 0.8294 | 0.9793 |
| 30 | 70 | 83,240 | 0.9406 | 0.9311 | 0.9883 | 0.9588 | 0.8293 | 0.9682 |
| 20 | 80 | 72,835 | 0.9558 | 0.9578 | 0.9883 | 0.9728 | 0.8257 | 0.9465 |
| 10 | 90 | 64,740 | 0.9719 | 0.9806 | 0.9883 | 0.9844 | 0.8239 | 0.8869 |
| 0 | 100 | 58,270 | 0.9883 | 1.0000 | 0.9883 | 0.9941 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8278 | 0.0000 | 0.0000 | 0.0000 | 0.8278 | 1.0000 |
| 90 | 10 | 299,940 | 0.8448 | 0.3908 | 0.9881 | 0.5601 | 0.8288 | 0.9984 |
| 80 | 20 | 291,350 | 0.8604 | 0.5902 | 0.9883 | 0.7390 | 0.8284 | 0.9965 |
| 70 | 30 | 194,230 | 0.8768 | 0.7123 | 0.9883 | 0.8279 | 0.8290 | 0.9940 |
| 60 | 40 | 145,675 | 0.8930 | 0.7944 | 0.9883 | 0.8808 | 0.8295 | 0.9907 |
| 50 | 50 | 116,540 | 0.9071 | 0.8502 | 0.9883 | 0.9141 | 0.8259 | 0.9861 |
| 40 | 60 | 97,115 | 0.9247 | 0.8968 | 0.9883 | 0.9403 | 0.8294 | 0.9793 |
| 30 | 70 | 83,240 | 0.9406 | 0.9311 | 0.9883 | 0.9588 | 0.8293 | 0.9682 |
| 20 | 80 | 72,835 | 0.9558 | 0.9578 | 0.9883 | 0.9728 | 0.8257 | 0.9465 |
| 10 | 90 | 64,740 | 0.9719 | 0.9806 | 0.9883 | 0.9844 | 0.8239 | 0.8869 |
| 0 | 100 | 58,270 | 0.9883 | 1.0000 | 0.9883 | 0.9941 | 0.0000 | 0.0000 |


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
0.15       0.6561   0.3661   0.6186   0.9988   0.9933   0.2244   <--
0.20       0.6561   0.3661   0.6186   0.9988   0.9933   0.2244  
0.25       0.6561   0.3661   0.6186   0.9988   0.9933   0.2244  
0.30       0.6561   0.3661   0.6186   0.9988   0.9933   0.2244  
0.35       0.6561   0.3661   0.6186   0.9988   0.9933   0.2244  
0.40       0.6561   0.3661   0.6186   0.9988   0.9933   0.2244  
0.45       0.6561   0.3661   0.6186   0.9988   0.9933   0.2244  
0.50       0.6561   0.3661   0.6186   0.9988   0.9933   0.2244  
0.55       0.6561   0.3661   0.6186   0.9988   0.9933   0.2244  
0.60       0.6561   0.3661   0.6186   0.9988   0.9933   0.2244  
0.65       0.6561   0.3661   0.6186   0.9988   0.9933   0.2244  
0.70       0.6561   0.3661   0.6186   0.9988   0.9933   0.2244  
0.75       0.6561   0.3661   0.6186   0.9988   0.9933   0.2244  
0.80       0.6561   0.3661   0.6186   0.9988   0.9933   0.2244  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6561, F1=0.3661, Normal Recall=0.6186, Normal Precision=0.9988, Attack Recall=0.9933, Attack Precision=0.2244

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
0.15       0.6936   0.5646   0.6187   0.9973   0.9933   0.3944   <--
0.20       0.6936   0.5646   0.6187   0.9973   0.9933   0.3944  
0.25       0.6936   0.5646   0.6187   0.9973   0.9933   0.3944  
0.30       0.6936   0.5646   0.6187   0.9973   0.9933   0.3944  
0.35       0.6936   0.5646   0.6187   0.9973   0.9933   0.3944  
0.40       0.6936   0.5646   0.6187   0.9973   0.9933   0.3944  
0.45       0.6936   0.5646   0.6187   0.9973   0.9933   0.3944  
0.50       0.6936   0.5646   0.6187   0.9973   0.9933   0.3944  
0.55       0.6936   0.5646   0.6187   0.9973   0.9933   0.3944  
0.60       0.6936   0.5646   0.6187   0.9973   0.9933   0.3944  
0.65       0.6936   0.5646   0.6187   0.9973   0.9933   0.3944  
0.70       0.6936   0.5646   0.6187   0.9973   0.9933   0.3944  
0.75       0.6936   0.5646   0.6187   0.9973   0.9933   0.3944  
0.80       0.6936   0.5646   0.6187   0.9973   0.9933   0.3944  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6936, F1=0.5646, Normal Recall=0.6187, Normal Precision=0.9973, Attack Recall=0.9933, Attack Precision=0.3944

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
0.15       0.7320   0.6899   0.6201   0.9954   0.9933   0.5284   <--
0.20       0.7320   0.6899   0.6201   0.9954   0.9933   0.5284  
0.25       0.7320   0.6899   0.6201   0.9954   0.9933   0.5284  
0.30       0.7320   0.6899   0.6201   0.9954   0.9933   0.5284  
0.35       0.7320   0.6899   0.6201   0.9954   0.9933   0.5284  
0.40       0.7320   0.6899   0.6201   0.9954   0.9933   0.5284  
0.45       0.7320   0.6899   0.6201   0.9954   0.9933   0.5284  
0.50       0.7320   0.6899   0.6201   0.9954   0.9933   0.5284  
0.55       0.7320   0.6899   0.6201   0.9954   0.9933   0.5284  
0.60       0.7320   0.6899   0.6201   0.9954   0.9933   0.5284  
0.65       0.7320   0.6899   0.6201   0.9954   0.9933   0.5284  
0.70       0.7320   0.6899   0.6201   0.9954   0.9933   0.5284  
0.75       0.7320   0.6899   0.6201   0.9954   0.9933   0.5284  
0.80       0.7320   0.6899   0.6201   0.9954   0.9933   0.5284  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7320, F1=0.6899, Normal Recall=0.6201, Normal Precision=0.9954, Attack Recall=0.9933, Attack Precision=0.5284

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
0.15       0.7700   0.7756   0.6212   0.9929   0.9933   0.6361   <--
0.20       0.7700   0.7756   0.6212   0.9929   0.9933   0.6361  
0.25       0.7700   0.7756   0.6212   0.9929   0.9933   0.6361  
0.30       0.7700   0.7756   0.6212   0.9929   0.9933   0.6361  
0.35       0.7700   0.7756   0.6212   0.9929   0.9933   0.6361  
0.40       0.7700   0.7756   0.6212   0.9929   0.9933   0.6361  
0.45       0.7700   0.7756   0.6212   0.9929   0.9933   0.6361  
0.50       0.7700   0.7756   0.6212   0.9929   0.9933   0.6361  
0.55       0.7700   0.7756   0.6212   0.9929   0.9933   0.6361  
0.60       0.7700   0.7756   0.6212   0.9929   0.9933   0.6361  
0.65       0.7700   0.7756   0.6212   0.9929   0.9933   0.6361  
0.70       0.7700   0.7756   0.6212   0.9929   0.9933   0.6361  
0.75       0.7700   0.7756   0.6212   0.9929   0.9933   0.6361  
0.80       0.7700   0.7756   0.6212   0.9929   0.9933   0.6361  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7700, F1=0.7756, Normal Recall=0.6212, Normal Precision=0.9929, Attack Recall=0.9933, Attack Precision=0.6361

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
0.15       0.8074   0.8376   0.6214   0.9894   0.9933   0.7240   <--
0.20       0.8074   0.8376   0.6214   0.9894   0.9933   0.7240  
0.25       0.8074   0.8376   0.6214   0.9894   0.9933   0.7240  
0.30       0.8074   0.8376   0.6214   0.9894   0.9933   0.7240  
0.35       0.8074   0.8376   0.6214   0.9894   0.9933   0.7240  
0.40       0.8074   0.8376   0.6214   0.9894   0.9933   0.7240  
0.45       0.8074   0.8376   0.6214   0.9894   0.9933   0.7240  
0.50       0.8074   0.8376   0.6214   0.9894   0.9933   0.7240  
0.55       0.8074   0.8376   0.6214   0.9894   0.9933   0.7240  
0.60       0.8074   0.8376   0.6214   0.9894   0.9933   0.7240  
0.65       0.8074   0.8376   0.6214   0.9894   0.9933   0.7240  
0.70       0.8074   0.8376   0.6214   0.9894   0.9933   0.7240  
0.75       0.8074   0.8376   0.6214   0.9894   0.9933   0.7240  
0.80       0.8074   0.8376   0.6214   0.9894   0.9933   0.7240  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8074, F1=0.8376, Normal Recall=0.6214, Normal Precision=0.9894, Attack Recall=0.9933, Attack Precision=0.7240

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
0.15       0.8462   0.5624   0.8304   0.9985   0.9885   0.3930   <--
0.20       0.8462   0.5624   0.8304   0.9985   0.9885   0.3930  
0.25       0.8462   0.5624   0.8304   0.9985   0.9885   0.3930  
0.30       0.8462   0.5624   0.8304   0.9985   0.9885   0.3930  
0.35       0.8462   0.5624   0.8304   0.9985   0.9885   0.3930  
0.40       0.8462   0.5624   0.8304   0.9985   0.9885   0.3930  
0.45       0.8462   0.5624   0.8304   0.9985   0.9885   0.3930  
0.50       0.8462   0.5624   0.8304   0.9985   0.9885   0.3930  
0.55       0.8462   0.5624   0.8304   0.9985   0.9885   0.3930  
0.60       0.8462   0.5624   0.8304   0.9985   0.9885   0.3930  
0.65       0.8462   0.5624   0.8304   0.9985   0.9885   0.3930  
0.70       0.8462   0.5624   0.8304   0.9985   0.9885   0.3930  
0.75       0.8462   0.5624   0.8304   0.9985   0.9885   0.3930  
0.80       0.8462   0.5624   0.8304   0.9985   0.9885   0.3930  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8462, F1=0.5624, Normal Recall=0.8304, Normal Precision=0.9985, Attack Recall=0.9885, Attack Precision=0.3930

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
0.15       0.8622   0.7415   0.8307   0.9964   0.9881   0.5934   <--
0.20       0.8622   0.7415   0.8307   0.9964   0.9881   0.5934  
0.25       0.8622   0.7415   0.8307   0.9964   0.9881   0.5934  
0.30       0.8622   0.7415   0.8307   0.9964   0.9881   0.5934  
0.35       0.8622   0.7415   0.8307   0.9964   0.9881   0.5934  
0.40       0.8622   0.7415   0.8307   0.9964   0.9881   0.5934  
0.45       0.8622   0.7415   0.8307   0.9964   0.9881   0.5934  
0.50       0.8622   0.7415   0.8307   0.9964   0.9881   0.5934  
0.55       0.8622   0.7415   0.8307   0.9964   0.9881   0.5934  
0.60       0.8622   0.7415   0.8307   0.9964   0.9881   0.5934  
0.65       0.8622   0.7415   0.8307   0.9964   0.9881   0.5934  
0.70       0.8622   0.7415   0.8307   0.9964   0.9881   0.5934  
0.75       0.8622   0.7415   0.8307   0.9964   0.9881   0.5934  
0.80       0.8622   0.7415   0.8307   0.9964   0.9881   0.5934  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8622, F1=0.7415, Normal Recall=0.8307, Normal Precision=0.9964, Attack Recall=0.9881, Attack Precision=0.5934

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
0.15       0.8773   0.8285   0.8297   0.9939   0.9881   0.7132   <--
0.20       0.8773   0.8285   0.8297   0.9939   0.9881   0.7132  
0.25       0.8773   0.8285   0.8297   0.9939   0.9881   0.7132  
0.30       0.8773   0.8285   0.8297   0.9939   0.9881   0.7132  
0.35       0.8773   0.8285   0.8297   0.9939   0.9881   0.7132  
0.40       0.8773   0.8285   0.8297   0.9939   0.9881   0.7132  
0.45       0.8773   0.8285   0.8297   0.9939   0.9881   0.7132  
0.50       0.8773   0.8285   0.8297   0.9939   0.9881   0.7132  
0.55       0.8773   0.8285   0.8297   0.9939   0.9881   0.7132  
0.60       0.8773   0.8285   0.8297   0.9939   0.9881   0.7132  
0.65       0.8773   0.8285   0.8297   0.9939   0.9881   0.7132  
0.70       0.8773   0.8285   0.8297   0.9939   0.9881   0.7132  
0.75       0.8773   0.8285   0.8297   0.9939   0.9881   0.7132  
0.80       0.8773   0.8285   0.8297   0.9939   0.9881   0.7132  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8773, F1=0.8285, Normal Recall=0.8297, Normal Precision=0.9939, Attack Recall=0.9881, Attack Precision=0.7132

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
0.15       0.8928   0.8806   0.8293   0.9906   0.9881   0.7942   <--
0.20       0.8928   0.8806   0.8293   0.9906   0.9881   0.7942  
0.25       0.8928   0.8806   0.8293   0.9906   0.9881   0.7942  
0.30       0.8928   0.8806   0.8293   0.9906   0.9881   0.7942  
0.35       0.8928   0.8806   0.8293   0.9906   0.9881   0.7942  
0.40       0.8928   0.8806   0.8293   0.9906   0.9881   0.7942  
0.45       0.8928   0.8806   0.8293   0.9906   0.9881   0.7942  
0.50       0.8928   0.8806   0.8293   0.9906   0.9881   0.7942  
0.55       0.8928   0.8806   0.8293   0.9906   0.9881   0.7942  
0.60       0.8928   0.8806   0.8293   0.9906   0.9881   0.7942  
0.65       0.8928   0.8806   0.8293   0.9906   0.9881   0.7942  
0.70       0.8928   0.8806   0.8293   0.9906   0.9881   0.7942  
0.75       0.8928   0.8806   0.8293   0.9906   0.9881   0.7942  
0.80       0.8928   0.8806   0.8293   0.9906   0.9881   0.7942  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8928, F1=0.8806, Normal Recall=0.8293, Normal Precision=0.9906, Attack Recall=0.9881, Attack Precision=0.7942

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
0.15       0.9080   0.9149   0.8280   0.9859   0.9881   0.8517   <--
0.20       0.9080   0.9149   0.8280   0.9859   0.9881   0.8517  
0.25       0.9080   0.9149   0.8280   0.9859   0.9881   0.8517  
0.30       0.9080   0.9149   0.8280   0.9859   0.9881   0.8517  
0.35       0.9080   0.9149   0.8280   0.9859   0.9881   0.8517  
0.40       0.9080   0.9149   0.8280   0.9859   0.9881   0.8517  
0.45       0.9080   0.9149   0.8280   0.9859   0.9881   0.8517  
0.50       0.9080   0.9149   0.8280   0.9859   0.9881   0.8517  
0.55       0.9080   0.9149   0.8280   0.9859   0.9881   0.8517  
0.60       0.9080   0.9149   0.8280   0.9859   0.9881   0.8517  
0.65       0.9080   0.9149   0.8280   0.9859   0.9881   0.8517  
0.70       0.9080   0.9149   0.8280   0.9859   0.9881   0.8517  
0.75       0.9080   0.9149   0.8280   0.9859   0.9881   0.8517  
0.80       0.9080   0.9149   0.8280   0.9859   0.9881   0.8517  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9080, F1=0.9149, Normal Recall=0.8280, Normal Precision=0.9859, Attack Recall=0.9881, Attack Precision=0.8517

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
0.15       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909   <--
0.20       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.25       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.30       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.35       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.40       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.45       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.50       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.55       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.60       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.65       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.70       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.75       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.80       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8448, F1=0.5603, Normal Recall=0.8289, Normal Precision=0.9985, Attack Recall=0.9886, Attack Precision=0.3909

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
0.15       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913   <--
0.20       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.25       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.30       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.35       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.40       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.45       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.50       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.55       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.60       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.65       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.70       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.75       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.80       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8611, F1=0.7399, Normal Recall=0.8292, Normal Precision=0.9965, Attack Recall=0.9883, Attack Precision=0.5913

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
0.15       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113   <--
0.20       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.25       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.30       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.35       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.40       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.45       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.50       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.55       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.60       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.65       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.70       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.75       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.80       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8761, F1=0.8272, Normal Recall=0.8281, Normal Precision=0.9940, Attack Recall=0.9883, Attack Precision=0.7113

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
0.15       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927   <--
0.20       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.25       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.30       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.35       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.40       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.45       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.50       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.55       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.60       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.65       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.70       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.75       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.80       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8919, F1=0.8798, Normal Recall=0.8277, Normal Precision=0.9907, Attack Recall=0.9883, Attack Precision=0.7927

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
0.15       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506   <--
0.20       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.25       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.30       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.35       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.40       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.45       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.50       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.55       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.60       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.65       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.70       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.75       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.80       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9073, F1=0.9143, Normal Recall=0.8264, Normal Precision=0.9861, Attack Recall=0.9883, Attack Precision=0.8506

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
0.15       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909   <--
0.20       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.25       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.30       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.35       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.40       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.45       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.50       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.55       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.60       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.65       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.70       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.75       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
0.80       0.8448   0.5603   0.8289   0.9985   0.9886   0.3909  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8448, F1=0.5603, Normal Recall=0.8289, Normal Precision=0.9985, Attack Recall=0.9886, Attack Precision=0.3909

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
0.15       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913   <--
0.20       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.25       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.30       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.35       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.40       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.45       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.50       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.55       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.60       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.65       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.70       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.75       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
0.80       0.8611   0.7399   0.8292   0.9965   0.9883   0.5913  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8611, F1=0.7399, Normal Recall=0.8292, Normal Precision=0.9965, Attack Recall=0.9883, Attack Precision=0.5913

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
0.15       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113   <--
0.20       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.25       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.30       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.35       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.40       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.45       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.50       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.55       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.60       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.65       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.70       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.75       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
0.80       0.8761   0.8272   0.8281   0.9940   0.9883   0.7113  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8761, F1=0.8272, Normal Recall=0.8281, Normal Precision=0.9940, Attack Recall=0.9883, Attack Precision=0.7113

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
0.15       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927   <--
0.20       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.25       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.30       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.35       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.40       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.45       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.50       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.55       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.60       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.65       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.70       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.75       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
0.80       0.8919   0.8798   0.8277   0.9907   0.9883   0.7927  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8919, F1=0.8798, Normal Recall=0.8277, Normal Precision=0.9907, Attack Recall=0.9883, Attack Precision=0.7927

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
0.15       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506   <--
0.20       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.25       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.30       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.35       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.40       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.45       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.50       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.55       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.60       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.65       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.70       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.75       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
0.80       0.9073   0.9143   0.8264   0.9861   0.9883   0.8506  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9073, F1=0.9143, Normal Recall=0.8264, Normal Precision=0.9861, Attack Recall=0.9883, Attack Precision=0.8506

```

