# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e1_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-22 08:31:28 |

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
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9062 | 0.9087 | 0.9099 | 0.9121 | 0.9132 | 0.9148 | 0.9167 | 0.9197 | 0.9207 | 0.9221 | 0.9244 |
| QAT+Prune only | 0.5430 | 0.5740 | 0.6049 | 0.6350 | 0.6669 | 0.6965 | 0.7267 | 0.7587 | 0.7885 | 0.8193 | 0.8500 |
| QAT+PTQ | 0.5442 | 0.5745 | 0.6049 | 0.6345 | 0.6659 | 0.6948 | 0.7244 | 0.7558 | 0.7854 | 0.8157 | 0.8458 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.5442 | 0.5745 | 0.6049 | 0.6345 | 0.6659 | 0.6948 | 0.7244 | 0.7558 | 0.7854 | 0.8157 | 0.8458 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6699 | 0.8041 | 0.8632 | 0.8949 | 0.9156 | 0.9302 | 0.9416 | 0.9491 | 0.9553 | 0.9607 |
| QAT+Prune only | 0.0000 | 0.2855 | 0.4625 | 0.5828 | 0.6712 | 0.7369 | 0.7887 | 0.8314 | 0.8654 | 0.8944 | 0.9189 |
| QAT+PTQ | 0.0000 | 0.2848 | 0.4613 | 0.5813 | 0.6695 | 0.7349 | 0.7865 | 0.8290 | 0.8631 | 0.8920 | 0.9164 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2848 | 0.4613 | 0.5813 | 0.6695 | 0.7349 | 0.7865 | 0.8290 | 0.8631 | 0.8920 | 0.9164 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9062 | 0.9067 | 0.9063 | 0.9069 | 0.9057 | 0.9052 | 0.9053 | 0.9089 | 0.9061 | 0.9019 | 0.0000 |
| QAT+Prune only | 0.5430 | 0.5432 | 0.5436 | 0.5429 | 0.5448 | 0.5431 | 0.5419 | 0.5458 | 0.5429 | 0.5437 | 0.0000 |
| QAT+PTQ | 0.5442 | 0.5442 | 0.5446 | 0.5440 | 0.5460 | 0.5439 | 0.5424 | 0.5460 | 0.5442 | 0.5449 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.5442 | 0.5442 | 0.5446 | 0.5440 | 0.5460 | 0.5439 | 0.5424 | 0.5460 | 0.5442 | 0.5449 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9062 | 0.0000 | 0.0000 | 0.0000 | 0.9062 | 1.0000 |
| 90 | 10 | 299,940 | 0.9087 | 0.5247 | 0.9265 | 0.6699 | 0.9067 | 0.9911 |
| 80 | 20 | 291,350 | 0.9099 | 0.7115 | 0.9244 | 0.8041 | 0.9063 | 0.9796 |
| 70 | 30 | 194,230 | 0.9121 | 0.8097 | 0.9244 | 0.8632 | 0.9069 | 0.9655 |
| 60 | 40 | 145,675 | 0.9132 | 0.8673 | 0.9244 | 0.8949 | 0.9057 | 0.9473 |
| 50 | 50 | 116,540 | 0.9148 | 0.9070 | 0.9244 | 0.9156 | 0.9052 | 0.9229 |
| 40 | 60 | 97,115 | 0.9167 | 0.9361 | 0.9244 | 0.9302 | 0.9053 | 0.8886 |
| 30 | 70 | 83,240 | 0.9197 | 0.9595 | 0.9244 | 0.9416 | 0.9089 | 0.8374 |
| 20 | 80 | 72,835 | 0.9207 | 0.9752 | 0.9244 | 0.9491 | 0.9061 | 0.7497 |
| 10 | 90 | 64,740 | 0.9221 | 0.9883 | 0.9244 | 0.9553 | 0.9019 | 0.5699 |
| 0 | 100 | 58,270 | 0.9244 | 1.0000 | 0.9244 | 0.9607 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5430 | 0.0000 | 0.0000 | 0.0000 | 0.5430 | 1.0000 |
| 90 | 10 | 299,940 | 0.5740 | 0.1715 | 0.8511 | 0.2855 | 0.5432 | 0.9705 |
| 80 | 20 | 291,350 | 0.6049 | 0.3177 | 0.8500 | 0.4625 | 0.5436 | 0.9355 |
| 70 | 30 | 194,230 | 0.6350 | 0.4435 | 0.8500 | 0.5828 | 0.5429 | 0.8941 |
| 60 | 40 | 145,675 | 0.6669 | 0.5545 | 0.8500 | 0.6712 | 0.5448 | 0.8449 |
| 50 | 50 | 116,540 | 0.6965 | 0.6504 | 0.8500 | 0.7369 | 0.5431 | 0.7835 |
| 40 | 60 | 97,115 | 0.7267 | 0.7357 | 0.8500 | 0.7887 | 0.5419 | 0.7066 |
| 30 | 70 | 83,240 | 0.7587 | 0.8136 | 0.8500 | 0.8314 | 0.5458 | 0.6092 |
| 20 | 80 | 72,835 | 0.7885 | 0.8815 | 0.8500 | 0.8654 | 0.5429 | 0.4750 |
| 10 | 90 | 64,740 | 0.8193 | 0.9437 | 0.8499 | 0.8944 | 0.5437 | 0.2870 |
| 0 | 100 | 58,270 | 0.8500 | 1.0000 | 0.8500 | 0.9189 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5442 | 0.0000 | 0.0000 | 0.0000 | 0.5442 | 1.0000 |
| 90 | 10 | 299,940 | 0.5745 | 0.1712 | 0.8471 | 0.2848 | 0.5442 | 0.9697 |
| 80 | 20 | 291,350 | 0.6049 | 0.3171 | 0.8458 | 0.4613 | 0.5446 | 0.9339 |
| 70 | 30 | 194,230 | 0.6345 | 0.4429 | 0.8458 | 0.5813 | 0.5440 | 0.8917 |
| 60 | 40 | 145,675 | 0.6659 | 0.5540 | 0.8458 | 0.6695 | 0.5460 | 0.8415 |
| 50 | 50 | 116,540 | 0.6948 | 0.6497 | 0.8458 | 0.7349 | 0.5439 | 0.7791 |
| 40 | 60 | 97,115 | 0.7244 | 0.7349 | 0.8458 | 0.7865 | 0.5424 | 0.7010 |
| 30 | 70 | 83,240 | 0.7558 | 0.8130 | 0.8458 | 0.8290 | 0.5460 | 0.6027 |
| 20 | 80 | 72,835 | 0.7854 | 0.8813 | 0.8458 | 0.8631 | 0.5442 | 0.4687 |
| 10 | 90 | 64,740 | 0.8157 | 0.9436 | 0.8458 | 0.8920 | 0.5449 | 0.2819 |
| 0 | 100 | 58,270 | 0.8458 | 1.0000 | 0.8458 | 0.9164 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.5442 | 0.0000 | 0.0000 | 0.0000 | 0.5442 | 1.0000 |
| 90 | 10 | 299,940 | 0.5745 | 0.1712 | 0.8471 | 0.2848 | 0.5442 | 0.9697 |
| 80 | 20 | 291,350 | 0.6049 | 0.3171 | 0.8458 | 0.4613 | 0.5446 | 0.9339 |
| 70 | 30 | 194,230 | 0.6345 | 0.4429 | 0.8458 | 0.5813 | 0.5440 | 0.8917 |
| 60 | 40 | 145,675 | 0.6659 | 0.5540 | 0.8458 | 0.6695 | 0.5460 | 0.8415 |
| 50 | 50 | 116,540 | 0.6948 | 0.6497 | 0.8458 | 0.7349 | 0.5439 | 0.7791 |
| 40 | 60 | 97,115 | 0.7244 | 0.7349 | 0.8458 | 0.7865 | 0.5424 | 0.7010 |
| 30 | 70 | 83,240 | 0.7558 | 0.8130 | 0.8458 | 0.8290 | 0.5460 | 0.6027 |
| 20 | 80 | 72,835 | 0.7854 | 0.8813 | 0.8458 | 0.8631 | 0.5442 | 0.4687 |
| 10 | 90 | 64,740 | 0.8157 | 0.9436 | 0.8458 | 0.8920 | 0.5449 | 0.2819 |
| 0 | 100 | 58,270 | 0.8458 | 1.0000 | 0.8458 | 0.9164 | 0.0000 | 0.0000 |


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
0.15       0.9085   0.6690   0.9067   0.9908   0.9245   0.5241   <--
0.20       0.9085   0.6690   0.9067   0.9908   0.9245   0.5241  
0.25       0.9085   0.6690   0.9067   0.9908   0.9245   0.5241  
0.30       0.9085   0.6690   0.9067   0.9908   0.9245   0.5241  
0.35       0.9085   0.6690   0.9067   0.9908   0.9245   0.5241  
0.40       0.9085   0.6690   0.9067   0.9908   0.9245   0.5241  
0.45       0.9085   0.6690   0.9067   0.9908   0.9245   0.5241  
0.50       0.9085   0.6690   0.9067   0.9908   0.9245   0.5241  
0.55       0.9085   0.6690   0.9067   0.9908   0.9245   0.5241  
0.60       0.9085   0.6690   0.9067   0.9908   0.9245   0.5241  
0.65       0.9085   0.6690   0.9067   0.9908   0.9245   0.5241  
0.70       0.9085   0.6690   0.9067   0.9908   0.9245   0.5241  
0.75       0.9085   0.6690   0.9067   0.9908   0.9245   0.5241  
0.80       0.9085   0.6690   0.9067   0.9908   0.9245   0.5241  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9085, F1=0.6690, Normal Recall=0.9067, Normal Precision=0.9908, Attack Recall=0.9245, Attack Precision=0.5241

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
0.15       0.9103   0.8047   0.9067   0.9796   0.9244   0.7125   <--
0.20       0.9103   0.8047   0.9067   0.9796   0.9244   0.7125  
0.25       0.9103   0.8047   0.9067   0.9796   0.9244   0.7125  
0.30       0.9103   0.8047   0.9067   0.9796   0.9244   0.7125  
0.35       0.9103   0.8047   0.9067   0.9796   0.9244   0.7125  
0.40       0.9103   0.8047   0.9067   0.9796   0.9244   0.7125  
0.45       0.9103   0.8047   0.9067   0.9796   0.9244   0.7125  
0.50       0.9103   0.8047   0.9067   0.9796   0.9244   0.7125  
0.55       0.9103   0.8047   0.9067   0.9796   0.9244   0.7125  
0.60       0.9103   0.8047   0.9067   0.9796   0.9244   0.7125  
0.65       0.9103   0.8047   0.9067   0.9796   0.9244   0.7125  
0.70       0.9103   0.8047   0.9067   0.9796   0.9244   0.7125  
0.75       0.9103   0.8047   0.9067   0.9796   0.9244   0.7125  
0.80       0.9103   0.8047   0.9067   0.9796   0.9244   0.7125  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9103, F1=0.8047, Normal Recall=0.9067, Normal Precision=0.9796, Attack Recall=0.9244, Attack Precision=0.7125

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
0.15       0.9121   0.8632   0.9069   0.9655   0.9244   0.8097   <--
0.20       0.9121   0.8632   0.9069   0.9655   0.9244   0.8097  
0.25       0.9121   0.8632   0.9069   0.9655   0.9244   0.8097  
0.30       0.9121   0.8632   0.9069   0.9655   0.9244   0.8097  
0.35       0.9121   0.8632   0.9069   0.9655   0.9244   0.8097  
0.40       0.9121   0.8632   0.9069   0.9655   0.9244   0.8097  
0.45       0.9121   0.8632   0.9069   0.9655   0.9244   0.8097  
0.50       0.9121   0.8632   0.9069   0.9655   0.9244   0.8097  
0.55       0.9121   0.8632   0.9069   0.9655   0.9244   0.8097  
0.60       0.9121   0.8632   0.9069   0.9655   0.9244   0.8097  
0.65       0.9121   0.8632   0.9069   0.9655   0.9244   0.8097  
0.70       0.9121   0.8632   0.9069   0.9655   0.9244   0.8097  
0.75       0.9121   0.8632   0.9069   0.9655   0.9244   0.8097  
0.80       0.9121   0.8632   0.9069   0.9655   0.9244   0.8097  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9121, F1=0.8632, Normal Recall=0.9069, Normal Precision=0.9655, Attack Recall=0.9244, Attack Precision=0.8097

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
0.15       0.9137   0.8955   0.9066   0.9473   0.9244   0.8684   <--
0.20       0.9137   0.8955   0.9066   0.9473   0.9244   0.8684  
0.25       0.9137   0.8955   0.9066   0.9473   0.9244   0.8684  
0.30       0.9137   0.8955   0.9066   0.9473   0.9244   0.8684  
0.35       0.9137   0.8955   0.9066   0.9473   0.9244   0.8684  
0.40       0.9137   0.8955   0.9066   0.9473   0.9244   0.8684  
0.45       0.9137   0.8955   0.9066   0.9473   0.9244   0.8684  
0.50       0.9137   0.8955   0.9066   0.9473   0.9244   0.8684  
0.55       0.9137   0.8955   0.9066   0.9473   0.9244   0.8684  
0.60       0.9137   0.8955   0.9066   0.9473   0.9244   0.8684  
0.65       0.9137   0.8955   0.9066   0.9473   0.9244   0.8684  
0.70       0.9137   0.8955   0.9066   0.9473   0.9244   0.8684  
0.75       0.9137   0.8955   0.9066   0.9473   0.9244   0.8684  
0.80       0.9137   0.8955   0.9066   0.9473   0.9244   0.8684  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9137, F1=0.8955, Normal Recall=0.9066, Normal Precision=0.9473, Attack Recall=0.9244, Attack Precision=0.8684

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
0.15       0.9153   0.9161   0.9063   0.9230   0.9244   0.9079   <--
0.20       0.9153   0.9161   0.9063   0.9230   0.9244   0.9079  
0.25       0.9153   0.9161   0.9063   0.9230   0.9244   0.9079  
0.30       0.9153   0.9161   0.9063   0.9230   0.9244   0.9079  
0.35       0.9153   0.9161   0.9063   0.9230   0.9244   0.9079  
0.40       0.9153   0.9161   0.9063   0.9230   0.9244   0.9079  
0.45       0.9153   0.9161   0.9063   0.9230   0.9244   0.9079  
0.50       0.9153   0.9161   0.9063   0.9230   0.9244   0.9079  
0.55       0.9153   0.9161   0.9063   0.9230   0.9244   0.9079  
0.60       0.9153   0.9161   0.9063   0.9230   0.9244   0.9079  
0.65       0.9153   0.9161   0.9063   0.9230   0.9244   0.9079  
0.70       0.9153   0.9161   0.9063   0.9230   0.9244   0.9079  
0.75       0.9153   0.9161   0.9063   0.9230   0.9244   0.9079  
0.80       0.9153   0.9161   0.9063   0.9230   0.9244   0.9079  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9153, F1=0.9161, Normal Recall=0.9063, Normal Precision=0.9230, Attack Recall=0.9244, Attack Precision=0.9079

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
0.15       0.5741   0.2857   0.5432   0.9706   0.8518   0.1716   <--
0.20       0.5741   0.2857   0.5432   0.9706   0.8518   0.1716  
0.25       0.5741   0.2857   0.5432   0.9706   0.8518   0.1716  
0.30       0.5741   0.2857   0.5432   0.9706   0.8518   0.1716  
0.35       0.5741   0.2857   0.5432   0.9706   0.8518   0.1716  
0.40       0.5741   0.2857   0.5432   0.9706   0.8518   0.1716  
0.45       0.5741   0.2857   0.5432   0.9706   0.8518   0.1716  
0.50       0.5741   0.2857   0.5432   0.9706   0.8518   0.1716  
0.55       0.5741   0.2857   0.5432   0.9706   0.8518   0.1716  
0.60       0.5741   0.2857   0.5432   0.9706   0.8518   0.1716  
0.65       0.5741   0.2857   0.5432   0.9706   0.8518   0.1716  
0.70       0.5741   0.2857   0.5432   0.9706   0.8518   0.1716  
0.75       0.5741   0.2857   0.5432   0.9706   0.8518   0.1716  
0.80       0.5741   0.2857   0.5432   0.9706   0.8518   0.1716  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5741, F1=0.2857, Normal Recall=0.5432, Normal Precision=0.9706, Attack Recall=0.8518, Attack Precision=0.1716

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
0.15       0.6043   0.4621   0.5429   0.9354   0.8500   0.3173   <--
0.20       0.6043   0.4621   0.5429   0.9354   0.8500   0.3173  
0.25       0.6043   0.4621   0.5429   0.9354   0.8500   0.3173  
0.30       0.6043   0.4621   0.5429   0.9354   0.8500   0.3173  
0.35       0.6043   0.4621   0.5429   0.9354   0.8500   0.3173  
0.40       0.6043   0.4621   0.5429   0.9354   0.8500   0.3173  
0.45       0.6043   0.4621   0.5429   0.9354   0.8500   0.3173  
0.50       0.6043   0.4621   0.5429   0.9354   0.8500   0.3173  
0.55       0.6043   0.4621   0.5429   0.9354   0.8500   0.3173  
0.60       0.6043   0.4621   0.5429   0.9354   0.8500   0.3173  
0.65       0.6043   0.4621   0.5429   0.9354   0.8500   0.3173  
0.70       0.6043   0.4621   0.5429   0.9354   0.8500   0.3173  
0.75       0.6043   0.4621   0.5429   0.9354   0.8500   0.3173  
0.80       0.6043   0.4621   0.5429   0.9354   0.8500   0.3173  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6043, F1=0.4621, Normal Recall=0.5429, Normal Precision=0.9354, Attack Recall=0.8500, Attack Precision=0.3173

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
0.15       0.6347   0.5827   0.5425   0.8940   0.8500   0.4433   <--
0.20       0.6347   0.5827   0.5425   0.8940   0.8500   0.4433  
0.25       0.6347   0.5827   0.5425   0.8940   0.8500   0.4433  
0.30       0.6347   0.5827   0.5425   0.8940   0.8500   0.4433  
0.35       0.6347   0.5827   0.5425   0.8940   0.8500   0.4433  
0.40       0.6347   0.5827   0.5425   0.8940   0.8500   0.4433  
0.45       0.6347   0.5827   0.5425   0.8940   0.8500   0.4433  
0.50       0.6347   0.5827   0.5425   0.8940   0.8500   0.4433  
0.55       0.6347   0.5827   0.5425   0.8940   0.8500   0.4433  
0.60       0.6347   0.5827   0.5425   0.8940   0.8500   0.4433  
0.65       0.6347   0.5827   0.5425   0.8940   0.8500   0.4433  
0.70       0.6347   0.5827   0.5425   0.8940   0.8500   0.4433  
0.75       0.6347   0.5827   0.5425   0.8940   0.8500   0.4433  
0.80       0.6347   0.5827   0.5425   0.8940   0.8500   0.4433  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6347, F1=0.5827, Normal Recall=0.5425, Normal Precision=0.8940, Attack Recall=0.8500, Attack Precision=0.4433

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
0.15       0.6656   0.6703   0.5427   0.8444   0.8500   0.5534   <--
0.20       0.6656   0.6703   0.5427   0.8444   0.8500   0.5534  
0.25       0.6656   0.6703   0.5427   0.8444   0.8500   0.5534  
0.30       0.6656   0.6703   0.5427   0.8444   0.8500   0.5534  
0.35       0.6656   0.6703   0.5427   0.8444   0.8500   0.5534  
0.40       0.6656   0.6703   0.5427   0.8444   0.8500   0.5534  
0.45       0.6656   0.6703   0.5427   0.8444   0.8500   0.5534  
0.50       0.6656   0.6703   0.5427   0.8444   0.8500   0.5534  
0.55       0.6656   0.6703   0.5427   0.8444   0.8500   0.5534  
0.60       0.6656   0.6703   0.5427   0.8444   0.8500   0.5534  
0.65       0.6656   0.6703   0.5427   0.8444   0.8500   0.5534  
0.70       0.6656   0.6703   0.5427   0.8444   0.8500   0.5534  
0.75       0.6656   0.6703   0.5427   0.8444   0.8500   0.5534  
0.80       0.6656   0.6703   0.5427   0.8444   0.8500   0.5534  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6656, F1=0.6703, Normal Recall=0.5427, Normal Precision=0.8444, Attack Recall=0.8500, Attack Precision=0.5534

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
0.15       0.6958   0.7365   0.5417   0.7831   0.8500   0.6497   <--
0.20       0.6958   0.7365   0.5417   0.7831   0.8500   0.6497  
0.25       0.6958   0.7365   0.5417   0.7831   0.8500   0.6497  
0.30       0.6958   0.7365   0.5417   0.7831   0.8500   0.6497  
0.35       0.6958   0.7365   0.5417   0.7831   0.8500   0.6497  
0.40       0.6958   0.7365   0.5417   0.7831   0.8500   0.6497  
0.45       0.6958   0.7365   0.5417   0.7831   0.8500   0.6497  
0.50       0.6958   0.7365   0.5417   0.7831   0.8500   0.6497  
0.55       0.6958   0.7365   0.5417   0.7831   0.8500   0.6497  
0.60       0.6958   0.7365   0.5417   0.7831   0.8500   0.6497  
0.65       0.6958   0.7365   0.5417   0.7831   0.8500   0.6497  
0.70       0.6958   0.7365   0.5417   0.7831   0.8500   0.6497  
0.75       0.6958   0.7365   0.5417   0.7831   0.8500   0.6497  
0.80       0.6958   0.7365   0.5417   0.7831   0.8500   0.6497  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6958, F1=0.7365, Normal Recall=0.5417, Normal Precision=0.7831, Attack Recall=0.8500, Attack Precision=0.6497

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
0.15       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712   <--
0.20       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.25       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.30       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.35       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.40       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.45       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.50       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.55       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.60       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.65       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.70       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.75       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.80       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5745, F1=0.2848, Normal Recall=0.5442, Normal Precision=0.9698, Attack Recall=0.8473, Attack Precision=0.1712

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
0.15       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168   <--
0.20       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.25       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.30       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.35       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.40       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.45       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.50       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.55       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.60       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.65       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.70       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.75       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.80       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6043, F1=0.4609, Normal Recall=0.5439, Normal Precision=0.9338, Attack Recall=0.8458, Attack Precision=0.3168

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
0.15       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426   <--
0.20       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.25       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.30       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.35       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.40       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.45       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.50       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.55       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.60       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.65       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.70       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.75       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.80       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6342, F1=0.5811, Normal Recall=0.5435, Normal Precision=0.8916, Attack Recall=0.8458, Attack Precision=0.4426

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
0.15       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528   <--
0.20       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.25       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.30       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.35       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.40       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.45       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.50       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.55       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.60       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.65       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.70       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.75       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.80       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6646, F1=0.6686, Normal Recall=0.5439, Normal Precision=0.8410, Attack Recall=0.8458, Attack Precision=0.5528

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
0.15       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491   <--
0.20       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.25       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.30       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.35       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.40       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.45       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.50       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.55       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.60       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.65       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.70       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.75       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.80       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6943, F1=0.7345, Normal Recall=0.5429, Normal Precision=0.7787, Attack Recall=0.8458, Attack Precision=0.6491

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
0.15       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712   <--
0.20       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.25       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.30       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.35       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.40       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.45       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.50       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.55       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.60       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.65       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.70       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.75       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
0.80       0.5745   0.2848   0.5442   0.9698   0.8473   0.1712  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5745, F1=0.2848, Normal Recall=0.5442, Normal Precision=0.9698, Attack Recall=0.8473, Attack Precision=0.1712

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
0.15       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168   <--
0.20       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.25       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.30       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.35       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.40       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.45       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.50       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.55       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.60       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.65       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.70       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.75       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
0.80       0.6043   0.4609   0.5439   0.9338   0.8458   0.3168  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6043, F1=0.4609, Normal Recall=0.5439, Normal Precision=0.9338, Attack Recall=0.8458, Attack Precision=0.3168

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
0.15       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426   <--
0.20       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.25       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.30       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.35       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.40       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.45       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.50       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.55       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.60       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.65       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.70       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.75       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
0.80       0.6342   0.5811   0.5435   0.8916   0.8458   0.4426  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6342, F1=0.5811, Normal Recall=0.5435, Normal Precision=0.8916, Attack Recall=0.8458, Attack Precision=0.4426

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
0.15       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528   <--
0.20       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.25       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.30       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.35       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.40       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.45       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.50       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.55       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.60       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.65       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.70       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.75       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
0.80       0.6646   0.6686   0.5439   0.8410   0.8458   0.5528  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6646, F1=0.6686, Normal Recall=0.5439, Normal Precision=0.8410, Attack Recall=0.8458, Attack Precision=0.5528

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
0.15       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491   <--
0.20       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.25       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.30       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.35       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.40       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.45       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.50       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.55       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.60       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.65       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.70       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.75       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
0.80       0.6943   0.7345   0.5429   0.7787   0.8458   0.6491  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6943, F1=0.7345, Normal Recall=0.5429, Normal Precision=0.7787, Attack Recall=0.8458, Attack Precision=0.6491

```

