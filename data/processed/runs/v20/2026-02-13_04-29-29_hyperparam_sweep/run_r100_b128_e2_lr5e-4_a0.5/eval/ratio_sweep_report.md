# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e2_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-21 15:51:18 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 2 |
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3397 | 0.4029 | 0.4670 | 0.5312 | 0.5957 | 0.6593 | 0.7244 | 0.7891 | 0.8523 | 0.9164 | 0.9811 |
| QAT+Prune only | 0.5933 | 0.6335 | 0.6739 | 0.7151 | 0.7547 | 0.7956 | 0.8367 | 0.8782 | 0.9191 | 0.9585 | 0.9998 |
| QAT+PTQ | 0.5933 | 0.6335 | 0.6738 | 0.7150 | 0.7547 | 0.7955 | 0.8366 | 0.8782 | 0.9191 | 0.9585 | 0.9998 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.5933 | 0.6335 | 0.6738 | 0.7150 | 0.7547 | 0.7955 | 0.8366 | 0.8782 | 0.9191 | 0.9585 | 0.9998 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2474 | 0.4241 | 0.5567 | 0.6600 | 0.7423 | 0.8103 | 0.8669 | 0.9140 | 0.9548 | 0.9905 |
| QAT+Prune only | 0.0000 | 0.3530 | 0.5508 | 0.6780 | 0.7653 | 0.8303 | 0.8802 | 0.9200 | 0.9519 | 0.9775 | 0.9999 |
| QAT+PTQ | 0.0000 | 0.3530 | 0.5508 | 0.6780 | 0.7653 | 0.8302 | 0.8801 | 0.9200 | 0.9518 | 0.9775 | 0.9999 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3530 | 0.5508 | 0.6780 | 0.7653 | 0.8302 | 0.8801 | 0.9200 | 0.9518 | 0.9775 | 0.9999 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3397 | 0.3386 | 0.3385 | 0.3384 | 0.3388 | 0.3375 | 0.3394 | 0.3409 | 0.3369 | 0.3335 | 0.0000 |
| QAT+Prune only | 0.5933 | 0.5928 | 0.5924 | 0.5930 | 0.5913 | 0.5915 | 0.5921 | 0.5945 | 0.5963 | 0.5871 | 0.0000 |
| QAT+PTQ | 0.5933 | 0.5928 | 0.5923 | 0.5930 | 0.5913 | 0.5912 | 0.5918 | 0.5946 | 0.5962 | 0.5870 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.5933 | 0.5928 | 0.5923 | 0.5930 | 0.5913 | 0.5912 | 0.5918 | 0.5946 | 0.5962 | 0.5870 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3397 | 0.0000 | 0.0000 | 0.0000 | 0.3397 | 1.0000 |
| 90 | 10 | 299,940 | 0.4029 | 0.1415 | 0.9815 | 0.2474 | 0.3386 | 0.9940 |
| 80 | 20 | 291,350 | 0.4670 | 0.2705 | 0.9811 | 0.4241 | 0.3385 | 0.9862 |
| 70 | 30 | 194,230 | 0.5312 | 0.3886 | 0.9811 | 0.5567 | 0.3384 | 0.9766 |
| 60 | 40 | 145,675 | 0.5957 | 0.4973 | 0.9811 | 0.6600 | 0.3388 | 0.9642 |
| 50 | 50 | 116,540 | 0.6593 | 0.5969 | 0.9811 | 0.7423 | 0.3375 | 0.9470 |
| 40 | 60 | 97,115 | 0.7244 | 0.6902 | 0.9811 | 0.8103 | 0.3394 | 0.9230 |
| 30 | 70 | 83,240 | 0.7891 | 0.7765 | 0.9811 | 0.8669 | 0.3409 | 0.8856 |
| 20 | 80 | 72,835 | 0.8523 | 0.8555 | 0.9811 | 0.9140 | 0.3369 | 0.8169 |
| 10 | 90 | 64,740 | 0.9164 | 0.9298 | 0.9811 | 0.9548 | 0.3335 | 0.6625 |
| 0 | 100 | 58,270 | 0.9811 | 1.0000 | 0.9811 | 0.9905 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5933 | 0.0000 | 0.0000 | 0.0000 | 0.5933 | 1.0000 |
| 90 | 10 | 299,940 | 0.6335 | 0.2143 | 0.9998 | 0.3530 | 0.5928 | 1.0000 |
| 80 | 20 | 291,350 | 0.6739 | 0.3801 | 0.9998 | 0.5508 | 0.5924 | 0.9999 |
| 70 | 30 | 194,230 | 0.7151 | 0.5129 | 0.9998 | 0.6780 | 0.5930 | 0.9999 |
| 60 | 40 | 145,675 | 0.7547 | 0.6199 | 0.9998 | 0.7653 | 0.5913 | 0.9998 |
| 50 | 50 | 116,540 | 0.7956 | 0.7099 | 0.9998 | 0.8303 | 0.5915 | 0.9997 |
| 40 | 60 | 97,115 | 0.8367 | 0.7862 | 0.9998 | 0.8802 | 0.5921 | 0.9995 |
| 30 | 70 | 83,240 | 0.8782 | 0.8519 | 0.9998 | 0.9200 | 0.5945 | 0.9993 |
| 20 | 80 | 72,835 | 0.9191 | 0.9083 | 0.9998 | 0.9519 | 0.5963 | 0.9987 |
| 10 | 90 | 64,740 | 0.9585 | 0.9561 | 0.9998 | 0.9775 | 0.5871 | 0.9971 |
| 0 | 100 | 58,270 | 0.9998 | 1.0000 | 0.9998 | 0.9999 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5933 | 0.0000 | 0.0000 | 0.0000 | 0.5933 | 1.0000 |
| 90 | 10 | 299,940 | 0.6335 | 0.2143 | 0.9998 | 0.3530 | 0.5928 | 1.0000 |
| 80 | 20 | 291,350 | 0.6738 | 0.3801 | 0.9998 | 0.5508 | 0.5923 | 0.9999 |
| 70 | 30 | 194,230 | 0.7150 | 0.5129 | 0.9998 | 0.6780 | 0.5930 | 0.9998 |
| 60 | 40 | 145,675 | 0.7547 | 0.6199 | 0.9998 | 0.7653 | 0.5913 | 0.9997 |
| 50 | 50 | 116,540 | 0.7955 | 0.7098 | 0.9998 | 0.8302 | 0.5912 | 0.9996 |
| 40 | 60 | 97,115 | 0.8366 | 0.7861 | 0.9998 | 0.8801 | 0.5918 | 0.9994 |
| 30 | 70 | 83,240 | 0.8782 | 0.8519 | 0.9998 | 0.9200 | 0.5946 | 0.9991 |
| 20 | 80 | 72,835 | 0.9191 | 0.9083 | 0.9998 | 0.9518 | 0.5962 | 0.9985 |
| 10 | 90 | 64,740 | 0.9585 | 0.9561 | 0.9998 | 0.9775 | 0.5870 | 0.9966 |
| 0 | 100 | 58,270 | 0.9998 | 1.0000 | 0.9998 | 0.9999 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.5933 | 0.0000 | 0.0000 | 0.0000 | 0.5933 | 1.0000 |
| 90 | 10 | 299,940 | 0.6335 | 0.2143 | 0.9998 | 0.3530 | 0.5928 | 1.0000 |
| 80 | 20 | 291,350 | 0.6738 | 0.3801 | 0.9998 | 0.5508 | 0.5923 | 0.9999 |
| 70 | 30 | 194,230 | 0.7150 | 0.5129 | 0.9998 | 0.6780 | 0.5930 | 0.9998 |
| 60 | 40 | 145,675 | 0.7547 | 0.6199 | 0.9998 | 0.7653 | 0.5913 | 0.9997 |
| 50 | 50 | 116,540 | 0.7955 | 0.7098 | 0.9998 | 0.8302 | 0.5912 | 0.9996 |
| 40 | 60 | 97,115 | 0.8366 | 0.7861 | 0.9998 | 0.8801 | 0.5918 | 0.9994 |
| 30 | 70 | 83,240 | 0.8782 | 0.8519 | 0.9998 | 0.9200 | 0.5946 | 0.9991 |
| 20 | 80 | 72,835 | 0.9191 | 0.9083 | 0.9998 | 0.9518 | 0.5962 | 0.9985 |
| 10 | 90 | 64,740 | 0.9585 | 0.9561 | 0.9998 | 0.9775 | 0.5870 | 0.9966 |
| 0 | 100 | 58,270 | 0.9998 | 1.0000 | 0.9998 | 0.9999 | 0.0000 | 0.0000 |


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
0.15       0.4029   0.2474   0.3386   0.9940   0.9816   0.1416   <--
0.20       0.4029   0.2474   0.3386   0.9940   0.9816   0.1416  
0.25       0.4029   0.2474   0.3386   0.9940   0.9816   0.1416  
0.30       0.4029   0.2474   0.3386   0.9940   0.9816   0.1416  
0.35       0.4029   0.2474   0.3386   0.9940   0.9816   0.1416  
0.40       0.4029   0.2474   0.3386   0.9940   0.9816   0.1416  
0.45       0.4029   0.2474   0.3386   0.9940   0.9816   0.1416  
0.50       0.4029   0.2474   0.3386   0.9940   0.9816   0.1416  
0.55       0.4029   0.2474   0.3386   0.9940   0.9816   0.1416  
0.60       0.4029   0.2474   0.3386   0.9940   0.9816   0.1416  
0.65       0.4029   0.2474   0.3386   0.9940   0.9816   0.1416  
0.70       0.4029   0.2474   0.3386   0.9940   0.9816   0.1416  
0.75       0.4029   0.2474   0.3386   0.9940   0.9816   0.1416  
0.80       0.4029   0.2474   0.3386   0.9940   0.9816   0.1416  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4029, F1=0.2474, Normal Recall=0.3386, Normal Precision=0.9940, Attack Recall=0.9816, Attack Precision=0.1416

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
0.15       0.4674   0.4242   0.3389   0.9863   0.9811   0.2706   <--
0.20       0.4674   0.4242   0.3389   0.9863   0.9811   0.2706  
0.25       0.4674   0.4242   0.3389   0.9863   0.9811   0.2706  
0.30       0.4674   0.4242   0.3389   0.9863   0.9811   0.2706  
0.35       0.4674   0.4242   0.3389   0.9863   0.9811   0.2706  
0.40       0.4674   0.4242   0.3389   0.9863   0.9811   0.2706  
0.45       0.4674   0.4242   0.3389   0.9863   0.9811   0.2706  
0.50       0.4674   0.4242   0.3389   0.9863   0.9811   0.2706  
0.55       0.4674   0.4242   0.3389   0.9863   0.9811   0.2706  
0.60       0.4674   0.4242   0.3389   0.9863   0.9811   0.2706  
0.65       0.4674   0.4242   0.3389   0.9863   0.9811   0.2706  
0.70       0.4674   0.4242   0.3389   0.9863   0.9811   0.2706  
0.75       0.4674   0.4242   0.3389   0.9863   0.9811   0.2706  
0.80       0.4674   0.4242   0.3389   0.9863   0.9811   0.2706  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4674, F1=0.4242, Normal Recall=0.3389, Normal Precision=0.9863, Attack Recall=0.9811, Attack Precision=0.2706

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
0.15       0.5324   0.5573   0.3400   0.9768   0.9811   0.3892   <--
0.20       0.5324   0.5573   0.3400   0.9768   0.9811   0.3892  
0.25       0.5324   0.5573   0.3400   0.9768   0.9811   0.3892  
0.30       0.5324   0.5573   0.3400   0.9768   0.9811   0.3892  
0.35       0.5324   0.5573   0.3400   0.9768   0.9811   0.3892  
0.40       0.5324   0.5573   0.3400   0.9768   0.9811   0.3892  
0.45       0.5324   0.5573   0.3400   0.9768   0.9811   0.3892  
0.50       0.5324   0.5573   0.3400   0.9768   0.9811   0.3892  
0.55       0.5324   0.5573   0.3400   0.9768   0.9811   0.3892  
0.60       0.5324   0.5573   0.3400   0.9768   0.9811   0.3892  
0.65       0.5324   0.5573   0.3400   0.9768   0.9811   0.3892  
0.70       0.5324   0.5573   0.3400   0.9768   0.9811   0.3892  
0.75       0.5324   0.5573   0.3400   0.9768   0.9811   0.3892  
0.80       0.5324   0.5573   0.3400   0.9768   0.9811   0.3892  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5324, F1=0.5573, Normal Recall=0.3400, Normal Precision=0.9768, Attack Recall=0.9811, Attack Precision=0.3892

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
0.15       0.5960   0.6602   0.3393   0.9642   0.9811   0.4975   <--
0.20       0.5960   0.6602   0.3393   0.9642   0.9811   0.4975  
0.25       0.5960   0.6602   0.3393   0.9642   0.9811   0.4975  
0.30       0.5960   0.6602   0.3393   0.9642   0.9811   0.4975  
0.35       0.5960   0.6602   0.3393   0.9642   0.9811   0.4975  
0.40       0.5960   0.6602   0.3393   0.9642   0.9811   0.4975  
0.45       0.5960   0.6602   0.3393   0.9642   0.9811   0.4975  
0.50       0.5960   0.6602   0.3393   0.9642   0.9811   0.4975  
0.55       0.5960   0.6602   0.3393   0.9642   0.9811   0.4975  
0.60       0.5960   0.6602   0.3393   0.9642   0.9811   0.4975  
0.65       0.5960   0.6602   0.3393   0.9642   0.9811   0.4975  
0.70       0.5960   0.6602   0.3393   0.9642   0.9811   0.4975  
0.75       0.5960   0.6602   0.3393   0.9642   0.9811   0.4975  
0.80       0.5960   0.6602   0.3393   0.9642   0.9811   0.4975  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5960, F1=0.6602, Normal Recall=0.3393, Normal Precision=0.9642, Attack Recall=0.9811, Attack Precision=0.4975

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
0.15       0.6606   0.7430   0.3402   0.9474   0.9811   0.5979   <--
0.20       0.6606   0.7430   0.3402   0.9474   0.9811   0.5979  
0.25       0.6606   0.7430   0.3402   0.9474   0.9811   0.5979  
0.30       0.6606   0.7430   0.3402   0.9474   0.9811   0.5979  
0.35       0.6606   0.7430   0.3402   0.9474   0.9811   0.5979  
0.40       0.6606   0.7430   0.3402   0.9474   0.9811   0.5979  
0.45       0.6606   0.7430   0.3402   0.9474   0.9811   0.5979  
0.50       0.6606   0.7430   0.3402   0.9474   0.9811   0.5979  
0.55       0.6606   0.7430   0.3402   0.9474   0.9811   0.5979  
0.60       0.6606   0.7430   0.3402   0.9474   0.9811   0.5979  
0.65       0.6606   0.7430   0.3402   0.9474   0.9811   0.5979  
0.70       0.6606   0.7430   0.3402   0.9474   0.9811   0.5979  
0.75       0.6606   0.7430   0.3402   0.9474   0.9811   0.5979  
0.80       0.6606   0.7430   0.3402   0.9474   0.9811   0.5979  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6606, F1=0.7430, Normal Recall=0.3402, Normal Precision=0.9474, Attack Recall=0.9811, Attack Precision=0.5979

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
0.15       0.6335   0.3530   0.5928   1.0000   0.9998   0.2143   <--
0.20       0.6335   0.3530   0.5928   1.0000   0.9998   0.2143  
0.25       0.6335   0.3530   0.5928   1.0000   0.9998   0.2143  
0.30       0.6335   0.3530   0.5928   1.0000   0.9998   0.2143  
0.35       0.6335   0.3530   0.5928   1.0000   0.9998   0.2143  
0.40       0.6335   0.3530   0.5928   1.0000   0.9998   0.2143  
0.45       0.6335   0.3530   0.5928   1.0000   0.9998   0.2143  
0.50       0.6335   0.3530   0.5928   1.0000   0.9998   0.2143  
0.55       0.6335   0.3530   0.5928   1.0000   0.9998   0.2143  
0.60       0.6335   0.3530   0.5928   1.0000   0.9998   0.2143  
0.65       0.6335   0.3530   0.5928   1.0000   0.9998   0.2143  
0.70       0.6335   0.3530   0.5928   1.0000   0.9998   0.2143  
0.75       0.6335   0.3530   0.5928   1.0000   0.9998   0.2143  
0.80       0.6335   0.3530   0.5928   1.0000   0.9998   0.2143  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6335, F1=0.3530, Normal Recall=0.5928, Normal Precision=1.0000, Attack Recall=0.9998, Attack Precision=0.2143

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
0.15       0.6746   0.5514   0.5933   0.9999   0.9998   0.3807   <--
0.20       0.6746   0.5514   0.5933   0.9999   0.9998   0.3807  
0.25       0.6746   0.5514   0.5933   0.9999   0.9998   0.3807  
0.30       0.6746   0.5514   0.5933   0.9999   0.9998   0.3807  
0.35       0.6746   0.5514   0.5933   0.9999   0.9998   0.3807  
0.40       0.6746   0.5514   0.5933   0.9999   0.9998   0.3807  
0.45       0.6746   0.5514   0.5933   0.9999   0.9998   0.3807  
0.50       0.6746   0.5514   0.5933   0.9999   0.9998   0.3807  
0.55       0.6746   0.5514   0.5933   0.9999   0.9998   0.3807  
0.60       0.6746   0.5514   0.5933   0.9999   0.9998   0.3807  
0.65       0.6746   0.5514   0.5933   0.9999   0.9998   0.3807  
0.70       0.6746   0.5514   0.5933   0.9999   0.9998   0.3807  
0.75       0.6746   0.5514   0.5933   0.9999   0.9998   0.3807  
0.80       0.6746   0.5514   0.5933   0.9999   0.9998   0.3807  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6746, F1=0.5514, Normal Recall=0.5933, Normal Precision=0.9999, Attack Recall=0.9998, Attack Precision=0.3807

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
0.15       0.7152   0.6780   0.5932   0.9999   0.9998   0.5130   <--
0.20       0.7152   0.6780   0.5932   0.9999   0.9998   0.5130  
0.25       0.7152   0.6780   0.5932   0.9999   0.9998   0.5130  
0.30       0.7152   0.6780   0.5932   0.9999   0.9998   0.5130  
0.35       0.7152   0.6780   0.5932   0.9999   0.9998   0.5130  
0.40       0.7152   0.6780   0.5932   0.9999   0.9998   0.5130  
0.45       0.7152   0.6780   0.5932   0.9999   0.9998   0.5130  
0.50       0.7152   0.6780   0.5932   0.9999   0.9998   0.5130  
0.55       0.7152   0.6780   0.5932   0.9999   0.9998   0.5130  
0.60       0.7152   0.6780   0.5932   0.9999   0.9998   0.5130  
0.65       0.7152   0.6780   0.5932   0.9999   0.9998   0.5130  
0.70       0.7152   0.6780   0.5932   0.9999   0.9998   0.5130  
0.75       0.7152   0.6780   0.5932   0.9999   0.9998   0.5130  
0.80       0.7152   0.6780   0.5932   0.9999   0.9998   0.5130  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7152, F1=0.6780, Normal Recall=0.5932, Normal Precision=0.9999, Attack Recall=0.9998, Attack Precision=0.5130

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
0.15       0.7558   0.7661   0.5931   0.9998   0.9998   0.6210   <--
0.20       0.7558   0.7661   0.5931   0.9998   0.9998   0.6210  
0.25       0.7558   0.7661   0.5931   0.9998   0.9998   0.6210  
0.30       0.7558   0.7661   0.5931   0.9998   0.9998   0.6210  
0.35       0.7558   0.7661   0.5931   0.9998   0.9998   0.6210  
0.40       0.7558   0.7661   0.5931   0.9998   0.9998   0.6210  
0.45       0.7558   0.7661   0.5931   0.9998   0.9998   0.6210  
0.50       0.7558   0.7661   0.5931   0.9998   0.9998   0.6210  
0.55       0.7558   0.7661   0.5931   0.9998   0.9998   0.6210  
0.60       0.7558   0.7661   0.5931   0.9998   0.9998   0.6210  
0.65       0.7558   0.7661   0.5931   0.9998   0.9998   0.6210  
0.70       0.7558   0.7661   0.5931   0.9998   0.9998   0.6210  
0.75       0.7558   0.7661   0.5931   0.9998   0.9998   0.6210  
0.80       0.7558   0.7661   0.5931   0.9998   0.9998   0.6210  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7558, F1=0.7661, Normal Recall=0.5931, Normal Precision=0.9998, Attack Recall=0.9998, Attack Precision=0.6210

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
0.15       0.7956   0.8303   0.5915   0.9997   0.9998   0.7099   <--
0.20       0.7956   0.8303   0.5915   0.9997   0.9998   0.7099  
0.25       0.7956   0.8303   0.5915   0.9997   0.9998   0.7099  
0.30       0.7956   0.8303   0.5915   0.9997   0.9998   0.7099  
0.35       0.7956   0.8303   0.5915   0.9997   0.9998   0.7099  
0.40       0.7956   0.8303   0.5915   0.9997   0.9998   0.7099  
0.45       0.7956   0.8303   0.5915   0.9997   0.9998   0.7099  
0.50       0.7956   0.8303   0.5915   0.9997   0.9998   0.7099  
0.55       0.7956   0.8303   0.5915   0.9997   0.9998   0.7099  
0.60       0.7956   0.8303   0.5915   0.9997   0.9998   0.7099  
0.65       0.7956   0.8303   0.5915   0.9997   0.9998   0.7099  
0.70       0.7956   0.8303   0.5915   0.9997   0.9998   0.7099  
0.75       0.7956   0.8303   0.5915   0.9997   0.9998   0.7099  
0.80       0.7956   0.8303   0.5915   0.9997   0.9998   0.7099  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7956, F1=0.8303, Normal Recall=0.5915, Normal Precision=0.9997, Attack Recall=0.9998, Attack Precision=0.7099

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
0.15       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143   <--
0.20       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.25       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.30       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.35       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.40       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.45       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.50       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.55       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.60       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.65       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.70       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.75       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.80       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6335, F1=0.3530, Normal Recall=0.5928, Normal Precision=1.0000, Attack Recall=0.9997, Attack Precision=0.2143

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
0.15       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806   <--
0.20       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.25       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.30       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.35       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.40       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.45       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.50       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.55       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.60       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.65       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.70       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.75       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.80       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6746, F1=0.5513, Normal Recall=0.5933, Normal Precision=0.9999, Attack Recall=0.9998, Attack Precision=0.3806

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
0.15       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130   <--
0.20       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.25       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.30       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.35       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.40       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.45       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.50       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.55       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.60       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.65       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.70       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.75       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.80       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7151, F1=0.6780, Normal Recall=0.5932, Normal Precision=0.9998, Attack Recall=0.9998, Attack Precision=0.5130

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
0.15       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209   <--
0.20       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.25       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.30       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.35       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.40       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.45       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.50       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.55       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.60       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.65       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.70       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.75       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.80       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7558, F1=0.7661, Normal Recall=0.5931, Normal Precision=0.9997, Attack Recall=0.9998, Attack Precision=0.6209

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
0.15       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098   <--
0.20       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.25       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.30       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.35       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.40       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.45       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.50       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.55       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.60       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.65       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.70       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.75       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.80       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7956, F1=0.8302, Normal Recall=0.5913, Normal Precision=0.9996, Attack Recall=0.9998, Attack Precision=0.7098

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
0.15       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143   <--
0.20       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.25       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.30       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.35       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.40       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.45       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.50       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.55       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.60       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.65       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.70       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.75       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
0.80       0.6335   0.3530   0.5928   1.0000   0.9997   0.2143  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6335, F1=0.3530, Normal Recall=0.5928, Normal Precision=1.0000, Attack Recall=0.9997, Attack Precision=0.2143

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
0.15       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806   <--
0.20       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.25       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.30       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.35       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.40       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.45       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.50       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.55       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.60       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.65       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.70       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.75       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
0.80       0.6746   0.5513   0.5933   0.9999   0.9998   0.3806  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6746, F1=0.5513, Normal Recall=0.5933, Normal Precision=0.9999, Attack Recall=0.9998, Attack Precision=0.3806

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
0.15       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130   <--
0.20       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.25       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.30       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.35       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.40       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.45       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.50       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.55       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.60       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.65       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.70       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.75       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
0.80       0.7151   0.6780   0.5932   0.9998   0.9998   0.5130  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7151, F1=0.6780, Normal Recall=0.5932, Normal Precision=0.9998, Attack Recall=0.9998, Attack Precision=0.5130

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
0.15       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209   <--
0.20       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.25       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.30       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.35       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.40       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.45       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.50       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.55       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.60       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.65       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.70       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.75       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
0.80       0.7558   0.7661   0.5931   0.9997   0.9998   0.6209  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7558, F1=0.7661, Normal Recall=0.5931, Normal Precision=0.9997, Attack Recall=0.9998, Attack Precision=0.6209

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
0.15       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098   <--
0.20       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.25       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.30       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.35       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.40       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.45       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.50       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.55       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.60       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.65       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.70       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.75       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
0.80       0.7956   0.8302   0.5913   0.9996   0.9998   0.7098  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7956, F1=0.8302, Normal Recall=0.5913, Normal Precision=0.9996, Attack Recall=0.9998, Attack Precision=0.7098

```

