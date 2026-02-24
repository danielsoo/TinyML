# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e2_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-21 17:31:07 |

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
| Original (TFLite) | 0.1472 | 0.2319 | 0.3167 | 0.4024 | 0.4866 | 0.5718 | 0.6574 | 0.7416 | 0.8270 | 0.9123 | 0.9969 |
| QAT+Prune only | 0.7338 | 0.7607 | 0.7865 | 0.8130 | 0.8393 | 0.8642 | 0.8913 | 0.9172 | 0.9441 | 0.9696 | 0.9972 |
| QAT+PTQ | 0.7327 | 0.7596 | 0.7856 | 0.8122 | 0.8386 | 0.8637 | 0.8908 | 0.9169 | 0.9439 | 0.9696 | 0.9972 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7327 | 0.7596 | 0.7856 | 0.8122 | 0.8386 | 0.8637 | 0.8908 | 0.9169 | 0.9439 | 0.9696 | 0.9972 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2061 | 0.3685 | 0.5002 | 0.6084 | 0.6995 | 0.7774 | 0.8438 | 0.9021 | 0.9534 | 0.9984 |
| QAT+Prune only | 0.0000 | 0.4545 | 0.6514 | 0.7618 | 0.8323 | 0.8801 | 0.9168 | 0.9440 | 0.9662 | 0.9834 | 0.9986 |
| QAT+PTQ | 0.0000 | 0.4534 | 0.6504 | 0.7611 | 0.8317 | 0.8797 | 0.9163 | 0.9438 | 0.9660 | 0.9833 | 0.9986 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4534 | 0.6504 | 0.7611 | 0.8317 | 0.8797 | 0.9163 | 0.9438 | 0.9660 | 0.9833 | 0.9986 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1472 | 0.1469 | 0.1467 | 0.1477 | 0.1464 | 0.1466 | 0.1481 | 0.1460 | 0.1474 | 0.1506 | 0.0000 |
| QAT+Prune only | 0.7338 | 0.7344 | 0.7339 | 0.7340 | 0.7340 | 0.7312 | 0.7326 | 0.7305 | 0.7319 | 0.7218 | 0.0000 |
| QAT+PTQ | 0.7327 | 0.7333 | 0.7327 | 0.7329 | 0.7328 | 0.7302 | 0.7311 | 0.7297 | 0.7308 | 0.7209 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7327 | 0.7333 | 0.7327 | 0.7329 | 0.7328 | 0.7302 | 0.7311 | 0.7297 | 0.7308 | 0.7209 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1472 | 0.0000 | 0.0000 | 0.0000 | 0.1472 | 1.0000 |
| 90 | 10 | 299,940 | 0.2319 | 0.1149 | 0.9969 | 0.2061 | 0.1469 | 0.9977 |
| 80 | 20 | 291,350 | 0.3167 | 0.2260 | 0.9969 | 0.3685 | 0.1467 | 0.9947 |
| 70 | 30 | 194,230 | 0.4024 | 0.3339 | 0.9969 | 0.5002 | 0.1477 | 0.9911 |
| 60 | 40 | 145,675 | 0.4866 | 0.4378 | 0.9969 | 0.6084 | 0.1464 | 0.9861 |
| 50 | 50 | 116,540 | 0.5718 | 0.5388 | 0.9969 | 0.6995 | 0.1466 | 0.9793 |
| 40 | 60 | 97,115 | 0.6574 | 0.6371 | 0.9969 | 0.7774 | 0.1481 | 0.9695 |
| 30 | 70 | 83,240 | 0.7416 | 0.7315 | 0.9969 | 0.8438 | 0.1460 | 0.9527 |
| 20 | 80 | 72,835 | 0.8270 | 0.8238 | 0.9969 | 0.9021 | 0.1474 | 0.9223 |
| 10 | 90 | 64,740 | 0.9123 | 0.9135 | 0.9969 | 0.9534 | 0.1506 | 0.8434 |
| 0 | 100 | 58,270 | 0.9969 | 1.0000 | 0.9969 | 0.9984 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7338 | 0.0000 | 0.0000 | 0.0000 | 0.7338 | 1.0000 |
| 90 | 10 | 299,940 | 0.7607 | 0.2944 | 0.9970 | 0.4545 | 0.7344 | 0.9996 |
| 80 | 20 | 291,350 | 0.7865 | 0.4837 | 0.9972 | 0.6514 | 0.7339 | 0.9990 |
| 70 | 30 | 194,230 | 0.8130 | 0.6164 | 0.9972 | 0.7618 | 0.7340 | 0.9984 |
| 60 | 40 | 145,675 | 0.8393 | 0.7142 | 0.9972 | 0.8323 | 0.7340 | 0.9975 |
| 50 | 50 | 116,540 | 0.8642 | 0.7877 | 0.9972 | 0.8801 | 0.7312 | 0.9962 |
| 40 | 60 | 97,115 | 0.8913 | 0.8483 | 0.9972 | 0.9168 | 0.7326 | 0.9943 |
| 30 | 70 | 83,240 | 0.9172 | 0.8962 | 0.9972 | 0.9440 | 0.7305 | 0.9911 |
| 20 | 80 | 72,835 | 0.9441 | 0.9370 | 0.9972 | 0.9662 | 0.7319 | 0.9848 |
| 10 | 90 | 64,740 | 0.9696 | 0.9699 | 0.9972 | 0.9834 | 0.7218 | 0.9661 |
| 0 | 100 | 58,270 | 0.9972 | 1.0000 | 0.9972 | 0.9986 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7327 | 0.0000 | 0.0000 | 0.0000 | 0.7327 | 1.0000 |
| 90 | 10 | 299,940 | 0.7596 | 0.2934 | 0.9970 | 0.4534 | 0.7333 | 0.9996 |
| 80 | 20 | 291,350 | 0.7856 | 0.4825 | 0.9972 | 0.6504 | 0.7327 | 0.9990 |
| 70 | 30 | 194,230 | 0.8122 | 0.6154 | 0.9972 | 0.7611 | 0.7329 | 0.9984 |
| 60 | 40 | 145,675 | 0.8386 | 0.7133 | 0.9972 | 0.8317 | 0.7328 | 0.9974 |
| 50 | 50 | 116,540 | 0.8637 | 0.7870 | 0.9972 | 0.8797 | 0.7302 | 0.9962 |
| 40 | 60 | 97,115 | 0.8908 | 0.8476 | 0.9972 | 0.9163 | 0.7311 | 0.9943 |
| 30 | 70 | 83,240 | 0.9169 | 0.8959 | 0.9972 | 0.9438 | 0.7297 | 0.9911 |
| 20 | 80 | 72,835 | 0.9439 | 0.9368 | 0.9972 | 0.9660 | 0.7308 | 0.9848 |
| 10 | 90 | 64,740 | 0.9696 | 0.9698 | 0.9972 | 0.9833 | 0.7209 | 0.9661 |
| 0 | 100 | 58,270 | 0.9972 | 1.0000 | 0.9972 | 0.9986 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7327 | 0.0000 | 0.0000 | 0.0000 | 0.7327 | 1.0000 |
| 90 | 10 | 299,940 | 0.7596 | 0.2934 | 0.9970 | 0.4534 | 0.7333 | 0.9996 |
| 80 | 20 | 291,350 | 0.7856 | 0.4825 | 0.9972 | 0.6504 | 0.7327 | 0.9990 |
| 70 | 30 | 194,230 | 0.8122 | 0.6154 | 0.9972 | 0.7611 | 0.7329 | 0.9984 |
| 60 | 40 | 145,675 | 0.8386 | 0.7133 | 0.9972 | 0.8317 | 0.7328 | 0.9974 |
| 50 | 50 | 116,540 | 0.8637 | 0.7870 | 0.9972 | 0.8797 | 0.7302 | 0.9962 |
| 40 | 60 | 97,115 | 0.8908 | 0.8476 | 0.9972 | 0.9163 | 0.7311 | 0.9943 |
| 30 | 70 | 83,240 | 0.9169 | 0.8959 | 0.9972 | 0.9438 | 0.7297 | 0.9911 |
| 20 | 80 | 72,835 | 0.9439 | 0.9368 | 0.9972 | 0.9660 | 0.7308 | 0.9848 |
| 10 | 90 | 64,740 | 0.9696 | 0.9698 | 0.9972 | 0.9833 | 0.7209 | 0.9661 |
| 0 | 100 | 58,270 | 0.9972 | 1.0000 | 0.9972 | 0.9986 | 0.0000 | 0.0000 |


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
0.15       0.2319   0.2061   0.1469   0.9977   0.9970   0.1149   <--
0.20       0.2319   0.2061   0.1469   0.9977   0.9970   0.1149  
0.25       0.2319   0.2061   0.1469   0.9977   0.9970   0.1149  
0.30       0.2319   0.2061   0.1469   0.9977   0.9970   0.1149  
0.35       0.2319   0.2061   0.1469   0.9977   0.9970   0.1149  
0.40       0.2319   0.2061   0.1469   0.9977   0.9970   0.1149  
0.45       0.2319   0.2061   0.1469   0.9977   0.9970   0.1149  
0.50       0.2319   0.2061   0.1469   0.9977   0.9970   0.1149  
0.55       0.2319   0.2061   0.1469   0.9977   0.9970   0.1149  
0.60       0.2319   0.2061   0.1469   0.9977   0.9970   0.1149  
0.65       0.2319   0.2061   0.1469   0.9977   0.9970   0.1149  
0.70       0.2319   0.2061   0.1469   0.9977   0.9970   0.1149  
0.75       0.2319   0.2061   0.1469   0.9977   0.9970   0.1149  
0.80       0.2319   0.2061   0.1469   0.9977   0.9970   0.1149  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2319, F1=0.2061, Normal Recall=0.1469, Normal Precision=0.9977, Attack Recall=0.9970, Attack Precision=0.1149

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
0.15       0.3169   0.3686   0.1469   0.9947   0.9969   0.2261   <--
0.20       0.3169   0.3686   0.1469   0.9947   0.9969   0.2261  
0.25       0.3169   0.3686   0.1469   0.9947   0.9969   0.2261  
0.30       0.3169   0.3686   0.1469   0.9947   0.9969   0.2261  
0.35       0.3169   0.3686   0.1469   0.9947   0.9969   0.2261  
0.40       0.3169   0.3686   0.1469   0.9947   0.9969   0.2261  
0.45       0.3169   0.3686   0.1469   0.9947   0.9969   0.2261  
0.50       0.3169   0.3686   0.1469   0.9947   0.9969   0.2261  
0.55       0.3169   0.3686   0.1469   0.9947   0.9969   0.2261  
0.60       0.3169   0.3686   0.1469   0.9947   0.9969   0.2261  
0.65       0.3169   0.3686   0.1469   0.9947   0.9969   0.2261  
0.70       0.3169   0.3686   0.1469   0.9947   0.9969   0.2261  
0.75       0.3169   0.3686   0.1469   0.9947   0.9969   0.2261  
0.80       0.3169   0.3686   0.1469   0.9947   0.9969   0.2261  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3169, F1=0.3686, Normal Recall=0.1469, Normal Precision=0.9947, Attack Recall=0.9969, Attack Precision=0.2261

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
0.15       0.4019   0.5000   0.1468   0.9910   0.9969   0.3337   <--
0.20       0.4019   0.5000   0.1468   0.9910   0.9969   0.3337  
0.25       0.4019   0.5000   0.1468   0.9910   0.9969   0.3337  
0.30       0.4019   0.5000   0.1468   0.9910   0.9969   0.3337  
0.35       0.4019   0.5000   0.1468   0.9910   0.9969   0.3337  
0.40       0.4019   0.5000   0.1468   0.9910   0.9969   0.3337  
0.45       0.4019   0.5000   0.1468   0.9910   0.9969   0.3337  
0.50       0.4019   0.5000   0.1468   0.9910   0.9969   0.3337  
0.55       0.4019   0.5000   0.1468   0.9910   0.9969   0.3337  
0.60       0.4019   0.5000   0.1468   0.9910   0.9969   0.3337  
0.65       0.4019   0.5000   0.1468   0.9910   0.9969   0.3337  
0.70       0.4019   0.5000   0.1468   0.9910   0.9969   0.3337  
0.75       0.4019   0.5000   0.1468   0.9910   0.9969   0.3337  
0.80       0.4019   0.5000   0.1468   0.9910   0.9969   0.3337  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4019, F1=0.5000, Normal Recall=0.1468, Normal Precision=0.9910, Attack Recall=0.9969, Attack Precision=0.3337

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
0.15       0.4872   0.6086   0.1474   0.9861   0.9969   0.4380   <--
0.20       0.4872   0.6086   0.1474   0.9861   0.9969   0.4380  
0.25       0.4872   0.6086   0.1474   0.9861   0.9969   0.4380  
0.30       0.4872   0.6086   0.1474   0.9861   0.9969   0.4380  
0.35       0.4872   0.6086   0.1474   0.9861   0.9969   0.4380  
0.40       0.4872   0.6086   0.1474   0.9861   0.9969   0.4380  
0.45       0.4872   0.6086   0.1474   0.9861   0.9969   0.4380  
0.50       0.4872   0.6086   0.1474   0.9861   0.9969   0.4380  
0.55       0.4872   0.6086   0.1474   0.9861   0.9969   0.4380  
0.60       0.4872   0.6086   0.1474   0.9861   0.9969   0.4380  
0.65       0.4872   0.6086   0.1474   0.9861   0.9969   0.4380  
0.70       0.4872   0.6086   0.1474   0.9861   0.9969   0.4380  
0.75       0.4872   0.6086   0.1474   0.9861   0.9969   0.4380  
0.80       0.4872   0.6086   0.1474   0.9861   0.9969   0.4380  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4872, F1=0.6086, Normal Recall=0.1474, Normal Precision=0.9861, Attack Recall=0.9969, Attack Precision=0.4380

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
0.15       0.5721   0.6997   0.1473   0.9793   0.9969   0.5390   <--
0.20       0.5721   0.6997   0.1473   0.9793   0.9969   0.5390  
0.25       0.5721   0.6997   0.1473   0.9793   0.9969   0.5390  
0.30       0.5721   0.6997   0.1473   0.9793   0.9969   0.5390  
0.35       0.5721   0.6997   0.1473   0.9793   0.9969   0.5390  
0.40       0.5721   0.6997   0.1473   0.9793   0.9969   0.5390  
0.45       0.5721   0.6997   0.1473   0.9793   0.9969   0.5390  
0.50       0.5721   0.6997   0.1473   0.9793   0.9969   0.5390  
0.55       0.5721   0.6997   0.1473   0.9793   0.9969   0.5390  
0.60       0.5721   0.6997   0.1473   0.9793   0.9969   0.5390  
0.65       0.5721   0.6997   0.1473   0.9793   0.9969   0.5390  
0.70       0.5721   0.6997   0.1473   0.9793   0.9969   0.5390  
0.75       0.5721   0.6997   0.1473   0.9793   0.9969   0.5390  
0.80       0.5721   0.6997   0.1473   0.9793   0.9969   0.5390  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5721, F1=0.6997, Normal Recall=0.1473, Normal Precision=0.9793, Attack Recall=0.9969, Attack Precision=0.5390

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
0.15       0.7607   0.4547   0.7344   0.9996   0.9974   0.2944   <--
0.20       0.7607   0.4547   0.7344   0.9996   0.9974   0.2944  
0.25       0.7607   0.4547   0.7344   0.9996   0.9974   0.2944  
0.30       0.7607   0.4547   0.7344   0.9996   0.9974   0.2944  
0.35       0.7607   0.4547   0.7344   0.9996   0.9974   0.2944  
0.40       0.7607   0.4547   0.7344   0.9996   0.9974   0.2944  
0.45       0.7607   0.4547   0.7344   0.9996   0.9974   0.2944  
0.50       0.7607   0.4547   0.7344   0.9996   0.9974   0.2944  
0.55       0.7607   0.4547   0.7344   0.9996   0.9974   0.2944  
0.60       0.7607   0.4547   0.7344   0.9996   0.9974   0.2944  
0.65       0.7607   0.4547   0.7344   0.9996   0.9974   0.2944  
0.70       0.7607   0.4547   0.7344   0.9996   0.9974   0.2944  
0.75       0.7607   0.4547   0.7344   0.9996   0.9974   0.2944  
0.80       0.7607   0.4547   0.7344   0.9996   0.9974   0.2944  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7607, F1=0.4547, Normal Recall=0.7344, Normal Precision=0.9996, Attack Recall=0.9974, Attack Precision=0.2944

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
0.15       0.7873   0.6522   0.7348   0.9990   0.9972   0.4846   <--
0.20       0.7873   0.6522   0.7348   0.9990   0.9972   0.4846  
0.25       0.7873   0.6522   0.7348   0.9990   0.9972   0.4846  
0.30       0.7873   0.6522   0.7348   0.9990   0.9972   0.4846  
0.35       0.7873   0.6522   0.7348   0.9990   0.9972   0.4846  
0.40       0.7873   0.6522   0.7348   0.9990   0.9972   0.4846  
0.45       0.7873   0.6522   0.7348   0.9990   0.9972   0.4846  
0.50       0.7873   0.6522   0.7348   0.9990   0.9972   0.4846  
0.55       0.7873   0.6522   0.7348   0.9990   0.9972   0.4846  
0.60       0.7873   0.6522   0.7348   0.9990   0.9972   0.4846  
0.65       0.7873   0.6522   0.7348   0.9990   0.9972   0.4846  
0.70       0.7873   0.6522   0.7348   0.9990   0.9972   0.4846  
0.75       0.7873   0.6522   0.7348   0.9990   0.9972   0.4846  
0.80       0.7873   0.6522   0.7348   0.9990   0.9972   0.4846  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7873, F1=0.6522, Normal Recall=0.7348, Normal Precision=0.9990, Attack Recall=0.9972, Attack Precision=0.4846

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
0.15       0.8128   0.7617   0.7338   0.9984   0.9972   0.6162   <--
0.20       0.8128   0.7617   0.7338   0.9984   0.9972   0.6162  
0.25       0.8128   0.7617   0.7338   0.9984   0.9972   0.6162  
0.30       0.8128   0.7617   0.7338   0.9984   0.9972   0.6162  
0.35       0.8128   0.7617   0.7338   0.9984   0.9972   0.6162  
0.40       0.8128   0.7617   0.7338   0.9984   0.9972   0.6162  
0.45       0.8128   0.7617   0.7338   0.9984   0.9972   0.6162  
0.50       0.8128   0.7617   0.7338   0.9984   0.9972   0.6162  
0.55       0.8128   0.7617   0.7338   0.9984   0.9972   0.6162  
0.60       0.8128   0.7617   0.7338   0.9984   0.9972   0.6162  
0.65       0.8128   0.7617   0.7338   0.9984   0.9972   0.6162  
0.70       0.8128   0.7617   0.7338   0.9984   0.9972   0.6162  
0.75       0.8128   0.7617   0.7338   0.9984   0.9972   0.6162  
0.80       0.8128   0.7617   0.7338   0.9984   0.9972   0.6162  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8128, F1=0.7617, Normal Recall=0.7338, Normal Precision=0.9984, Attack Recall=0.9972, Attack Precision=0.6162

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
0.15       0.8391   0.8321   0.7337   0.9974   0.9972   0.7140   <--
0.20       0.8391   0.8321   0.7337   0.9974   0.9972   0.7140  
0.25       0.8391   0.8321   0.7337   0.9974   0.9972   0.7140  
0.30       0.8391   0.8321   0.7337   0.9974   0.9972   0.7140  
0.35       0.8391   0.8321   0.7337   0.9974   0.9972   0.7140  
0.40       0.8391   0.8321   0.7337   0.9974   0.9972   0.7140  
0.45       0.8391   0.8321   0.7337   0.9974   0.9972   0.7140  
0.50       0.8391   0.8321   0.7337   0.9974   0.9972   0.7140  
0.55       0.8391   0.8321   0.7337   0.9974   0.9972   0.7140  
0.60       0.8391   0.8321   0.7337   0.9974   0.9972   0.7140  
0.65       0.8391   0.8321   0.7337   0.9974   0.9972   0.7140  
0.70       0.8391   0.8321   0.7337   0.9974   0.9972   0.7140  
0.75       0.8391   0.8321   0.7337   0.9974   0.9972   0.7140  
0.80       0.8391   0.8321   0.7337   0.9974   0.9972   0.7140  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8391, F1=0.8321, Normal Recall=0.7337, Normal Precision=0.9974, Attack Recall=0.9972, Attack Precision=0.7140

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
0.15       0.8646   0.8805   0.7320   0.9962   0.9972   0.7882   <--
0.20       0.8646   0.8805   0.7320   0.9962   0.9972   0.7882  
0.25       0.8646   0.8805   0.7320   0.9962   0.9972   0.7882  
0.30       0.8646   0.8805   0.7320   0.9962   0.9972   0.7882  
0.35       0.8646   0.8805   0.7320   0.9962   0.9972   0.7882  
0.40       0.8646   0.8805   0.7320   0.9962   0.9972   0.7882  
0.45       0.8646   0.8805   0.7320   0.9962   0.9972   0.7882  
0.50       0.8646   0.8805   0.7320   0.9962   0.9972   0.7882  
0.55       0.8646   0.8805   0.7320   0.9962   0.9972   0.7882  
0.60       0.8646   0.8805   0.7320   0.9962   0.9972   0.7882  
0.65       0.8646   0.8805   0.7320   0.9962   0.9972   0.7882  
0.70       0.8646   0.8805   0.7320   0.9962   0.9972   0.7882  
0.75       0.8646   0.8805   0.7320   0.9962   0.9972   0.7882  
0.80       0.8646   0.8805   0.7320   0.9962   0.9972   0.7882  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8646, F1=0.8805, Normal Recall=0.7320, Normal Precision=0.9962, Attack Recall=0.9972, Attack Precision=0.7882

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
0.15       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935   <--
0.20       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.25       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.30       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.35       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.40       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.45       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.50       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.55       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.60       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.65       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.70       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.75       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.80       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7597, F1=0.4536, Normal Recall=0.7333, Normal Precision=0.9996, Attack Recall=0.9974, Attack Precision=0.2935

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
0.15       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835   <--
0.20       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.25       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.30       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.35       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.40       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.45       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.50       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.55       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.60       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.65       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.70       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.75       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.80       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7864, F1=0.6512, Normal Recall=0.7337, Normal Precision=0.9990, Attack Recall=0.9972, Attack Precision=0.4835

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
0.15       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152   <--
0.20       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.25       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.30       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.35       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.40       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.45       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.50       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.55       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.60       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.65       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.70       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.75       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.80       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8121, F1=0.7610, Normal Recall=0.7327, Normal Precision=0.9984, Attack Recall=0.9972, Attack Precision=0.6152

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
0.15       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131   <--
0.20       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.25       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.30       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.35       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.40       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.45       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.50       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.55       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.60       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.65       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.70       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.75       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.80       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8384, F1=0.8316, Normal Recall=0.7326, Normal Precision=0.9974, Attack Recall=0.9972, Attack Precision=0.7131

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
0.15       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875   <--
0.20       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.25       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.30       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.35       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.40       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.45       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.50       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.55       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.60       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.65       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.70       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.75       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.80       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8640, F1=0.8800, Normal Recall=0.7309, Normal Precision=0.9962, Attack Recall=0.9972, Attack Precision=0.7875

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
0.15       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935   <--
0.20       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.25       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.30       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.35       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.40       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.45       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.50       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.55       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.60       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.65       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.70       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.75       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
0.80       0.7597   0.4536   0.7333   0.9996   0.9974   0.2935  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7597, F1=0.4536, Normal Recall=0.7333, Normal Precision=0.9996, Attack Recall=0.9974, Attack Precision=0.2935

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
0.15       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835   <--
0.20       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.25       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.30       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.35       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.40       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.45       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.50       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.55       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.60       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.65       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.70       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.75       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
0.80       0.7864   0.6512   0.7337   0.9990   0.9972   0.4835  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7864, F1=0.6512, Normal Recall=0.7337, Normal Precision=0.9990, Attack Recall=0.9972, Attack Precision=0.4835

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
0.15       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152   <--
0.20       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.25       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.30       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.35       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.40       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.45       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.50       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.55       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.60       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.65       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.70       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.75       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
0.80       0.8121   0.7610   0.7327   0.9984   0.9972   0.6152  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8121, F1=0.7610, Normal Recall=0.7327, Normal Precision=0.9984, Attack Recall=0.9972, Attack Precision=0.6152

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
0.15       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131   <--
0.20       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.25       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.30       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.35       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.40       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.45       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.50       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.55       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.60       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.65       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.70       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.75       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
0.80       0.8384   0.8316   0.7326   0.9974   0.9972   0.7131  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8384, F1=0.8316, Normal Recall=0.7326, Normal Precision=0.9974, Attack Recall=0.9972, Attack Precision=0.7131

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
0.15       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875   <--
0.20       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.25       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.30       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.35       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.40       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.45       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.50       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.55       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.60       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.65       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.70       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.75       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
0.80       0.8640   0.8800   0.7309   0.9962   0.9972   0.7875  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8640, F1=0.8800, Normal Recall=0.7309, Normal Precision=0.9962, Attack Recall=0.9972, Attack Precision=0.7875

```

