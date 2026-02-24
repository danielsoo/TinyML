# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e1_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-18 11:51:38 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 80 |
| **Local epochs** | 1 |
| **Batch size** | 256 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9160 | 0.9221 | 0.9279 | 0.9340 | 0.9391 | 0.9447 | 0.9514 | 0.9564 | 0.9622 | 0.9676 | 0.9740 |
| QAT+Prune only | 0.8283 | 0.8265 | 0.8259 | 0.8265 | 0.8253 | 0.8241 | 0.8247 | 0.8230 | 0.8234 | 0.8223 | 0.8224 |
| QAT+PTQ | 0.8269 | 0.8252 | 0.8247 | 0.8254 | 0.8246 | 0.8234 | 0.8241 | 0.8227 | 0.8232 | 0.8222 | 0.8226 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8269 | 0.8252 | 0.8247 | 0.8254 | 0.8246 | 0.8234 | 0.8241 | 0.8227 | 0.8232 | 0.8222 | 0.8226 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.7142 | 0.8439 | 0.8986 | 0.9275 | 0.9462 | 0.9601 | 0.9690 | 0.9763 | 0.9819 | 0.9868 |
| QAT+Prune only | 0.0000 | 0.4862 | 0.6539 | 0.7399 | 0.7902 | 0.8238 | 0.8491 | 0.8667 | 0.8817 | 0.8928 | 0.9026 |
| QAT+PTQ | 0.0000 | 0.4844 | 0.6525 | 0.7387 | 0.7895 | 0.8232 | 0.8487 | 0.8666 | 0.8816 | 0.8928 | 0.9027 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4844 | 0.6525 | 0.7387 | 0.7895 | 0.8232 | 0.8487 | 0.8666 | 0.8816 | 0.8928 | 0.9027 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9160 | 0.9163 | 0.9164 | 0.9169 | 0.9158 | 0.9153 | 0.9174 | 0.9152 | 0.9152 | 0.9104 | 0.0000 |
| QAT+Prune only | 0.8283 | 0.8271 | 0.8268 | 0.8283 | 0.8273 | 0.8257 | 0.8280 | 0.8243 | 0.8271 | 0.8208 | 0.0000 |
| QAT+PTQ | 0.8269 | 0.8256 | 0.8253 | 0.8266 | 0.8259 | 0.8242 | 0.8263 | 0.8230 | 0.8256 | 0.8187 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8269 | 0.8256 | 0.8253 | 0.8266 | 0.8259 | 0.8242 | 0.8263 | 0.8230 | 0.8256 | 0.8187 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9160 | 0.0000 | 0.0000 | 0.0000 | 0.9160 | 1.0000 |
| 90 | 10 | 299,940 | 0.9221 | 0.5638 | 0.9739 | 0.7142 | 0.9163 | 0.9968 |
| 80 | 20 | 291,350 | 0.9279 | 0.7444 | 0.9740 | 0.8439 | 0.9164 | 0.9930 |
| 70 | 30 | 194,230 | 0.9340 | 0.8340 | 0.9740 | 0.8986 | 0.9169 | 0.9880 |
| 60 | 40 | 145,675 | 0.9391 | 0.8852 | 0.9740 | 0.9275 | 0.9158 | 0.9814 |
| 50 | 50 | 116,540 | 0.9447 | 0.9200 | 0.9740 | 0.9462 | 0.9153 | 0.9724 |
| 40 | 60 | 97,115 | 0.9514 | 0.9465 | 0.9740 | 0.9601 | 0.9174 | 0.9592 |
| 30 | 70 | 83,240 | 0.9564 | 0.9640 | 0.9740 | 0.9690 | 0.9152 | 0.9378 |
| 20 | 80 | 72,835 | 0.9622 | 0.9787 | 0.9740 | 0.9763 | 0.9152 | 0.8980 |
| 10 | 90 | 64,740 | 0.9676 | 0.9899 | 0.9740 | 0.9819 | 0.9104 | 0.7955 |
| 0 | 100 | 58,270 | 0.9740 | 1.0000 | 0.9740 | 0.9868 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8283 | 0.0000 | 0.0000 | 0.0000 | 0.8283 | 1.0000 |
| 90 | 10 | 299,940 | 0.8265 | 0.3454 | 0.8211 | 0.4862 | 0.8271 | 0.9765 |
| 80 | 20 | 291,350 | 0.8259 | 0.5427 | 0.8224 | 0.6539 | 0.8268 | 0.9490 |
| 70 | 30 | 194,230 | 0.8265 | 0.6724 | 0.8224 | 0.7399 | 0.8283 | 0.9158 |
| 60 | 40 | 145,675 | 0.8253 | 0.7605 | 0.8224 | 0.7902 | 0.8273 | 0.8748 |
| 50 | 50 | 116,540 | 0.8241 | 0.8251 | 0.8224 | 0.8238 | 0.8257 | 0.8230 |
| 40 | 60 | 97,115 | 0.8247 | 0.8776 | 0.8224 | 0.8491 | 0.8280 | 0.7566 |
| 30 | 70 | 83,240 | 0.8230 | 0.9161 | 0.8224 | 0.8667 | 0.8243 | 0.6655 |
| 20 | 80 | 72,835 | 0.8234 | 0.9501 | 0.8224 | 0.8817 | 0.8271 | 0.5380 |
| 10 | 90 | 64,740 | 0.8223 | 0.9764 | 0.8224 | 0.8928 | 0.8208 | 0.3393 |
| 0 | 100 | 58,270 | 0.8224 | 1.0000 | 0.8224 | 0.9026 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8269 | 0.0000 | 0.0000 | 0.0000 | 0.8269 | 1.0000 |
| 90 | 10 | 299,940 | 0.8252 | 0.3435 | 0.8212 | 0.4844 | 0.8256 | 0.9765 |
| 80 | 20 | 291,350 | 0.8247 | 0.5406 | 0.8226 | 0.6525 | 0.8253 | 0.9490 |
| 70 | 30 | 194,230 | 0.8254 | 0.6703 | 0.8226 | 0.7387 | 0.8266 | 0.9158 |
| 60 | 40 | 145,675 | 0.8246 | 0.7590 | 0.8226 | 0.7895 | 0.8259 | 0.8747 |
| 50 | 50 | 116,540 | 0.8234 | 0.8239 | 0.8226 | 0.8232 | 0.8242 | 0.8229 |
| 40 | 60 | 97,115 | 0.8241 | 0.8766 | 0.8226 | 0.8487 | 0.8263 | 0.7564 |
| 30 | 70 | 83,240 | 0.8227 | 0.9156 | 0.8226 | 0.8666 | 0.8230 | 0.6653 |
| 20 | 80 | 72,835 | 0.8232 | 0.9497 | 0.8226 | 0.8816 | 0.8256 | 0.5378 |
| 10 | 90 | 64,740 | 0.8222 | 0.9761 | 0.8226 | 0.8928 | 0.8187 | 0.3389 |
| 0 | 100 | 58,270 | 0.8226 | 1.0000 | 0.8226 | 0.9027 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8269 | 0.0000 | 0.0000 | 0.0000 | 0.8269 | 1.0000 |
| 90 | 10 | 299,940 | 0.8252 | 0.3435 | 0.8212 | 0.4844 | 0.8256 | 0.9765 |
| 80 | 20 | 291,350 | 0.8247 | 0.5406 | 0.8226 | 0.6525 | 0.8253 | 0.9490 |
| 70 | 30 | 194,230 | 0.8254 | 0.6703 | 0.8226 | 0.7387 | 0.8266 | 0.9158 |
| 60 | 40 | 145,675 | 0.8246 | 0.7590 | 0.8226 | 0.7895 | 0.8259 | 0.8747 |
| 50 | 50 | 116,540 | 0.8234 | 0.8239 | 0.8226 | 0.8232 | 0.8242 | 0.8229 |
| 40 | 60 | 97,115 | 0.8241 | 0.8766 | 0.8226 | 0.8487 | 0.8263 | 0.7564 |
| 30 | 70 | 83,240 | 0.8227 | 0.9156 | 0.8226 | 0.8666 | 0.8230 | 0.6653 |
| 20 | 80 | 72,835 | 0.8232 | 0.9497 | 0.8226 | 0.8816 | 0.8256 | 0.5378 |
| 10 | 90 | 64,740 | 0.8222 | 0.9761 | 0.8226 | 0.8928 | 0.8187 | 0.3389 |
| 0 | 100 | 58,270 | 0.8226 | 1.0000 | 0.8226 | 0.9027 | 0.0000 | 0.0000 |


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
0.15       0.9222   0.7147   0.9163   0.9970   0.9750   0.5641   <--
0.20       0.9222   0.7147   0.9163   0.9970   0.9750   0.5641  
0.25       0.9222   0.7147   0.9163   0.9970   0.9750   0.5641  
0.30       0.9222   0.7147   0.9163   0.9970   0.9750   0.5641  
0.35       0.9222   0.7147   0.9163   0.9970   0.9750   0.5641  
0.40       0.9222   0.7147   0.9163   0.9970   0.9750   0.5641  
0.45       0.9222   0.7147   0.9163   0.9970   0.9750   0.5641  
0.50       0.9222   0.7147   0.9163   0.9970   0.9750   0.5641  
0.55       0.9222   0.7147   0.9163   0.9970   0.9750   0.5641  
0.60       0.9222   0.7147   0.9163   0.9970   0.9750   0.5641  
0.65       0.9222   0.7147   0.9163   0.9970   0.9750   0.5641  
0.70       0.9222   0.7147   0.9163   0.9970   0.9750   0.5641  
0.75       0.9222   0.7147   0.9163   0.9970   0.9750   0.5641  
0.80       0.9222   0.7147   0.9163   0.9970   0.9750   0.5641  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9222, F1=0.7147, Normal Recall=0.9163, Normal Precision=0.9970, Attack Recall=0.9750, Attack Precision=0.5641

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
0.15       0.9282   0.8444   0.9168   0.9930   0.9740   0.7453   <--
0.20       0.9282   0.8444   0.9168   0.9930   0.9740   0.7453  
0.25       0.9282   0.8444   0.9168   0.9930   0.9740   0.7453  
0.30       0.9282   0.8444   0.9168   0.9930   0.9740   0.7453  
0.35       0.9282   0.8444   0.9168   0.9930   0.9740   0.7453  
0.40       0.9282   0.8444   0.9168   0.9930   0.9740   0.7453  
0.45       0.9282   0.8444   0.9168   0.9930   0.9740   0.7453  
0.50       0.9282   0.8444   0.9168   0.9930   0.9740   0.7453  
0.55       0.9282   0.8444   0.9168   0.9930   0.9740   0.7453  
0.60       0.9282   0.8444   0.9168   0.9930   0.9740   0.7453  
0.65       0.9282   0.8444   0.9168   0.9930   0.9740   0.7453  
0.70       0.9282   0.8444   0.9168   0.9930   0.9740   0.7453  
0.75       0.9282   0.8444   0.9168   0.9930   0.9740   0.7453  
0.80       0.9282   0.8444   0.9168   0.9930   0.9740   0.7453  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9282, F1=0.8444, Normal Recall=0.9168, Normal Precision=0.9930, Attack Recall=0.9740, Attack Precision=0.7453

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
0.15       0.9337   0.8981   0.9164   0.9880   0.9740   0.8331   <--
0.20       0.9337   0.8981   0.9164   0.9880   0.9740   0.8331  
0.25       0.9337   0.8981   0.9164   0.9880   0.9740   0.8331  
0.30       0.9337   0.8981   0.9164   0.9880   0.9740   0.8331  
0.35       0.9337   0.8981   0.9164   0.9880   0.9740   0.8331  
0.40       0.9337   0.8981   0.9164   0.9880   0.9740   0.8331  
0.45       0.9337   0.8981   0.9164   0.9880   0.9740   0.8331  
0.50       0.9337   0.8981   0.9164   0.9880   0.9740   0.8331  
0.55       0.9337   0.8981   0.9164   0.9880   0.9740   0.8331  
0.60       0.9337   0.8981   0.9164   0.9880   0.9740   0.8331  
0.65       0.9337   0.8981   0.9164   0.9880   0.9740   0.8331  
0.70       0.9337   0.8981   0.9164   0.9880   0.9740   0.8331  
0.75       0.9337   0.8981   0.9164   0.9880   0.9740   0.8331  
0.80       0.9337   0.8981   0.9164   0.9880   0.9740   0.8331  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9337, F1=0.8981, Normal Recall=0.9164, Normal Precision=0.9880, Attack Recall=0.9740, Attack Precision=0.8331

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
0.15       0.9394   0.9278   0.9163   0.9814   0.9740   0.8858   <--
0.20       0.9394   0.9278   0.9163   0.9814   0.9740   0.8858  
0.25       0.9394   0.9278   0.9163   0.9814   0.9740   0.8858  
0.30       0.9394   0.9278   0.9163   0.9814   0.9740   0.8858  
0.35       0.9394   0.9278   0.9163   0.9814   0.9740   0.8858  
0.40       0.9394   0.9278   0.9163   0.9814   0.9740   0.8858  
0.45       0.9394   0.9278   0.9163   0.9814   0.9740   0.8858  
0.50       0.9394   0.9278   0.9163   0.9814   0.9740   0.8858  
0.55       0.9394   0.9278   0.9163   0.9814   0.9740   0.8858  
0.60       0.9394   0.9278   0.9163   0.9814   0.9740   0.8858  
0.65       0.9394   0.9278   0.9163   0.9814   0.9740   0.8858  
0.70       0.9394   0.9278   0.9163   0.9814   0.9740   0.8858  
0.75       0.9394   0.9278   0.9163   0.9814   0.9740   0.8858  
0.80       0.9394   0.9278   0.9163   0.9814   0.9740   0.8858  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9394, F1=0.9278, Normal Recall=0.9163, Normal Precision=0.9814, Attack Recall=0.9740, Attack Precision=0.8858

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
0.15       0.9449   0.9465   0.9159   0.9724   0.9740   0.9205   <--
0.20       0.9449   0.9465   0.9159   0.9724   0.9740   0.9205  
0.25       0.9449   0.9465   0.9159   0.9724   0.9740   0.9205  
0.30       0.9449   0.9465   0.9159   0.9724   0.9740   0.9205  
0.35       0.9449   0.9465   0.9159   0.9724   0.9740   0.9205  
0.40       0.9449   0.9465   0.9159   0.9724   0.9740   0.9205  
0.45       0.9449   0.9465   0.9159   0.9724   0.9740   0.9205  
0.50       0.9449   0.9465   0.9159   0.9724   0.9740   0.9205  
0.55       0.9449   0.9465   0.9159   0.9724   0.9740   0.9205  
0.60       0.9449   0.9465   0.9159   0.9724   0.9740   0.9205  
0.65       0.9449   0.9465   0.9159   0.9724   0.9740   0.9205  
0.70       0.9449   0.9465   0.9159   0.9724   0.9740   0.9205  
0.75       0.9449   0.9465   0.9159   0.9724   0.9740   0.9205  
0.80       0.9449   0.9465   0.9159   0.9724   0.9740   0.9205  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9449, F1=0.9465, Normal Recall=0.9159, Normal Precision=0.9724, Attack Recall=0.9740, Attack Precision=0.9205

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
0.15       0.8269   0.4880   0.8271   0.9770   0.8251   0.3465   <--
0.20       0.8269   0.4880   0.8271   0.9770   0.8251   0.3465  
0.25       0.8269   0.4880   0.8271   0.9770   0.8251   0.3465  
0.30       0.8269   0.4880   0.8271   0.9770   0.8251   0.3465  
0.35       0.8269   0.4880   0.8271   0.9770   0.8251   0.3465  
0.40       0.8269   0.4880   0.8271   0.9770   0.8251   0.3465  
0.45       0.8269   0.4880   0.8271   0.9770   0.8251   0.3465  
0.50       0.8269   0.4880   0.8271   0.9770   0.8251   0.3465  
0.55       0.8269   0.4880   0.8271   0.9770   0.8251   0.3465  
0.60       0.8269   0.4880   0.8271   0.9770   0.8251   0.3465  
0.65       0.8269   0.4880   0.8271   0.9770   0.8251   0.3465  
0.70       0.8269   0.4880   0.8271   0.9770   0.8251   0.3465  
0.75       0.8269   0.4880   0.8271   0.9770   0.8251   0.3465  
0.80       0.8269   0.4880   0.8271   0.9770   0.8251   0.3465  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8269, F1=0.4880, Normal Recall=0.8271, Normal Precision=0.9770, Attack Recall=0.8251, Attack Precision=0.3465

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
0.15       0.8266   0.6548   0.8276   0.9491   0.8224   0.5440   <--
0.20       0.8266   0.6548   0.8276   0.9491   0.8224   0.5440  
0.25       0.8266   0.6548   0.8276   0.9491   0.8224   0.5440  
0.30       0.8266   0.6548   0.8276   0.9491   0.8224   0.5440  
0.35       0.8266   0.6548   0.8276   0.9491   0.8224   0.5440  
0.40       0.8266   0.6548   0.8276   0.9491   0.8224   0.5440  
0.45       0.8266   0.6548   0.8276   0.9491   0.8224   0.5440  
0.50       0.8266   0.6548   0.8276   0.9491   0.8224   0.5440  
0.55       0.8266   0.6548   0.8276   0.9491   0.8224   0.5440  
0.60       0.8266   0.6548   0.8276   0.9491   0.8224   0.5440  
0.65       0.8266   0.6548   0.8276   0.9491   0.8224   0.5440  
0.70       0.8266   0.6548   0.8276   0.9491   0.8224   0.5440  
0.75       0.8266   0.6548   0.8276   0.9491   0.8224   0.5440  
0.80       0.8266   0.6548   0.8276   0.9491   0.8224   0.5440  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8266, F1=0.6548, Normal Recall=0.8276, Normal Precision=0.9491, Attack Recall=0.8224, Attack Precision=0.5440

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
0.15       0.8262   0.7395   0.8278   0.9158   0.8224   0.6718   <--
0.20       0.8262   0.7395   0.8278   0.9158   0.8224   0.6718  
0.25       0.8262   0.7395   0.8278   0.9158   0.8224   0.6718  
0.30       0.8262   0.7395   0.8278   0.9158   0.8224   0.6718  
0.35       0.8262   0.7395   0.8278   0.9158   0.8224   0.6718  
0.40       0.8262   0.7395   0.8278   0.9158   0.8224   0.6718  
0.45       0.8262   0.7395   0.8278   0.9158   0.8224   0.6718  
0.50       0.8262   0.7395   0.8278   0.9158   0.8224   0.6718  
0.55       0.8262   0.7395   0.8278   0.9158   0.8224   0.6718  
0.60       0.8262   0.7395   0.8278   0.9158   0.8224   0.6718  
0.65       0.8262   0.7395   0.8278   0.9158   0.8224   0.6718  
0.70       0.8262   0.7395   0.8278   0.9158   0.8224   0.6718  
0.75       0.8262   0.7395   0.8278   0.9158   0.8224   0.6718  
0.80       0.8262   0.7395   0.8278   0.9158   0.8224   0.6718  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8262, F1=0.7395, Normal Recall=0.8278, Normal Precision=0.9158, Attack Recall=0.8224, Attack Precision=0.6718

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
0.15       0.8259   0.7908   0.8283   0.8749   0.8224   0.7615   <--
0.20       0.8259   0.7908   0.8283   0.8749   0.8224   0.7615  
0.25       0.8259   0.7908   0.8283   0.8749   0.8224   0.7615  
0.30       0.8259   0.7908   0.8283   0.8749   0.8224   0.7615  
0.35       0.8259   0.7908   0.8283   0.8749   0.8224   0.7615  
0.40       0.8259   0.7908   0.8283   0.8749   0.8224   0.7615  
0.45       0.8259   0.7908   0.8283   0.8749   0.8224   0.7615  
0.50       0.8259   0.7908   0.8283   0.8749   0.8224   0.7615  
0.55       0.8259   0.7908   0.8283   0.8749   0.8224   0.7615  
0.60       0.8259   0.7908   0.8283   0.8749   0.8224   0.7615  
0.65       0.8259   0.7908   0.8283   0.8749   0.8224   0.7615  
0.70       0.8259   0.7908   0.8283   0.8749   0.8224   0.7615  
0.75       0.8259   0.7908   0.8283   0.8749   0.8224   0.7615  
0.80       0.8259   0.7908   0.8283   0.8749   0.8224   0.7615  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8259, F1=0.7908, Normal Recall=0.8283, Normal Precision=0.8749, Attack Recall=0.8224, Attack Precision=0.7615

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
0.15       0.8245   0.8242   0.8267   0.8232   0.8224   0.8259   <--
0.20       0.8245   0.8242   0.8267   0.8232   0.8224   0.8259  
0.25       0.8245   0.8242   0.8267   0.8232   0.8224   0.8259  
0.30       0.8245   0.8242   0.8267   0.8232   0.8224   0.8259  
0.35       0.8245   0.8242   0.8267   0.8232   0.8224   0.8259  
0.40       0.8245   0.8242   0.8267   0.8232   0.8224   0.8259  
0.45       0.8245   0.8242   0.8267   0.8232   0.8224   0.8259  
0.50       0.8245   0.8242   0.8267   0.8232   0.8224   0.8259  
0.55       0.8245   0.8242   0.8267   0.8232   0.8224   0.8259  
0.60       0.8245   0.8242   0.8267   0.8232   0.8224   0.8259  
0.65       0.8245   0.8242   0.8267   0.8232   0.8224   0.8259  
0.70       0.8245   0.8242   0.8267   0.8232   0.8224   0.8259  
0.75       0.8245   0.8242   0.8267   0.8232   0.8224   0.8259  
0.80       0.8245   0.8242   0.8267   0.8232   0.8224   0.8259  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8245, F1=0.8242, Normal Recall=0.8267, Normal Precision=0.8232, Attack Recall=0.8224, Attack Precision=0.8259

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
0.15       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447   <--
0.20       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.25       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.30       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.35       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.40       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.45       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.50       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.55       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.60       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.65       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.70       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.75       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.80       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8256, F1=0.4863, Normal Recall=0.8256, Normal Precision=0.9770, Attack Recall=0.8254, Attack Precision=0.3447

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
0.15       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419   <--
0.20       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.25       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.30       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.35       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.40       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.45       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.50       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.55       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.60       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.65       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.70       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.75       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.80       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8255, F1=0.6534, Normal Recall=0.8262, Normal Precision=0.9490, Attack Recall=0.8226, Attack Precision=0.5419

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
0.15       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700   <--
0.20       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.25       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.30       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.35       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.40       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.45       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.50       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.55       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.60       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.65       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.70       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.75       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.80       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8252, F1=0.7385, Normal Recall=0.8264, Normal Precision=0.9157, Attack Recall=0.8226, Attack Precision=0.6700

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
0.15       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599   <--
0.20       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.25       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.30       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.35       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.40       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.45       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.50       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.55       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.60       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.65       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.70       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.75       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.80       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8251, F1=0.7900, Normal Recall=0.8267, Normal Precision=0.8748, Attack Recall=0.8226, Attack Precision=0.7599

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
0.15       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247   <--
0.20       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.25       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.30       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.35       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.40       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.45       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.50       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.55       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.60       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.65       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.70       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.75       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.80       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8239, F1=0.8237, Normal Recall=0.8252, Normal Precision=0.8230, Attack Recall=0.8226, Attack Precision=0.8247

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
0.15       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447   <--
0.20       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.25       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.30       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.35       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.40       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.45       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.50       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.55       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.60       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.65       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.70       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.75       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
0.80       0.8256   0.4863   0.8256   0.9770   0.8254   0.3447  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8256, F1=0.4863, Normal Recall=0.8256, Normal Precision=0.9770, Attack Recall=0.8254, Attack Precision=0.3447

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
0.15       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419   <--
0.20       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.25       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.30       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.35       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.40       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.45       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.50       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.55       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.60       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.65       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.70       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.75       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
0.80       0.8255   0.6534   0.8262   0.9490   0.8226   0.5419  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8255, F1=0.6534, Normal Recall=0.8262, Normal Precision=0.9490, Attack Recall=0.8226, Attack Precision=0.5419

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
0.15       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700   <--
0.20       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.25       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.30       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.35       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.40       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.45       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.50       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.55       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.60       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.65       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.70       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.75       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
0.80       0.8252   0.7385   0.8264   0.9157   0.8226   0.6700  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8252, F1=0.7385, Normal Recall=0.8264, Normal Precision=0.9157, Attack Recall=0.8226, Attack Precision=0.6700

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
0.15       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599   <--
0.20       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.25       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.30       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.35       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.40       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.45       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.50       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.55       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.60       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.65       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.70       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.75       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
0.80       0.8251   0.7900   0.8267   0.8748   0.8226   0.7599  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8251, F1=0.7900, Normal Recall=0.8267, Normal Precision=0.8748, Attack Recall=0.8226, Attack Precision=0.7599

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
0.15       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247   <--
0.20       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.25       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.30       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.35       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.40       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.45       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.50       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.55       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.60       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.65       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.70       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.75       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
0.80       0.8239   0.8237   0.8252   0.8230   0.8226   0.8247  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8239, F1=0.8237, Normal Recall=0.8252, Normal Precision=0.8230, Attack Recall=0.8226, Attack Precision=0.8247

```

