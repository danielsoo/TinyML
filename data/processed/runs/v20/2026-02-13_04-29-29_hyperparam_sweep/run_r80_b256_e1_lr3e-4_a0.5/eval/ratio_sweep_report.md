# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e1_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-18 07:12:05 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8538 | 0.8680 | 0.8810 | 0.8953 | 0.9087 | 0.9208 | 0.9357 | 0.9492 | 0.9628 | 0.9756 | 0.9895 |
| QAT+Prune only | 0.7251 | 0.7394 | 0.7535 | 0.7670 | 0.7811 | 0.7951 | 0.8102 | 0.8232 | 0.8394 | 0.8515 | 0.8668 |
| QAT+PTQ | 0.7259 | 0.7400 | 0.7538 | 0.7670 | 0.7810 | 0.7946 | 0.8095 | 0.8223 | 0.8383 | 0.8502 | 0.8651 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7259 | 0.7400 | 0.7538 | 0.7670 | 0.7810 | 0.7946 | 0.8095 | 0.8223 | 0.8383 | 0.8502 | 0.8651 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.5999 | 0.7688 | 0.8501 | 0.8966 | 0.9259 | 0.9486 | 0.9646 | 0.9770 | 0.9865 | 0.9947 |
| QAT+Prune only | 0.0000 | 0.3991 | 0.5845 | 0.6906 | 0.7601 | 0.8088 | 0.8457 | 0.8728 | 0.8962 | 0.9131 | 0.9286 |
| QAT+PTQ | 0.0000 | 0.3992 | 0.5843 | 0.6902 | 0.7596 | 0.8082 | 0.8450 | 0.8721 | 0.8954 | 0.9122 | 0.9277 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3992 | 0.5843 | 0.6902 | 0.7596 | 0.8082 | 0.8450 | 0.8721 | 0.8954 | 0.9122 | 0.9277 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8538 | 0.8545 | 0.8538 | 0.8549 | 0.8549 | 0.8522 | 0.8551 | 0.8553 | 0.8561 | 0.8511 | 0.0000 |
| QAT+Prune only | 0.7251 | 0.7255 | 0.7252 | 0.7242 | 0.7241 | 0.7234 | 0.7254 | 0.7215 | 0.7299 | 0.7142 | 0.0000 |
| QAT+PTQ | 0.7259 | 0.7263 | 0.7260 | 0.7250 | 0.7249 | 0.7241 | 0.7260 | 0.7223 | 0.7310 | 0.7155 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7259 | 0.7263 | 0.7260 | 0.7250 | 0.7249 | 0.7241 | 0.7260 | 0.7223 | 0.7310 | 0.7155 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8538 | 0.0000 | 0.0000 | 0.0000 | 0.8538 | 1.0000 |
| 90 | 10 | 299,940 | 0.8680 | 0.4304 | 0.9895 | 0.5999 | 0.8545 | 0.9986 |
| 80 | 20 | 291,350 | 0.8810 | 0.6286 | 0.9895 | 0.7688 | 0.8538 | 0.9969 |
| 70 | 30 | 194,230 | 0.8953 | 0.7451 | 0.9895 | 0.8501 | 0.8549 | 0.9948 |
| 60 | 40 | 145,675 | 0.9087 | 0.8197 | 0.9895 | 0.8966 | 0.8549 | 0.9919 |
| 50 | 50 | 116,540 | 0.9208 | 0.8700 | 0.9895 | 0.9259 | 0.8522 | 0.9878 |
| 40 | 60 | 97,115 | 0.9357 | 0.9110 | 0.9895 | 0.9486 | 0.8551 | 0.9819 |
| 30 | 70 | 83,240 | 0.9492 | 0.9410 | 0.9895 | 0.9646 | 0.8553 | 0.9721 |
| 20 | 80 | 72,835 | 0.9628 | 0.9649 | 0.9895 | 0.9770 | 0.8561 | 0.9531 |
| 10 | 90 | 64,740 | 0.9756 | 0.9836 | 0.9895 | 0.9865 | 0.8511 | 0.8999 |
| 0 | 100 | 58,270 | 0.9895 | 1.0000 | 0.9895 | 0.9947 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7251 | 0.0000 | 0.0000 | 0.0000 | 0.7251 | 1.0000 |
| 90 | 10 | 299,940 | 0.7394 | 0.2593 | 0.8652 | 0.3991 | 0.7255 | 0.9798 |
| 80 | 20 | 291,350 | 0.7535 | 0.4409 | 0.8668 | 0.5845 | 0.7252 | 0.9561 |
| 70 | 30 | 194,230 | 0.7670 | 0.5739 | 0.8668 | 0.6906 | 0.7242 | 0.9269 |
| 60 | 40 | 145,675 | 0.7811 | 0.6768 | 0.8668 | 0.7601 | 0.7241 | 0.8907 |
| 50 | 50 | 116,540 | 0.7951 | 0.7581 | 0.8668 | 0.8088 | 0.7234 | 0.8445 |
| 40 | 60 | 97,115 | 0.8102 | 0.8256 | 0.8668 | 0.8457 | 0.7254 | 0.7840 |
| 30 | 70 | 83,240 | 0.8232 | 0.8790 | 0.8668 | 0.8728 | 0.7215 | 0.6989 |
| 20 | 80 | 72,835 | 0.8394 | 0.9277 | 0.8668 | 0.8962 | 0.7299 | 0.5781 |
| 10 | 90 | 64,740 | 0.8515 | 0.9647 | 0.8668 | 0.9131 | 0.7142 | 0.3733 |
| 0 | 100 | 58,270 | 0.8668 | 1.0000 | 0.8668 | 0.9286 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7259 | 0.0000 | 0.0000 | 0.0000 | 0.7259 | 1.0000 |
| 90 | 10 | 299,940 | 0.7400 | 0.2596 | 0.8638 | 0.3992 | 0.7263 | 0.9796 |
| 80 | 20 | 291,350 | 0.7538 | 0.4412 | 0.8651 | 0.5843 | 0.7260 | 0.9556 |
| 70 | 30 | 194,230 | 0.7670 | 0.5741 | 0.8651 | 0.6902 | 0.7250 | 0.9262 |
| 60 | 40 | 145,675 | 0.7810 | 0.6771 | 0.8651 | 0.7596 | 0.7249 | 0.8897 |
| 50 | 50 | 116,540 | 0.7946 | 0.7582 | 0.8651 | 0.8082 | 0.7241 | 0.8430 |
| 40 | 60 | 97,115 | 0.8095 | 0.8257 | 0.8651 | 0.8450 | 0.7260 | 0.7821 |
| 30 | 70 | 83,240 | 0.8223 | 0.8791 | 0.8651 | 0.8721 | 0.7223 | 0.6966 |
| 20 | 80 | 72,835 | 0.8383 | 0.9279 | 0.8652 | 0.8954 | 0.7310 | 0.5754 |
| 10 | 90 | 64,740 | 0.8502 | 0.9647 | 0.8651 | 0.9122 | 0.7155 | 0.3709 |
| 0 | 100 | 58,270 | 0.8651 | 1.0000 | 0.8651 | 0.9277 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7259 | 0.0000 | 0.0000 | 0.0000 | 0.7259 | 1.0000 |
| 90 | 10 | 299,940 | 0.7400 | 0.2596 | 0.8638 | 0.3992 | 0.7263 | 0.9796 |
| 80 | 20 | 291,350 | 0.7538 | 0.4412 | 0.8651 | 0.5843 | 0.7260 | 0.9556 |
| 70 | 30 | 194,230 | 0.7670 | 0.5741 | 0.8651 | 0.6902 | 0.7250 | 0.9262 |
| 60 | 40 | 145,675 | 0.7810 | 0.6771 | 0.8651 | 0.7596 | 0.7249 | 0.8897 |
| 50 | 50 | 116,540 | 0.7946 | 0.7582 | 0.8651 | 0.8082 | 0.7241 | 0.8430 |
| 40 | 60 | 97,115 | 0.8095 | 0.8257 | 0.8651 | 0.8450 | 0.7260 | 0.7821 |
| 30 | 70 | 83,240 | 0.8223 | 0.8791 | 0.8651 | 0.8721 | 0.7223 | 0.6966 |
| 20 | 80 | 72,835 | 0.8383 | 0.9279 | 0.8652 | 0.8954 | 0.7310 | 0.5754 |
| 10 | 90 | 64,740 | 0.8502 | 0.9647 | 0.8651 | 0.9122 | 0.7155 | 0.3709 |
| 0 | 100 | 58,270 | 0.8651 | 1.0000 | 0.8651 | 0.9277 | 0.0000 | 0.0000 |


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
0.15       0.8681   0.6001   0.8545   0.9987   0.9901   0.4305   <--
0.20       0.8681   0.6001   0.8545   0.9987   0.9901   0.4305  
0.25       0.8681   0.6001   0.8545   0.9987   0.9901   0.4305  
0.30       0.8681   0.6001   0.8545   0.9987   0.9901   0.4305  
0.35       0.8681   0.6001   0.8545   0.9987   0.9901   0.4305  
0.40       0.8681   0.6001   0.8545   0.9987   0.9901   0.4305  
0.45       0.8681   0.6001   0.8545   0.9987   0.9901   0.4305  
0.50       0.8681   0.6001   0.8545   0.9987   0.9901   0.4305  
0.55       0.8681   0.6001   0.8545   0.9987   0.9901   0.4305  
0.60       0.8681   0.6001   0.8545   0.9987   0.9901   0.4305  
0.65       0.8681   0.6001   0.8545   0.9987   0.9901   0.4305  
0.70       0.8681   0.6001   0.8545   0.9987   0.9901   0.4305  
0.75       0.8681   0.6001   0.8545   0.9987   0.9901   0.4305  
0.80       0.8681   0.6001   0.8545   0.9987   0.9901   0.4305  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8681, F1=0.6001, Normal Recall=0.8545, Normal Precision=0.9987, Attack Recall=0.9901, Attack Precision=0.4305

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
0.15       0.8818   0.7701   0.8549   0.9969   0.9895   0.6303   <--
0.20       0.8818   0.7701   0.8549   0.9969   0.9895   0.6303  
0.25       0.8818   0.7701   0.8549   0.9969   0.9895   0.6303  
0.30       0.8818   0.7701   0.8549   0.9969   0.9895   0.6303  
0.35       0.8818   0.7701   0.8549   0.9969   0.9895   0.6303  
0.40       0.8818   0.7701   0.8549   0.9969   0.9895   0.6303  
0.45       0.8818   0.7701   0.8549   0.9969   0.9895   0.6303  
0.50       0.8818   0.7701   0.8549   0.9969   0.9895   0.6303  
0.55       0.8818   0.7701   0.8549   0.9969   0.9895   0.6303  
0.60       0.8818   0.7701   0.8549   0.9969   0.9895   0.6303  
0.65       0.8818   0.7701   0.8549   0.9969   0.9895   0.6303  
0.70       0.8818   0.7701   0.8549   0.9969   0.9895   0.6303  
0.75       0.8818   0.7701   0.8549   0.9969   0.9895   0.6303  
0.80       0.8818   0.7701   0.8549   0.9969   0.9895   0.6303  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8818, F1=0.7701, Normal Recall=0.8549, Normal Precision=0.9969, Attack Recall=0.9895, Attack Precision=0.6303

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
0.15       0.8948   0.8495   0.8543   0.9947   0.9895   0.7442   <--
0.20       0.8948   0.8495   0.8543   0.9947   0.9895   0.7442  
0.25       0.8948   0.8495   0.8543   0.9947   0.9895   0.7442  
0.30       0.8948   0.8495   0.8543   0.9947   0.9895   0.7442  
0.35       0.8948   0.8495   0.8543   0.9947   0.9895   0.7442  
0.40       0.8948   0.8495   0.8543   0.9947   0.9895   0.7442  
0.45       0.8948   0.8495   0.8543   0.9947   0.9895   0.7442  
0.50       0.8948   0.8495   0.8543   0.9947   0.9895   0.7442  
0.55       0.8948   0.8495   0.8543   0.9947   0.9895   0.7442  
0.60       0.8948   0.8495   0.8543   0.9947   0.9895   0.7442  
0.65       0.8948   0.8495   0.8543   0.9947   0.9895   0.7442  
0.70       0.8948   0.8495   0.8543   0.9947   0.9895   0.7442  
0.75       0.8948   0.8495   0.8543   0.9947   0.9895   0.7442  
0.80       0.8948   0.8495   0.8543   0.9947   0.9895   0.7442  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8948, F1=0.8495, Normal Recall=0.8543, Normal Precision=0.9947, Attack Recall=0.9895, Attack Precision=0.7442

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
0.15       0.9081   0.8960   0.8539   0.9919   0.9895   0.8186   <--
0.20       0.9081   0.8960   0.8539   0.9919   0.9895   0.8186  
0.25       0.9081   0.8960   0.8539   0.9919   0.9895   0.8186  
0.30       0.9081   0.8960   0.8539   0.9919   0.9895   0.8186  
0.35       0.9081   0.8960   0.8539   0.9919   0.9895   0.8186  
0.40       0.9081   0.8960   0.8539   0.9919   0.9895   0.8186  
0.45       0.9081   0.8960   0.8539   0.9919   0.9895   0.8186  
0.50       0.9081   0.8960   0.8539   0.9919   0.9895   0.8186  
0.55       0.9081   0.8960   0.8539   0.9919   0.9895   0.8186  
0.60       0.9081   0.8960   0.8539   0.9919   0.9895   0.8186  
0.65       0.9081   0.8960   0.8539   0.9919   0.9895   0.8186  
0.70       0.9081   0.8960   0.8539   0.9919   0.9895   0.8186  
0.75       0.9081   0.8960   0.8539   0.9919   0.9895   0.8186  
0.80       0.9081   0.8960   0.8539   0.9919   0.9895   0.8186  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9081, F1=0.8960, Normal Recall=0.8539, Normal Precision=0.9919, Attack Recall=0.9895, Attack Precision=0.8186

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
0.15       0.9215   0.9265   0.8535   0.9878   0.9895   0.8711   <--
0.20       0.9215   0.9265   0.8535   0.9878   0.9895   0.8711  
0.25       0.9215   0.9265   0.8535   0.9878   0.9895   0.8711  
0.30       0.9215   0.9265   0.8535   0.9878   0.9895   0.8711  
0.35       0.9215   0.9265   0.8535   0.9878   0.9895   0.8711  
0.40       0.9215   0.9265   0.8535   0.9878   0.9895   0.8711  
0.45       0.9215   0.9265   0.8535   0.9878   0.9895   0.8711  
0.50       0.9215   0.9265   0.8535   0.9878   0.9895   0.8711  
0.55       0.9215   0.9265   0.8535   0.9878   0.9895   0.8711  
0.60       0.9215   0.9265   0.8535   0.9878   0.9895   0.8711  
0.65       0.9215   0.9265   0.8535   0.9878   0.9895   0.8711  
0.70       0.9215   0.9265   0.8535   0.9878   0.9895   0.8711  
0.75       0.9215   0.9265   0.8535   0.9878   0.9895   0.8711  
0.80       0.9215   0.9265   0.8535   0.9878   0.9895   0.8711  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9215, F1=0.9265, Normal Recall=0.8535, Normal Precision=0.9878, Attack Recall=0.9895, Attack Precision=0.8711

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
0.15       0.7397   0.4000   0.7255   0.9801   0.8677   0.2599   <--
0.20       0.7397   0.4000   0.7255   0.9801   0.8677   0.2599  
0.25       0.7397   0.4000   0.7255   0.9801   0.8677   0.2599  
0.30       0.7397   0.4000   0.7255   0.9801   0.8677   0.2599  
0.35       0.7397   0.4000   0.7255   0.9801   0.8677   0.2599  
0.40       0.7397   0.4000   0.7255   0.9801   0.8677   0.2599  
0.45       0.7397   0.4000   0.7255   0.9801   0.8677   0.2599  
0.50       0.7397   0.4000   0.7255   0.9801   0.8677   0.2599  
0.55       0.7397   0.4000   0.7255   0.9801   0.8677   0.2599  
0.60       0.7397   0.4000   0.7255   0.9801   0.8677   0.2599  
0.65       0.7397   0.4000   0.7255   0.9801   0.8677   0.2599  
0.70       0.7397   0.4000   0.7255   0.9801   0.8677   0.2599  
0.75       0.7397   0.4000   0.7255   0.9801   0.8677   0.2599  
0.80       0.7397   0.4000   0.7255   0.9801   0.8677   0.2599  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7397, F1=0.4000, Normal Recall=0.7255, Normal Precision=0.9801, Attack Recall=0.8677, Attack Precision=0.2599

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
0.15       0.7540   0.5849   0.7258   0.9561   0.8668   0.4414   <--
0.20       0.7540   0.5849   0.7258   0.9561   0.8668   0.4414  
0.25       0.7540   0.5849   0.7258   0.9561   0.8668   0.4414  
0.30       0.7540   0.5849   0.7258   0.9561   0.8668   0.4414  
0.35       0.7540   0.5849   0.7258   0.9561   0.8668   0.4414  
0.40       0.7540   0.5849   0.7258   0.9561   0.8668   0.4414  
0.45       0.7540   0.5849   0.7258   0.9561   0.8668   0.4414  
0.50       0.7540   0.5849   0.7258   0.9561   0.8668   0.4414  
0.55       0.7540   0.5849   0.7258   0.9561   0.8668   0.4414  
0.60       0.7540   0.5849   0.7258   0.9561   0.8668   0.4414  
0.65       0.7540   0.5849   0.7258   0.9561   0.8668   0.4414  
0.70       0.7540   0.5849   0.7258   0.9561   0.8668   0.4414  
0.75       0.7540   0.5849   0.7258   0.9561   0.8668   0.4414  
0.80       0.7540   0.5849   0.7258   0.9561   0.8668   0.4414  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7540, F1=0.5849, Normal Recall=0.7258, Normal Precision=0.9561, Attack Recall=0.8668, Attack Precision=0.4414

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
0.15       0.7677   0.6912   0.7252   0.9270   0.8668   0.5748   <--
0.20       0.7677   0.6912   0.7252   0.9270   0.8668   0.5748  
0.25       0.7677   0.6912   0.7252   0.9270   0.8668   0.5748  
0.30       0.7677   0.6912   0.7252   0.9270   0.8668   0.5748  
0.35       0.7677   0.6912   0.7252   0.9270   0.8668   0.5748  
0.40       0.7677   0.6912   0.7252   0.9270   0.8668   0.5748  
0.45       0.7677   0.6912   0.7252   0.9270   0.8668   0.5748  
0.50       0.7677   0.6912   0.7252   0.9270   0.8668   0.5748  
0.55       0.7677   0.6912   0.7252   0.9270   0.8668   0.5748  
0.60       0.7677   0.6912   0.7252   0.9270   0.8668   0.5748  
0.65       0.7677   0.6912   0.7252   0.9270   0.8668   0.5748  
0.70       0.7677   0.6912   0.7252   0.9270   0.8668   0.5748  
0.75       0.7677   0.6912   0.7252   0.9270   0.8668   0.5748  
0.80       0.7677   0.6912   0.7252   0.9270   0.8668   0.5748  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7677, F1=0.6912, Normal Recall=0.7252, Normal Precision=0.9270, Attack Recall=0.8668, Attack Precision=0.5748

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
0.15       0.7818   0.7606   0.7251   0.8909   0.8668   0.6777   <--
0.20       0.7818   0.7606   0.7251   0.8909   0.8668   0.6777  
0.25       0.7818   0.7606   0.7251   0.8909   0.8668   0.6777  
0.30       0.7818   0.7606   0.7251   0.8909   0.8668   0.6777  
0.35       0.7818   0.7606   0.7251   0.8909   0.8668   0.6777  
0.40       0.7818   0.7606   0.7251   0.8909   0.8668   0.6777  
0.45       0.7818   0.7606   0.7251   0.8909   0.8668   0.6777  
0.50       0.7818   0.7606   0.7251   0.8909   0.8668   0.6777  
0.55       0.7818   0.7606   0.7251   0.8909   0.8668   0.6777  
0.60       0.7818   0.7606   0.7251   0.8909   0.8668   0.6777  
0.65       0.7818   0.7606   0.7251   0.8909   0.8668   0.6777  
0.70       0.7818   0.7606   0.7251   0.8909   0.8668   0.6777  
0.75       0.7818   0.7606   0.7251   0.8909   0.8668   0.6777  
0.80       0.7818   0.7606   0.7251   0.8909   0.8668   0.6777  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7818, F1=0.7606, Normal Recall=0.7251, Normal Precision=0.8909, Attack Recall=0.8668, Attack Precision=0.6777

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
0.15       0.7957   0.8092   0.7246   0.8447   0.8668   0.7589   <--
0.20       0.7957   0.8092   0.7246   0.8447   0.8668   0.7589  
0.25       0.7957   0.8092   0.7246   0.8447   0.8668   0.7589  
0.30       0.7957   0.8092   0.7246   0.8447   0.8668   0.7589  
0.35       0.7957   0.8092   0.7246   0.8447   0.8668   0.7589  
0.40       0.7957   0.8092   0.7246   0.8447   0.8668   0.7589  
0.45       0.7957   0.8092   0.7246   0.8447   0.8668   0.7589  
0.50       0.7957   0.8092   0.7246   0.8447   0.8668   0.7589  
0.55       0.7957   0.8092   0.7246   0.8447   0.8668   0.7589  
0.60       0.7957   0.8092   0.7246   0.8447   0.8668   0.7589  
0.65       0.7957   0.8092   0.7246   0.8447   0.8668   0.7589  
0.70       0.7957   0.8092   0.7246   0.8447   0.8668   0.7589  
0.75       0.7957   0.8092   0.7246   0.8447   0.8668   0.7589  
0.80       0.7957   0.8092   0.7246   0.8447   0.8668   0.7589  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7957, F1=0.8092, Normal Recall=0.7246, Normal Precision=0.8447, Attack Recall=0.8668, Attack Precision=0.7589

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
0.15       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601   <--
0.20       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.25       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.30       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.35       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.40       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.45       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.50       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.55       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.60       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.65       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.70       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.75       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.80       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7402, F1=0.4000, Normal Recall=0.7263, Normal Precision=0.9799, Attack Recall=0.8660, Attack Precision=0.2601

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
0.15       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416   <--
0.20       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.25       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.30       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.35       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.40       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.45       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.50       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.55       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.60       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.65       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.70       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.75       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.80       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7542, F1=0.5847, Normal Recall=0.7265, Normal Precision=0.9557, Attack Recall=0.8651, Attack Precision=0.4416

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
0.15       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749   <--
0.20       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.25       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.30       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.35       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.40       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.45       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.50       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.55       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.60       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.65       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.70       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.75       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.80       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7676, F1=0.6908, Normal Recall=0.7258, Normal Precision=0.9262, Attack Recall=0.8651, Attack Precision=0.5749

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
0.15       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778   <--
0.20       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.25       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.30       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.35       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.40       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.45       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.50       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.55       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.60       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.65       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.70       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.75       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.80       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7816, F1=0.7601, Normal Recall=0.7258, Normal Precision=0.8898, Attack Recall=0.8651, Attack Precision=0.6778

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
0.15       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589   <--
0.20       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.25       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.30       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.35       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.40       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.45       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.50       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.55       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.60       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.65       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.70       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.75       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.80       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7951, F1=0.8085, Normal Recall=0.7251, Normal Precision=0.8432, Attack Recall=0.8651, Attack Precision=0.7589

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
0.15       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601   <--
0.20       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.25       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.30       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.35       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.40       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.45       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.50       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.55       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.60       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.65       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.70       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.75       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
0.80       0.7402   0.4000   0.7263   0.9799   0.8660   0.2601  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7402, F1=0.4000, Normal Recall=0.7263, Normal Precision=0.9799, Attack Recall=0.8660, Attack Precision=0.2601

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
0.15       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416   <--
0.20       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.25       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.30       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.35       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.40       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.45       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.50       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.55       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.60       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.65       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.70       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.75       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
0.80       0.7542   0.5847   0.7265   0.9557   0.8651   0.4416  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7542, F1=0.5847, Normal Recall=0.7265, Normal Precision=0.9557, Attack Recall=0.8651, Attack Precision=0.4416

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
0.15       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749   <--
0.20       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.25       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.30       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.35       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.40       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.45       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.50       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.55       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.60       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.65       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.70       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.75       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
0.80       0.7676   0.6908   0.7258   0.9262   0.8651   0.5749  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7676, F1=0.6908, Normal Recall=0.7258, Normal Precision=0.9262, Attack Recall=0.8651, Attack Precision=0.5749

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
0.15       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778   <--
0.20       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.25       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.30       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.35       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.40       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.45       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.50       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.55       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.60       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.65       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.70       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.75       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
0.80       0.7816   0.7601   0.7258   0.8898   0.8651   0.6778  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7816, F1=0.7601, Normal Recall=0.7258, Normal Precision=0.8898, Attack Recall=0.8651, Attack Precision=0.6778

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
0.15       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589   <--
0.20       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.25       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.30       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.35       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.40       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.45       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.50       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.55       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.60       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.65       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.70       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.75       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
0.80       0.7951   0.8085   0.7251   0.8432   0.8651   0.7589  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7951, F1=0.8085, Normal Recall=0.7251, Normal Precision=0.8432, Attack Recall=0.8651, Attack Precision=0.7589

```

