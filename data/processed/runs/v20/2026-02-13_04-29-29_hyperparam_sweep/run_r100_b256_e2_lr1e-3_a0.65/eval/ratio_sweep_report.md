# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e2_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-22 22:43:05 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5669 | 0.6046 | 0.6413 | 0.6786 | 0.7139 | 0.7506 | 0.7881 | 0.8251 | 0.8616 | 0.8986 | 0.9362 |
| QAT+Prune only | 0.8495 | 0.8634 | 0.8762 | 0.8904 | 0.9031 | 0.9150 | 0.9297 | 0.9422 | 0.9552 | 0.9690 | 0.9823 |
| QAT+PTQ | 0.8493 | 0.8630 | 0.8758 | 0.8900 | 0.9027 | 0.9145 | 0.9294 | 0.9418 | 0.9550 | 0.9687 | 0.9820 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8493 | 0.8630 | 0.8758 | 0.8900 | 0.9027 | 0.9145 | 0.9294 | 0.9418 | 0.9550 | 0.9687 | 0.9820 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3212 | 0.5108 | 0.6361 | 0.7236 | 0.7897 | 0.8413 | 0.8823 | 0.9154 | 0.9432 | 0.9670 |
| QAT+Prune only | 0.0000 | 0.5900 | 0.7604 | 0.8432 | 0.8902 | 0.9203 | 0.9437 | 0.9597 | 0.9723 | 0.9827 | 0.9911 |
| QAT+PTQ | 0.0000 | 0.5892 | 0.7598 | 0.8426 | 0.8898 | 0.9199 | 0.9435 | 0.9594 | 0.9721 | 0.9826 | 0.9909 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5892 | 0.7598 | 0.8426 | 0.8898 | 0.9199 | 0.9435 | 0.9594 | 0.9721 | 0.9826 | 0.9909 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5669 | 0.5678 | 0.5676 | 0.5683 | 0.5656 | 0.5650 | 0.5660 | 0.5658 | 0.5632 | 0.5601 | 0.0000 |
| QAT+Prune only | 0.8495 | 0.8500 | 0.8497 | 0.8510 | 0.8502 | 0.8476 | 0.8509 | 0.8486 | 0.8468 | 0.8491 | 0.0000 |
| QAT+PTQ | 0.8493 | 0.8497 | 0.8493 | 0.8505 | 0.8498 | 0.8470 | 0.8506 | 0.8479 | 0.8468 | 0.8491 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8493 | 0.8497 | 0.8493 | 0.8505 | 0.8498 | 0.8470 | 0.8506 | 0.8479 | 0.8468 | 0.8491 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5669 | 0.0000 | 0.0000 | 0.0000 | 0.5669 | 1.0000 |
| 90 | 10 | 299,940 | 0.6046 | 0.1939 | 0.9355 | 0.3212 | 0.5678 | 0.9875 |
| 80 | 20 | 291,350 | 0.6413 | 0.3512 | 0.9362 | 0.5108 | 0.5676 | 0.9727 |
| 70 | 30 | 194,230 | 0.6786 | 0.4817 | 0.9362 | 0.6361 | 0.5683 | 0.9541 |
| 60 | 40 | 145,675 | 0.7139 | 0.5896 | 0.9362 | 0.7236 | 0.5656 | 0.9301 |
| 50 | 50 | 116,540 | 0.7506 | 0.6828 | 0.9362 | 0.7897 | 0.5650 | 0.8985 |
| 40 | 60 | 97,115 | 0.7881 | 0.7639 | 0.9362 | 0.8413 | 0.5660 | 0.8554 |
| 30 | 70 | 83,240 | 0.8251 | 0.8342 | 0.9362 | 0.8823 | 0.5658 | 0.7917 |
| 20 | 80 | 72,835 | 0.8616 | 0.8955 | 0.9362 | 0.9154 | 0.5632 | 0.6881 |
| 10 | 90 | 64,740 | 0.8986 | 0.9504 | 0.9362 | 0.9432 | 0.5601 | 0.4937 |
| 0 | 100 | 58,270 | 0.9362 | 1.0000 | 0.9362 | 0.9670 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8495 | 0.0000 | 0.0000 | 0.0000 | 0.8495 | 1.0000 |
| 90 | 10 | 299,940 | 0.8634 | 0.4215 | 0.9832 | 0.5900 | 0.8500 | 0.9978 |
| 80 | 20 | 291,350 | 0.8762 | 0.6203 | 0.9823 | 0.7604 | 0.8497 | 0.9948 |
| 70 | 30 | 194,230 | 0.8904 | 0.7385 | 0.9823 | 0.8432 | 0.8510 | 0.9912 |
| 60 | 40 | 145,675 | 0.9031 | 0.8139 | 0.9823 | 0.8902 | 0.8502 | 0.9863 |
| 50 | 50 | 116,540 | 0.9150 | 0.8657 | 0.9823 | 0.9203 | 0.8476 | 0.9795 |
| 40 | 60 | 97,115 | 0.9297 | 0.9081 | 0.9823 | 0.9437 | 0.8509 | 0.9697 |
| 30 | 70 | 83,240 | 0.9422 | 0.9380 | 0.9823 | 0.9597 | 0.8486 | 0.9535 |
| 20 | 80 | 72,835 | 0.9552 | 0.9625 | 0.9823 | 0.9723 | 0.8468 | 0.9228 |
| 10 | 90 | 64,740 | 0.9690 | 0.9832 | 0.9823 | 0.9827 | 0.8491 | 0.8418 |
| 0 | 100 | 58,270 | 0.9823 | 1.0000 | 0.9823 | 0.9911 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8493 | 0.0000 | 0.0000 | 0.0000 | 0.8493 | 1.0000 |
| 90 | 10 | 299,940 | 0.8630 | 0.4207 | 0.9827 | 0.5892 | 0.8497 | 0.9977 |
| 80 | 20 | 291,350 | 0.8758 | 0.6196 | 0.9820 | 0.7598 | 0.8493 | 0.9947 |
| 70 | 30 | 194,230 | 0.8900 | 0.7379 | 0.9820 | 0.8426 | 0.8505 | 0.9910 |
| 60 | 40 | 145,675 | 0.9027 | 0.8134 | 0.9820 | 0.8898 | 0.8498 | 0.9861 |
| 50 | 50 | 116,540 | 0.9145 | 0.8652 | 0.9820 | 0.9199 | 0.8470 | 0.9792 |
| 40 | 60 | 97,115 | 0.9294 | 0.9079 | 0.9820 | 0.9435 | 0.8506 | 0.9692 |
| 30 | 70 | 83,240 | 0.9418 | 0.9378 | 0.9820 | 0.9594 | 0.8479 | 0.9528 |
| 20 | 80 | 72,835 | 0.9550 | 0.9625 | 0.9820 | 0.9721 | 0.8468 | 0.9217 |
| 10 | 90 | 64,740 | 0.9687 | 0.9832 | 0.9820 | 0.9826 | 0.8491 | 0.8397 |
| 0 | 100 | 58,270 | 0.9820 | 1.0000 | 0.9820 | 0.9909 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8493 | 0.0000 | 0.0000 | 0.0000 | 0.8493 | 1.0000 |
| 90 | 10 | 299,940 | 0.8630 | 0.4207 | 0.9827 | 0.5892 | 0.8497 | 0.9977 |
| 80 | 20 | 291,350 | 0.8758 | 0.6196 | 0.9820 | 0.7598 | 0.8493 | 0.9947 |
| 70 | 30 | 194,230 | 0.8900 | 0.7379 | 0.9820 | 0.8426 | 0.8505 | 0.9910 |
| 60 | 40 | 145,675 | 0.9027 | 0.8134 | 0.9820 | 0.8898 | 0.8498 | 0.9861 |
| 50 | 50 | 116,540 | 0.9145 | 0.8652 | 0.9820 | 0.9199 | 0.8470 | 0.9792 |
| 40 | 60 | 97,115 | 0.9294 | 0.9079 | 0.9820 | 0.9435 | 0.8506 | 0.9692 |
| 30 | 70 | 83,240 | 0.9418 | 0.9378 | 0.9820 | 0.9594 | 0.8479 | 0.9528 |
| 20 | 80 | 72,835 | 0.9550 | 0.9625 | 0.9820 | 0.9721 | 0.8468 | 0.9217 |
| 10 | 90 | 64,740 | 0.9687 | 0.9832 | 0.9820 | 0.9826 | 0.8491 | 0.8397 |
| 0 | 100 | 58,270 | 0.9820 | 1.0000 | 0.9820 | 0.9909 | 0.0000 | 0.0000 |


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
0.15       0.6046   0.3214   0.5678   0.9877   0.9363   0.1940   <--
0.20       0.6046   0.3214   0.5678   0.9877   0.9363   0.1940  
0.25       0.6046   0.3214   0.5678   0.9877   0.9363   0.1940  
0.30       0.6046   0.3214   0.5678   0.9877   0.9363   0.1940  
0.35       0.6046   0.3214   0.5678   0.9877   0.9363   0.1940  
0.40       0.6046   0.3214   0.5678   0.9877   0.9363   0.1940  
0.45       0.6046   0.3214   0.5678   0.9877   0.9363   0.1940  
0.50       0.6046   0.3214   0.5678   0.9877   0.9363   0.1940  
0.55       0.6046   0.3214   0.5678   0.9877   0.9363   0.1940  
0.60       0.6046   0.3214   0.5678   0.9877   0.9363   0.1940  
0.65       0.6046   0.3214   0.5678   0.9877   0.9363   0.1940  
0.70       0.6046   0.3214   0.5678   0.9877   0.9363   0.1940  
0.75       0.6046   0.3214   0.5678   0.9877   0.9363   0.1940  
0.80       0.6046   0.3214   0.5678   0.9877   0.9363   0.1940  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6046, F1=0.3214, Normal Recall=0.5678, Normal Precision=0.9877, Attack Recall=0.9363, Attack Precision=0.1940

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
0.15       0.6416   0.5110   0.5680   0.9727   0.9362   0.3514   <--
0.20       0.6416   0.5110   0.5680   0.9727   0.9362   0.3514  
0.25       0.6416   0.5110   0.5680   0.9727   0.9362   0.3514  
0.30       0.6416   0.5110   0.5680   0.9727   0.9362   0.3514  
0.35       0.6416   0.5110   0.5680   0.9727   0.9362   0.3514  
0.40       0.6416   0.5110   0.5680   0.9727   0.9362   0.3514  
0.45       0.6416   0.5110   0.5680   0.9727   0.9362   0.3514  
0.50       0.6416   0.5110   0.5680   0.9727   0.9362   0.3514  
0.55       0.6416   0.5110   0.5680   0.9727   0.9362   0.3514  
0.60       0.6416   0.5110   0.5680   0.9727   0.9362   0.3514  
0.65       0.6416   0.5110   0.5680   0.9727   0.9362   0.3514  
0.70       0.6416   0.5110   0.5680   0.9727   0.9362   0.3514  
0.75       0.6416   0.5110   0.5680   0.9727   0.9362   0.3514  
0.80       0.6416   0.5110   0.5680   0.9727   0.9362   0.3514  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6416, F1=0.5110, Normal Recall=0.5680, Normal Precision=0.9727, Attack Recall=0.9362, Attack Precision=0.3514

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
0.15       0.6785   0.6360   0.5680   0.9541   0.9362   0.4816   <--
0.20       0.6785   0.6360   0.5680   0.9541   0.9362   0.4816  
0.25       0.6785   0.6360   0.5680   0.9541   0.9362   0.4816  
0.30       0.6785   0.6360   0.5680   0.9541   0.9362   0.4816  
0.35       0.6785   0.6360   0.5680   0.9541   0.9362   0.4816  
0.40       0.6785   0.6360   0.5680   0.9541   0.9362   0.4816  
0.45       0.6785   0.6360   0.5680   0.9541   0.9362   0.4816  
0.50       0.6785   0.6360   0.5680   0.9541   0.9362   0.4816  
0.55       0.6785   0.6360   0.5680   0.9541   0.9362   0.4816  
0.60       0.6785   0.6360   0.5680   0.9541   0.9362   0.4816  
0.65       0.6785   0.6360   0.5680   0.9541   0.9362   0.4816  
0.70       0.6785   0.6360   0.5680   0.9541   0.9362   0.4816  
0.75       0.6785   0.6360   0.5680   0.9541   0.9362   0.4816  
0.80       0.6785   0.6360   0.5680   0.9541   0.9362   0.4816  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6785, F1=0.6360, Normal Recall=0.5680, Normal Precision=0.9541, Attack Recall=0.9362, Attack Precision=0.4816

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
0.15       0.7150   0.7244   0.5675   0.9303   0.9362   0.5907   <--
0.20       0.7150   0.7244   0.5675   0.9303   0.9362   0.5907  
0.25       0.7150   0.7244   0.5675   0.9303   0.9362   0.5907  
0.30       0.7150   0.7244   0.5675   0.9303   0.9362   0.5907  
0.35       0.7150   0.7244   0.5675   0.9303   0.9362   0.5907  
0.40       0.7150   0.7244   0.5675   0.9303   0.9362   0.5907  
0.45       0.7150   0.7244   0.5675   0.9303   0.9362   0.5907  
0.50       0.7150   0.7244   0.5675   0.9303   0.9362   0.5907  
0.55       0.7150   0.7244   0.5675   0.9303   0.9362   0.5907  
0.60       0.7150   0.7244   0.5675   0.9303   0.9362   0.5907  
0.65       0.7150   0.7244   0.5675   0.9303   0.9362   0.5907  
0.70       0.7150   0.7244   0.5675   0.9303   0.9362   0.5907  
0.75       0.7150   0.7244   0.5675   0.9303   0.9362   0.5907  
0.80       0.7150   0.7244   0.5675   0.9303   0.9362   0.5907  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7150, F1=0.7244, Normal Recall=0.5675, Normal Precision=0.9303, Attack Recall=0.9362, Attack Precision=0.5907

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
0.15       0.7524   0.7908   0.5686   0.8991   0.9362   0.6845   <--
0.20       0.7524   0.7908   0.5686   0.8991   0.9362   0.6845  
0.25       0.7524   0.7908   0.5686   0.8991   0.9362   0.6845  
0.30       0.7524   0.7908   0.5686   0.8991   0.9362   0.6845  
0.35       0.7524   0.7908   0.5686   0.8991   0.9362   0.6845  
0.40       0.7524   0.7908   0.5686   0.8991   0.9362   0.6845  
0.45       0.7524   0.7908   0.5686   0.8991   0.9362   0.6845  
0.50       0.7524   0.7908   0.5686   0.8991   0.9362   0.6845  
0.55       0.7524   0.7908   0.5686   0.8991   0.9362   0.6845  
0.60       0.7524   0.7908   0.5686   0.8991   0.9362   0.6845  
0.65       0.7524   0.7908   0.5686   0.8991   0.9362   0.6845  
0.70       0.7524   0.7908   0.5686   0.8991   0.9362   0.6845  
0.75       0.7524   0.7908   0.5686   0.8991   0.9362   0.6845  
0.80       0.7524   0.7908   0.5686   0.8991   0.9362   0.6845  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7524, F1=0.7908, Normal Recall=0.5686, Normal Precision=0.8991, Attack Recall=0.9362, Attack Precision=0.6845

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
0.15       0.8633   0.5897   0.8500   0.9977   0.9825   0.4213   <--
0.20       0.8633   0.5897   0.8500   0.9977   0.9825   0.4213  
0.25       0.8633   0.5897   0.8500   0.9977   0.9825   0.4213  
0.30       0.8633   0.5897   0.8500   0.9977   0.9825   0.4213  
0.35       0.8633   0.5897   0.8500   0.9977   0.9825   0.4213  
0.40       0.8633   0.5897   0.8500   0.9977   0.9825   0.4213  
0.45       0.8633   0.5897   0.8500   0.9977   0.9825   0.4213  
0.50       0.8633   0.5897   0.8500   0.9977   0.9825   0.4213  
0.55       0.8633   0.5897   0.8500   0.9977   0.9825   0.4213  
0.60       0.8633   0.5897   0.8500   0.9977   0.9825   0.4213  
0.65       0.8633   0.5897   0.8500   0.9977   0.9825   0.4213  
0.70       0.8633   0.5897   0.8500   0.9977   0.9825   0.4213  
0.75       0.8633   0.5897   0.8500   0.9977   0.9825   0.4213  
0.80       0.8633   0.5897   0.8500   0.9977   0.9825   0.4213  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8633, F1=0.5897, Normal Recall=0.8500, Normal Precision=0.9977, Attack Recall=0.9825, Attack Precision=0.4213

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
0.15       0.8767   0.7611   0.8502   0.9948   0.9823   0.6212   <--
0.20       0.8767   0.7611   0.8502   0.9948   0.9823   0.6212  
0.25       0.8767   0.7611   0.8502   0.9948   0.9823   0.6212  
0.30       0.8767   0.7611   0.8502   0.9948   0.9823   0.6212  
0.35       0.8767   0.7611   0.8502   0.9948   0.9823   0.6212  
0.40       0.8767   0.7611   0.8502   0.9948   0.9823   0.6212  
0.45       0.8767   0.7611   0.8502   0.9948   0.9823   0.6212  
0.50       0.8767   0.7611   0.8502   0.9948   0.9823   0.6212  
0.55       0.8767   0.7611   0.8502   0.9948   0.9823   0.6212  
0.60       0.8767   0.7611   0.8502   0.9948   0.9823   0.6212  
0.65       0.8767   0.7611   0.8502   0.9948   0.9823   0.6212  
0.70       0.8767   0.7611   0.8502   0.9948   0.9823   0.6212  
0.75       0.8767   0.7611   0.8502   0.9948   0.9823   0.6212  
0.80       0.8767   0.7611   0.8502   0.9948   0.9823   0.6212  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8767, F1=0.7611, Normal Recall=0.8502, Normal Precision=0.9948, Attack Recall=0.9823, Attack Precision=0.6212

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
0.15       0.8896   0.8423   0.8499   0.9911   0.9823   0.7372   <--
0.20       0.8896   0.8423   0.8499   0.9911   0.9823   0.7372  
0.25       0.8896   0.8423   0.8499   0.9911   0.9823   0.7372  
0.30       0.8896   0.8423   0.8499   0.9911   0.9823   0.7372  
0.35       0.8896   0.8423   0.8499   0.9911   0.9823   0.7372  
0.40       0.8896   0.8423   0.8499   0.9911   0.9823   0.7372  
0.45       0.8896   0.8423   0.8499   0.9911   0.9823   0.7372  
0.50       0.8896   0.8423   0.8499   0.9911   0.9823   0.7372  
0.55       0.8896   0.8423   0.8499   0.9911   0.9823   0.7372  
0.60       0.8896   0.8423   0.8499   0.9911   0.9823   0.7372  
0.65       0.8896   0.8423   0.8499   0.9911   0.9823   0.7372  
0.70       0.8896   0.8423   0.8499   0.9911   0.9823   0.7372  
0.75       0.8896   0.8423   0.8499   0.9911   0.9823   0.7372  
0.80       0.8896   0.8423   0.8499   0.9911   0.9823   0.7372  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8896, F1=0.8423, Normal Recall=0.8499, Normal Precision=0.9911, Attack Recall=0.9823, Attack Precision=0.7372

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
0.15       0.9024   0.8895   0.8491   0.9863   0.9823   0.8127   <--
0.20       0.9024   0.8895   0.8491   0.9863   0.9823   0.8127  
0.25       0.9024   0.8895   0.8491   0.9863   0.9823   0.8127  
0.30       0.9024   0.8895   0.8491   0.9863   0.9823   0.8127  
0.35       0.9024   0.8895   0.8491   0.9863   0.9823   0.8127  
0.40       0.9024   0.8895   0.8491   0.9863   0.9823   0.8127  
0.45       0.9024   0.8895   0.8491   0.9863   0.9823   0.8127  
0.50       0.9024   0.8895   0.8491   0.9863   0.9823   0.8127  
0.55       0.9024   0.8895   0.8491   0.9863   0.9823   0.8127  
0.60       0.9024   0.8895   0.8491   0.9863   0.9823   0.8127  
0.65       0.9024   0.8895   0.8491   0.9863   0.9823   0.8127  
0.70       0.9024   0.8895   0.8491   0.9863   0.9823   0.8127  
0.75       0.9024   0.8895   0.8491   0.9863   0.9823   0.8127  
0.80       0.9024   0.8895   0.8491   0.9863   0.9823   0.8127  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9024, F1=0.8895, Normal Recall=0.8491, Normal Precision=0.9863, Attack Recall=0.9823, Attack Precision=0.8127

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
0.15       0.9151   0.9205   0.8479   0.9795   0.9823   0.8660   <--
0.20       0.9151   0.9205   0.8479   0.9795   0.9823   0.8660  
0.25       0.9151   0.9205   0.8479   0.9795   0.9823   0.8660  
0.30       0.9151   0.9205   0.8479   0.9795   0.9823   0.8660  
0.35       0.9151   0.9205   0.8479   0.9795   0.9823   0.8660  
0.40       0.9151   0.9205   0.8479   0.9795   0.9823   0.8660  
0.45       0.9151   0.9205   0.8479   0.9795   0.9823   0.8660  
0.50       0.9151   0.9205   0.8479   0.9795   0.9823   0.8660  
0.55       0.9151   0.9205   0.8479   0.9795   0.9823   0.8660  
0.60       0.9151   0.9205   0.8479   0.9795   0.9823   0.8660  
0.65       0.9151   0.9205   0.8479   0.9795   0.9823   0.8660  
0.70       0.9151   0.9205   0.8479   0.9795   0.9823   0.8660  
0.75       0.9151   0.9205   0.8479   0.9795   0.9823   0.8660  
0.80       0.9151   0.9205   0.8479   0.9795   0.9823   0.8660  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9151, F1=0.9205, Normal Recall=0.8479, Normal Precision=0.9795, Attack Recall=0.9823, Attack Precision=0.8660

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
0.15       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206   <--
0.20       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.25       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.30       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.35       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.40       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.45       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.50       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.55       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.60       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.65       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.70       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.75       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.80       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8629, F1=0.5889, Normal Recall=0.8497, Normal Precision=0.9977, Attack Recall=0.9820, Attack Precision=0.4206

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
0.15       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207   <--
0.20       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.25       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.30       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.35       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.40       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.45       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.50       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.55       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.60       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.65       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.70       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.75       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.80       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8764, F1=0.7606, Normal Recall=0.8500, Normal Precision=0.9947, Attack Recall=0.9820, Attack Precision=0.6207

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
0.15       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369   <--
0.20       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.25       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.30       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.35       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.40       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.45       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.50       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.55       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.60       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.65       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.70       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.75       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.80       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8894, F1=0.8420, Normal Recall=0.8498, Normal Precision=0.9910, Attack Recall=0.9820, Attack Precision=0.7369

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
0.15       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126   <--
0.20       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.25       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.30       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.35       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.40       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.45       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.50       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.55       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.60       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.65       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.70       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.75       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.80       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9022, F1=0.8893, Normal Recall=0.8490, Normal Precision=0.9861, Attack Recall=0.9820, Attack Precision=0.8126

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
0.15       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657   <--
0.20       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.25       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.30       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.35       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.40       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.45       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.50       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.55       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.60       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.65       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.70       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.75       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.80       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9148, F1=0.9202, Normal Recall=0.8477, Normal Precision=0.9792, Attack Recall=0.9820, Attack Precision=0.8657

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
0.15       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206   <--
0.20       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.25       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.30       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.35       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.40       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.45       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.50       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.55       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.60       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.65       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.70       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.75       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
0.80       0.8629   0.5889   0.8497   0.9977   0.9820   0.4206  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8629, F1=0.5889, Normal Recall=0.8497, Normal Precision=0.9977, Attack Recall=0.9820, Attack Precision=0.4206

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
0.15       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207   <--
0.20       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.25       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.30       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.35       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.40       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.45       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.50       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.55       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.60       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.65       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.70       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.75       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
0.80       0.8764   0.7606   0.8500   0.9947   0.9820   0.6207  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8764, F1=0.7606, Normal Recall=0.8500, Normal Precision=0.9947, Attack Recall=0.9820, Attack Precision=0.6207

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
0.15       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369   <--
0.20       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.25       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.30       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.35       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.40       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.45       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.50       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.55       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.60       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.65       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.70       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.75       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
0.80       0.8894   0.8420   0.8498   0.9910   0.9820   0.7369  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8894, F1=0.8420, Normal Recall=0.8498, Normal Precision=0.9910, Attack Recall=0.9820, Attack Precision=0.7369

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
0.15       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126   <--
0.20       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.25       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.30       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.35       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.40       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.45       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.50       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.55       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.60       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.65       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.70       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.75       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
0.80       0.9022   0.8893   0.8490   0.9861   0.9820   0.8126  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9022, F1=0.8893, Normal Recall=0.8490, Normal Precision=0.9861, Attack Recall=0.9820, Attack Precision=0.8126

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
0.15       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657   <--
0.20       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.25       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.30       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.35       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.40       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.45       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.50       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.55       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.60       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.65       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.70       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.75       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
0.80       0.9148   0.9202   0.8477   0.9792   0.9820   0.8657  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9148, F1=0.9202, Normal Recall=0.8477, Normal Precision=0.9792, Attack Recall=0.9820, Attack Precision=0.8657

```

