# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-13 11:09:33 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7108 | 0.7387 | 0.7654 | 0.7930 | 0.8197 | 0.8459 | 0.8741 | 0.9010 | 0.9262 | 0.9543 | 0.9812 |
| QAT+Prune only | 0.8386 | 0.8541 | 0.8694 | 0.8862 | 0.9013 | 0.9168 | 0.9336 | 0.9492 | 0.9642 | 0.9800 | 0.9962 |
| QAT+PTQ | 0.8382 | 0.8536 | 0.8689 | 0.8859 | 0.9010 | 0.9166 | 0.9333 | 0.9491 | 0.9641 | 0.9800 | 0.9962 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8382 | 0.8536 | 0.8689 | 0.8859 | 0.9010 | 0.9166 | 0.9333 | 0.9491 | 0.9641 | 0.9800 | 0.9962 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.4290 | 0.6259 | 0.7398 | 0.8133 | 0.8643 | 0.9034 | 0.9328 | 0.9551 | 0.9748 | 0.9905 |
| QAT+Prune only | 0.0000 | 0.5774 | 0.7531 | 0.8401 | 0.8898 | 0.9229 | 0.9474 | 0.9649 | 0.9780 | 0.9890 | 0.9981 |
| QAT+PTQ | 0.0000 | 0.5766 | 0.7525 | 0.8397 | 0.8895 | 0.9227 | 0.9472 | 0.9648 | 0.9780 | 0.9889 | 0.9981 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5766 | 0.7525 | 0.8397 | 0.8895 | 0.9227 | 0.9472 | 0.9648 | 0.9780 | 0.9889 | 0.9981 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7108 | 0.7117 | 0.7115 | 0.7123 | 0.7121 | 0.7106 | 0.7134 | 0.7138 | 0.7061 | 0.7121 | 0.0000 |
| QAT+Prune only | 0.8386 | 0.8383 | 0.8377 | 0.8391 | 0.8380 | 0.8375 | 0.8398 | 0.8396 | 0.8361 | 0.8343 | 0.0000 |
| QAT+PTQ | 0.8382 | 0.8378 | 0.8371 | 0.8386 | 0.8375 | 0.8369 | 0.8390 | 0.8391 | 0.8357 | 0.8338 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8382 | 0.8378 | 0.8371 | 0.8386 | 0.8375 | 0.8369 | 0.8390 | 0.8391 | 0.8357 | 0.8338 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7108 | 0.0000 | 0.0000 | 0.0000 | 0.7108 | 1.0000 |
| 90 | 10 | 299,940 | 0.7387 | 0.2745 | 0.9816 | 0.4290 | 0.7117 | 0.9971 |
| 80 | 20 | 291,350 | 0.7654 | 0.4595 | 0.9812 | 0.6259 | 0.7115 | 0.9934 |
| 70 | 30 | 194,230 | 0.7930 | 0.5938 | 0.9812 | 0.7398 | 0.7123 | 0.9888 |
| 60 | 40 | 145,675 | 0.8197 | 0.6944 | 0.9812 | 0.8133 | 0.7121 | 0.9827 |
| 50 | 50 | 116,540 | 0.8459 | 0.7722 | 0.9812 | 0.8643 | 0.7106 | 0.9742 |
| 40 | 60 | 97,115 | 0.8741 | 0.8370 | 0.9812 | 0.9034 | 0.7134 | 0.9620 |
| 30 | 70 | 83,240 | 0.9010 | 0.8889 | 0.9812 | 0.9328 | 0.7138 | 0.9421 |
| 20 | 80 | 72,835 | 0.9262 | 0.9303 | 0.9812 | 0.9551 | 0.7061 | 0.9037 |
| 10 | 90 | 64,740 | 0.9543 | 0.9684 | 0.9812 | 0.9748 | 0.7121 | 0.8081 |
| 0 | 100 | 58,270 | 0.9812 | 1.0000 | 0.9812 | 0.9905 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8386 | 0.0000 | 0.0000 | 0.0000 | 0.8386 | 1.0000 |
| 90 | 10 | 299,940 | 0.8541 | 0.4064 | 0.9964 | 0.5774 | 0.8383 | 0.9995 |
| 80 | 20 | 291,350 | 0.8694 | 0.6054 | 0.9962 | 0.7531 | 0.8377 | 0.9989 |
| 70 | 30 | 194,230 | 0.8862 | 0.7263 | 0.9962 | 0.8401 | 0.8391 | 0.9981 |
| 60 | 40 | 145,675 | 0.9013 | 0.8039 | 0.9962 | 0.8898 | 0.8380 | 0.9970 |
| 50 | 50 | 116,540 | 0.9168 | 0.8597 | 0.9962 | 0.9229 | 0.8375 | 0.9955 |
| 40 | 60 | 97,115 | 0.9336 | 0.9032 | 0.9962 | 0.9474 | 0.8398 | 0.9932 |
| 30 | 70 | 83,240 | 0.9492 | 0.9355 | 0.9962 | 0.9649 | 0.8396 | 0.9895 |
| 20 | 80 | 72,835 | 0.9642 | 0.9605 | 0.9962 | 0.9780 | 0.8361 | 0.9821 |
| 10 | 90 | 64,740 | 0.9800 | 0.9818 | 0.9962 | 0.9890 | 0.8343 | 0.9605 |
| 0 | 100 | 58,270 | 0.9962 | 1.0000 | 0.9962 | 0.9981 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8382 | 0.0000 | 0.0000 | 0.0000 | 0.8382 | 1.0000 |
| 90 | 10 | 299,940 | 0.8536 | 0.4056 | 0.9964 | 0.5766 | 0.8378 | 0.9995 |
| 80 | 20 | 291,350 | 0.8689 | 0.6046 | 0.9962 | 0.7525 | 0.8371 | 0.9989 |
| 70 | 30 | 194,230 | 0.8859 | 0.7257 | 0.9962 | 0.8397 | 0.8386 | 0.9981 |
| 60 | 40 | 145,675 | 0.9010 | 0.8035 | 0.9962 | 0.8895 | 0.8375 | 0.9970 |
| 50 | 50 | 116,540 | 0.9166 | 0.8593 | 0.9962 | 0.9227 | 0.8369 | 0.9955 |
| 40 | 60 | 97,115 | 0.9333 | 0.9027 | 0.9962 | 0.9472 | 0.8390 | 0.9932 |
| 30 | 70 | 83,240 | 0.9491 | 0.9352 | 0.9962 | 0.9648 | 0.8391 | 0.9895 |
| 20 | 80 | 72,835 | 0.9641 | 0.9604 | 0.9962 | 0.9780 | 0.8357 | 0.9821 |
| 10 | 90 | 64,740 | 0.9800 | 0.9818 | 0.9962 | 0.9889 | 0.8338 | 0.9605 |
| 0 | 100 | 58,270 | 0.9962 | 1.0000 | 0.9962 | 0.9981 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8382 | 0.0000 | 0.0000 | 0.0000 | 0.8382 | 1.0000 |
| 90 | 10 | 299,940 | 0.8536 | 0.4056 | 0.9964 | 0.5766 | 0.8378 | 0.9995 |
| 80 | 20 | 291,350 | 0.8689 | 0.6046 | 0.9962 | 0.7525 | 0.8371 | 0.9989 |
| 70 | 30 | 194,230 | 0.8859 | 0.7257 | 0.9962 | 0.8397 | 0.8386 | 0.9981 |
| 60 | 40 | 145,675 | 0.9010 | 0.8035 | 0.9962 | 0.8895 | 0.8375 | 0.9970 |
| 50 | 50 | 116,540 | 0.9166 | 0.8593 | 0.9962 | 0.9227 | 0.8369 | 0.9955 |
| 40 | 60 | 97,115 | 0.9333 | 0.9027 | 0.9962 | 0.9472 | 0.8390 | 0.9932 |
| 30 | 70 | 83,240 | 0.9491 | 0.9352 | 0.9962 | 0.9648 | 0.8391 | 0.9895 |
| 20 | 80 | 72,835 | 0.9641 | 0.9604 | 0.9962 | 0.9780 | 0.8357 | 0.9821 |
| 10 | 90 | 64,740 | 0.9800 | 0.9818 | 0.9962 | 0.9889 | 0.8338 | 0.9605 |
| 0 | 100 | 58,270 | 0.9962 | 1.0000 | 0.9962 | 0.9981 | 0.0000 | 0.0000 |


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
0.15       0.7387   0.4289   0.7117   0.9971   0.9814   0.2744   <--
0.20       0.7387   0.4289   0.7117   0.9971   0.9814   0.2744  
0.25       0.7387   0.4289   0.7117   0.9971   0.9814   0.2744  
0.30       0.7387   0.4289   0.7117   0.9971   0.9814   0.2744  
0.35       0.7387   0.4289   0.7117   0.9971   0.9814   0.2744  
0.40       0.7387   0.4289   0.7117   0.9971   0.9814   0.2744  
0.45       0.7387   0.4289   0.7117   0.9971   0.9814   0.2744  
0.50       0.7387   0.4289   0.7117   0.9971   0.9814   0.2744  
0.55       0.7387   0.4289   0.7117   0.9971   0.9814   0.2744  
0.60       0.7387   0.4289   0.7117   0.9971   0.9814   0.2744  
0.65       0.7387   0.4289   0.7117   0.9971   0.9814   0.2744  
0.70       0.7387   0.4289   0.7117   0.9971   0.9814   0.2744  
0.75       0.7387   0.4289   0.7117   0.9971   0.9814   0.2744  
0.80       0.7387   0.4289   0.7117   0.9971   0.9814   0.2744  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7387, F1=0.4289, Normal Recall=0.7117, Normal Precision=0.9971, Attack Recall=0.9814, Attack Precision=0.2744

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
0.15       0.7654   0.6259   0.7114   0.9934   0.9812   0.4595   <--
0.20       0.7654   0.6259   0.7114   0.9934   0.9812   0.4595  
0.25       0.7654   0.6259   0.7114   0.9934   0.9812   0.4595  
0.30       0.7654   0.6259   0.7114   0.9934   0.9812   0.4595  
0.35       0.7654   0.6259   0.7114   0.9934   0.9812   0.4595  
0.40       0.7654   0.6259   0.7114   0.9934   0.9812   0.4595  
0.45       0.7654   0.6259   0.7114   0.9934   0.9812   0.4595  
0.50       0.7654   0.6259   0.7114   0.9934   0.9812   0.4595  
0.55       0.7654   0.6259   0.7114   0.9934   0.9812   0.4595  
0.60       0.7654   0.6259   0.7114   0.9934   0.9812   0.4595  
0.65       0.7654   0.6259   0.7114   0.9934   0.9812   0.4595  
0.70       0.7654   0.6259   0.7114   0.9934   0.9812   0.4595  
0.75       0.7654   0.6259   0.7114   0.9934   0.9812   0.4595  
0.80       0.7654   0.6259   0.7114   0.9934   0.9812   0.4595  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7654, F1=0.6259, Normal Recall=0.7114, Normal Precision=0.9934, Attack Recall=0.9812, Attack Precision=0.4595

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
0.15       0.7927   0.7396   0.7120   0.9888   0.9812   0.5935   <--
0.20       0.7927   0.7396   0.7120   0.9888   0.9812   0.5935  
0.25       0.7927   0.7396   0.7120   0.9888   0.9812   0.5935  
0.30       0.7927   0.7396   0.7120   0.9888   0.9812   0.5935  
0.35       0.7927   0.7396   0.7120   0.9888   0.9812   0.5935  
0.40       0.7927   0.7396   0.7120   0.9888   0.9812   0.5935  
0.45       0.7927   0.7396   0.7120   0.9888   0.9812   0.5935  
0.50       0.7927   0.7396   0.7120   0.9888   0.9812   0.5935  
0.55       0.7927   0.7396   0.7120   0.9888   0.9812   0.5935  
0.60       0.7927   0.7396   0.7120   0.9888   0.9812   0.5935  
0.65       0.7927   0.7396   0.7120   0.9888   0.9812   0.5935  
0.70       0.7927   0.7396   0.7120   0.9888   0.9812   0.5935  
0.75       0.7927   0.7396   0.7120   0.9888   0.9812   0.5935  
0.80       0.7927   0.7396   0.7120   0.9888   0.9812   0.5935  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7927, F1=0.7396, Normal Recall=0.7120, Normal Precision=0.9888, Attack Recall=0.9812, Attack Precision=0.5935

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
0.15       0.8187   0.8124   0.7104   0.9827   0.9812   0.6931   <--
0.20       0.8187   0.8124   0.7104   0.9827   0.9812   0.6931  
0.25       0.8187   0.8124   0.7104   0.9827   0.9812   0.6931  
0.30       0.8187   0.8124   0.7104   0.9827   0.9812   0.6931  
0.35       0.8187   0.8124   0.7104   0.9827   0.9812   0.6931  
0.40       0.8187   0.8124   0.7104   0.9827   0.9812   0.6931  
0.45       0.8187   0.8124   0.7104   0.9827   0.9812   0.6931  
0.50       0.8187   0.8124   0.7104   0.9827   0.9812   0.6931  
0.55       0.8187   0.8124   0.7104   0.9827   0.9812   0.6931  
0.60       0.8187   0.8124   0.7104   0.9827   0.9812   0.6931  
0.65       0.8187   0.8124   0.7104   0.9827   0.9812   0.6931  
0.70       0.8187   0.8124   0.7104   0.9827   0.9812   0.6931  
0.75       0.8187   0.8124   0.7104   0.9827   0.9812   0.6931  
0.80       0.8187   0.8124   0.7104   0.9827   0.9812   0.6931  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8187, F1=0.8124, Normal Recall=0.7104, Normal Precision=0.9827, Attack Recall=0.9812, Attack Precision=0.6931

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
0.15       0.8457   0.8641   0.7103   0.9742   0.9812   0.7720   <--
0.20       0.8457   0.8641   0.7103   0.9742   0.9812   0.7720  
0.25       0.8457   0.8641   0.7103   0.9742   0.9812   0.7720  
0.30       0.8457   0.8641   0.7103   0.9742   0.9812   0.7720  
0.35       0.8457   0.8641   0.7103   0.9742   0.9812   0.7720  
0.40       0.8457   0.8641   0.7103   0.9742   0.9812   0.7720  
0.45       0.8457   0.8641   0.7103   0.9742   0.9812   0.7720  
0.50       0.8457   0.8641   0.7103   0.9742   0.9812   0.7720  
0.55       0.8457   0.8641   0.7103   0.9742   0.9812   0.7720  
0.60       0.8457   0.8641   0.7103   0.9742   0.9812   0.7720  
0.65       0.8457   0.8641   0.7103   0.9742   0.9812   0.7720  
0.70       0.8457   0.8641   0.7103   0.9742   0.9812   0.7720  
0.75       0.8457   0.8641   0.7103   0.9742   0.9812   0.7720  
0.80       0.8457   0.8641   0.7103   0.9742   0.9812   0.7720  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8457, F1=0.8641, Normal Recall=0.7103, Normal Precision=0.9742, Attack Recall=0.9812, Attack Precision=0.7720

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
0.15       0.8541   0.5774   0.8383   0.9995   0.9964   0.4064   <--
0.20       0.8541   0.5774   0.8383   0.9995   0.9964   0.4064  
0.25       0.8541   0.5774   0.8383   0.9995   0.9964   0.4064  
0.30       0.8541   0.5774   0.8383   0.9995   0.9964   0.4064  
0.35       0.8541   0.5774   0.8383   0.9995   0.9964   0.4064  
0.40       0.8541   0.5774   0.8383   0.9995   0.9964   0.4064  
0.45       0.8541   0.5774   0.8383   0.9995   0.9964   0.4064  
0.50       0.8541   0.5774   0.8383   0.9995   0.9964   0.4064  
0.55       0.8541   0.5774   0.8383   0.9995   0.9964   0.4064  
0.60       0.8541   0.5774   0.8383   0.9995   0.9964   0.4064  
0.65       0.8541   0.5774   0.8383   0.9995   0.9964   0.4064  
0.70       0.8541   0.5774   0.8383   0.9995   0.9964   0.4064  
0.75       0.8541   0.5774   0.8383   0.9995   0.9964   0.4064  
0.80       0.8541   0.5774   0.8383   0.9995   0.9964   0.4064  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8541, F1=0.5774, Normal Recall=0.8383, Normal Precision=0.9995, Attack Recall=0.9964, Attack Precision=0.4064

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
0.15       0.8703   0.7544   0.8388   0.9989   0.9962   0.6071   <--
0.20       0.8703   0.7544   0.8388   0.9989   0.9962   0.6071  
0.25       0.8703   0.7544   0.8388   0.9989   0.9962   0.6071  
0.30       0.8703   0.7544   0.8388   0.9989   0.9962   0.6071  
0.35       0.8703   0.7544   0.8388   0.9989   0.9962   0.6071  
0.40       0.8703   0.7544   0.8388   0.9989   0.9962   0.6071  
0.45       0.8703   0.7544   0.8388   0.9989   0.9962   0.6071  
0.50       0.8703   0.7544   0.8388   0.9989   0.9962   0.6071  
0.55       0.8703   0.7544   0.8388   0.9989   0.9962   0.6071  
0.60       0.8703   0.7544   0.8388   0.9989   0.9962   0.6071  
0.65       0.8703   0.7544   0.8388   0.9989   0.9962   0.6071  
0.70       0.8703   0.7544   0.8388   0.9989   0.9962   0.6071  
0.75       0.8703   0.7544   0.8388   0.9989   0.9962   0.6071  
0.80       0.8703   0.7544   0.8388   0.9989   0.9962   0.6071  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8703, F1=0.7544, Normal Recall=0.8388, Normal Precision=0.9989, Attack Recall=0.9962, Attack Precision=0.6071

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
0.15       0.8862   0.8401   0.8391   0.9981   0.9962   0.7262   <--
0.20       0.8862   0.8401   0.8391   0.9981   0.9962   0.7262  
0.25       0.8862   0.8401   0.8391   0.9981   0.9962   0.7262  
0.30       0.8862   0.8401   0.8391   0.9981   0.9962   0.7262  
0.35       0.8862   0.8401   0.8391   0.9981   0.9962   0.7262  
0.40       0.8862   0.8401   0.8391   0.9981   0.9962   0.7262  
0.45       0.8862   0.8401   0.8391   0.9981   0.9962   0.7262  
0.50       0.8862   0.8401   0.8391   0.9981   0.9962   0.7262  
0.55       0.8862   0.8401   0.8391   0.9981   0.9962   0.7262  
0.60       0.8862   0.8401   0.8391   0.9981   0.9962   0.7262  
0.65       0.8862   0.8401   0.8391   0.9981   0.9962   0.7262  
0.70       0.8862   0.8401   0.8391   0.9981   0.9962   0.7262  
0.75       0.8862   0.8401   0.8391   0.9981   0.9962   0.7262  
0.80       0.8862   0.8401   0.8391   0.9981   0.9962   0.7262  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8862, F1=0.8401, Normal Recall=0.8391, Normal Precision=0.9981, Attack Recall=0.9962, Attack Precision=0.7262

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
0.15       0.9016   0.8901   0.8386   0.9970   0.9962   0.8044   <--
0.20       0.9016   0.8901   0.8386   0.9970   0.9962   0.8044  
0.25       0.9016   0.8901   0.8386   0.9970   0.9962   0.8044  
0.30       0.9016   0.8901   0.8386   0.9970   0.9962   0.8044  
0.35       0.9016   0.8901   0.8386   0.9970   0.9962   0.8044  
0.40       0.9016   0.8901   0.8386   0.9970   0.9962   0.8044  
0.45       0.9016   0.8901   0.8386   0.9970   0.9962   0.8044  
0.50       0.9016   0.8901   0.8386   0.9970   0.9962   0.8044  
0.55       0.9016   0.8901   0.8386   0.9970   0.9962   0.8044  
0.60       0.9016   0.8901   0.8386   0.9970   0.9962   0.8044  
0.65       0.9016   0.8901   0.8386   0.9970   0.9962   0.8044  
0.70       0.9016   0.8901   0.8386   0.9970   0.9962   0.8044  
0.75       0.9016   0.8901   0.8386   0.9970   0.9962   0.8044  
0.80       0.9016   0.8901   0.8386   0.9970   0.9962   0.8044  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9016, F1=0.8901, Normal Recall=0.8386, Normal Precision=0.9970, Attack Recall=0.9962, Attack Precision=0.8044

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
0.15       0.9171   0.9232   0.8380   0.9955   0.9962   0.8601   <--
0.20       0.9171   0.9232   0.8380   0.9955   0.9962   0.8601  
0.25       0.9171   0.9232   0.8380   0.9955   0.9962   0.8601  
0.30       0.9171   0.9232   0.8380   0.9955   0.9962   0.8601  
0.35       0.9171   0.9232   0.8380   0.9955   0.9962   0.8601  
0.40       0.9171   0.9232   0.8380   0.9955   0.9962   0.8601  
0.45       0.9171   0.9232   0.8380   0.9955   0.9962   0.8601  
0.50       0.9171   0.9232   0.8380   0.9955   0.9962   0.8601  
0.55       0.9171   0.9232   0.8380   0.9955   0.9962   0.8601  
0.60       0.9171   0.9232   0.8380   0.9955   0.9962   0.8601  
0.65       0.9171   0.9232   0.8380   0.9955   0.9962   0.8601  
0.70       0.9171   0.9232   0.8380   0.9955   0.9962   0.8601  
0.75       0.9171   0.9232   0.8380   0.9955   0.9962   0.8601  
0.80       0.9171   0.9232   0.8380   0.9955   0.9962   0.8601  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9171, F1=0.9232, Normal Recall=0.8380, Normal Precision=0.9955, Attack Recall=0.9962, Attack Precision=0.8601

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
0.15       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056   <--
0.20       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.25       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.30       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.35       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.40       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.45       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.50       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.55       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.60       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.65       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.70       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.75       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.80       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8536, F1=0.5766, Normal Recall=0.8378, Normal Precision=0.9995, Attack Recall=0.9964, Attack Precision=0.4056

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
0.15       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063   <--
0.20       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.25       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.30       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.35       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.40       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.45       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.50       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.55       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.60       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.65       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.70       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.75       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.80       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8699, F1=0.7538, Normal Recall=0.8383, Normal Precision=0.9989, Attack Recall=0.9962, Attack Precision=0.6063

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
0.15       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258   <--
0.20       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.25       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.30       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.35       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.40       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.45       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.50       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.55       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.60       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.65       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.70       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.75       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.80       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8859, F1=0.8397, Normal Recall=0.8387, Normal Precision=0.9981, Attack Recall=0.9962, Attack Precision=0.7258

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
0.15       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041   <--
0.20       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.25       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.30       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.35       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.40       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.45       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.50       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.55       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.60       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.65       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.70       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.75       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.80       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9014, F1=0.8899, Normal Recall=0.8382, Normal Precision=0.9970, Attack Recall=0.9962, Attack Precision=0.8041

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
0.15       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598   <--
0.20       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.25       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.30       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.35       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.40       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.45       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.50       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.55       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.60       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.65       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.70       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.75       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.80       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9169, F1=0.9230, Normal Recall=0.8376, Normal Precision=0.9955, Attack Recall=0.9962, Attack Precision=0.8598

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
0.15       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056   <--
0.20       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.25       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.30       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.35       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.40       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.45       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.50       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.55       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.60       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.65       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.70       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.75       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
0.80       0.8536   0.5766   0.8378   0.9995   0.9964   0.4056  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8536, F1=0.5766, Normal Recall=0.8378, Normal Precision=0.9995, Attack Recall=0.9964, Attack Precision=0.4056

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
0.15       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063   <--
0.20       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.25       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.30       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.35       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.40       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.45       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.50       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.55       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.60       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.65       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.70       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.75       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
0.80       0.8699   0.7538   0.8383   0.9989   0.9962   0.6063  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8699, F1=0.7538, Normal Recall=0.8383, Normal Precision=0.9989, Attack Recall=0.9962, Attack Precision=0.6063

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
0.15       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258   <--
0.20       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.25       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.30       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.35       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.40       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.45       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.50       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.55       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.60       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.65       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.70       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.75       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
0.80       0.8859   0.8397   0.8387   0.9981   0.9962   0.7258  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8859, F1=0.8397, Normal Recall=0.8387, Normal Precision=0.9981, Attack Recall=0.9962, Attack Precision=0.7258

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
0.15       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041   <--
0.20       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.25       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.30       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.35       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.40       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.45       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.50       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.55       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.60       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.65       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.70       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.75       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
0.80       0.9014   0.8899   0.8382   0.9970   0.9962   0.8041  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9014, F1=0.8899, Normal Recall=0.8382, Normal Precision=0.9970, Attack Recall=0.9962, Attack Precision=0.8041

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
0.15       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598   <--
0.20       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.25       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.30       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.35       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.40       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.45       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.50       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.55       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.60       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.65       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.70       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.75       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
0.80       0.9169   0.9230   0.8376   0.9955   0.9962   0.8598  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9169, F1=0.9230, Normal Recall=0.8376, Normal Precision=0.9955, Attack Recall=0.9962, Attack Precision=0.8598

```

