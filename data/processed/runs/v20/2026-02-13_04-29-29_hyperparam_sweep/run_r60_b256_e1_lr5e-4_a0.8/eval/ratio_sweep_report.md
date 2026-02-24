# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e1_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-15 09:18:51 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9200 | 0.9276 | 0.9349 | 0.9429 | 0.9497 | 0.9569 | 0.9651 | 0.9728 | 0.9796 | 0.9871 | 0.9949 |
| QAT+Prune only | 0.5436 | 0.5894 | 0.6342 | 0.6803 | 0.7260 | 0.7708 | 0.8163 | 0.8625 | 0.9089 | 0.9530 | 0.9995 |
| QAT+PTQ | 0.5437 | 0.5895 | 0.6343 | 0.6803 | 0.7263 | 0.7711 | 0.8165 | 0.8625 | 0.9089 | 0.9528 | 0.9995 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.5437 | 0.5895 | 0.6343 | 0.6803 | 0.7263 | 0.7711 | 0.8165 | 0.8625 | 0.9089 | 0.9528 | 0.9995 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.7331 | 0.8593 | 0.9126 | 0.9405 | 0.9585 | 0.9716 | 0.9808 | 0.9874 | 0.9929 | 0.9975 |
| QAT+Prune only | 0.0000 | 0.3274 | 0.5222 | 0.6523 | 0.7448 | 0.8135 | 0.8672 | 0.9105 | 0.9461 | 0.9745 | 0.9997 |
| QAT+PTQ | 0.0000 | 0.3275 | 0.5223 | 0.6523 | 0.7450 | 0.8136 | 0.8673 | 0.9105 | 0.9461 | 0.9744 | 0.9997 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3275 | 0.5223 | 0.6523 | 0.7450 | 0.8136 | 0.8673 | 0.9105 | 0.9461 | 0.9744 | 0.9997 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9200 | 0.9201 | 0.9198 | 0.9205 | 0.9195 | 0.9189 | 0.9203 | 0.9211 | 0.9185 | 0.9171 | 0.0000 |
| QAT+Prune only | 0.5436 | 0.5438 | 0.5429 | 0.5436 | 0.5437 | 0.5422 | 0.5415 | 0.5428 | 0.5469 | 0.5343 | 0.0000 |
| QAT+PTQ | 0.5437 | 0.5439 | 0.5431 | 0.5436 | 0.5442 | 0.5427 | 0.5420 | 0.5429 | 0.5466 | 0.5327 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.5437 | 0.5439 | 0.5431 | 0.5436 | 0.5442 | 0.5427 | 0.5420 | 0.5429 | 0.5466 | 0.5327 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9200 | 0.0000 | 0.0000 | 0.0000 | 0.9200 | 1.0000 |
| 90 | 10 | 299,940 | 0.9276 | 0.5804 | 0.9951 | 0.7331 | 0.9201 | 0.9994 |
| 80 | 20 | 291,350 | 0.9349 | 0.7563 | 0.9949 | 0.8593 | 0.9198 | 0.9986 |
| 70 | 30 | 194,230 | 0.9429 | 0.8429 | 0.9949 | 0.9126 | 0.9205 | 0.9976 |
| 60 | 40 | 145,675 | 0.9497 | 0.8918 | 0.9949 | 0.9405 | 0.9195 | 0.9963 |
| 50 | 50 | 116,540 | 0.9569 | 0.9247 | 0.9949 | 0.9585 | 0.9189 | 0.9945 |
| 40 | 60 | 97,115 | 0.9651 | 0.9493 | 0.9949 | 0.9716 | 0.9203 | 0.9918 |
| 30 | 70 | 83,240 | 0.9728 | 0.9671 | 0.9949 | 0.9808 | 0.9211 | 0.9873 |
| 20 | 80 | 72,835 | 0.9796 | 0.9799 | 0.9949 | 0.9874 | 0.9185 | 0.9784 |
| 10 | 90 | 64,740 | 0.9871 | 0.9908 | 0.9949 | 0.9929 | 0.9171 | 0.9525 |
| 0 | 100 | 58,270 | 0.9949 | 1.0000 | 0.9949 | 0.9975 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5436 | 0.0000 | 0.0000 | 0.0000 | 0.5436 | 1.0000 |
| 90 | 10 | 299,940 | 0.5894 | 0.1958 | 0.9994 | 0.3274 | 0.5438 | 0.9999 |
| 80 | 20 | 291,350 | 0.6342 | 0.3534 | 0.9995 | 0.5222 | 0.5429 | 0.9998 |
| 70 | 30 | 194,230 | 0.6803 | 0.4841 | 0.9995 | 0.6523 | 0.5436 | 0.9996 |
| 60 | 40 | 145,675 | 0.7260 | 0.5936 | 0.9995 | 0.7448 | 0.5437 | 0.9993 |
| 50 | 50 | 116,540 | 0.7708 | 0.6859 | 0.9995 | 0.8135 | 0.5422 | 0.9990 |
| 40 | 60 | 97,115 | 0.8163 | 0.7658 | 0.9995 | 0.8672 | 0.5415 | 0.9985 |
| 30 | 70 | 83,240 | 0.8625 | 0.8361 | 0.9995 | 0.9105 | 0.5428 | 0.9977 |
| 20 | 80 | 72,835 | 0.9089 | 0.8982 | 0.9995 | 0.9461 | 0.5469 | 0.9961 |
| 10 | 90 | 64,740 | 0.9530 | 0.9508 | 0.9995 | 0.9745 | 0.5343 | 0.9911 |
| 0 | 100 | 58,270 | 0.9995 | 1.0000 | 0.9995 | 0.9997 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5437 | 0.0000 | 0.0000 | 0.0000 | 0.5437 | 1.0000 |
| 90 | 10 | 299,940 | 0.5895 | 0.1958 | 0.9994 | 0.3275 | 0.5439 | 0.9999 |
| 80 | 20 | 291,350 | 0.6343 | 0.3535 | 0.9995 | 0.5223 | 0.5431 | 0.9997 |
| 70 | 30 | 194,230 | 0.6803 | 0.4841 | 0.9995 | 0.6523 | 0.5436 | 0.9996 |
| 60 | 40 | 145,675 | 0.7263 | 0.5938 | 0.9995 | 0.7450 | 0.5442 | 0.9993 |
| 50 | 50 | 116,540 | 0.7711 | 0.6861 | 0.9995 | 0.8136 | 0.5427 | 0.9990 |
| 40 | 60 | 97,115 | 0.8165 | 0.7660 | 0.9995 | 0.8673 | 0.5420 | 0.9985 |
| 30 | 70 | 83,240 | 0.8625 | 0.8361 | 0.9995 | 0.9105 | 0.5429 | 0.9976 |
| 20 | 80 | 72,835 | 0.9089 | 0.8981 | 0.9995 | 0.9461 | 0.5466 | 0.9960 |
| 10 | 90 | 64,740 | 0.9528 | 0.9506 | 0.9995 | 0.9744 | 0.5327 | 0.9908 |
| 0 | 100 | 58,270 | 0.9995 | 1.0000 | 0.9995 | 0.9997 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.5437 | 0.0000 | 0.0000 | 0.0000 | 0.5437 | 1.0000 |
| 90 | 10 | 299,940 | 0.5895 | 0.1958 | 0.9994 | 0.3275 | 0.5439 | 0.9999 |
| 80 | 20 | 291,350 | 0.6343 | 0.3535 | 0.9995 | 0.5223 | 0.5431 | 0.9997 |
| 70 | 30 | 194,230 | 0.6803 | 0.4841 | 0.9995 | 0.6523 | 0.5436 | 0.9996 |
| 60 | 40 | 145,675 | 0.7263 | 0.5938 | 0.9995 | 0.7450 | 0.5442 | 0.9993 |
| 50 | 50 | 116,540 | 0.7711 | 0.6861 | 0.9995 | 0.8136 | 0.5427 | 0.9990 |
| 40 | 60 | 97,115 | 0.8165 | 0.7660 | 0.9995 | 0.8673 | 0.5420 | 0.9985 |
| 30 | 70 | 83,240 | 0.8625 | 0.8361 | 0.9995 | 0.9105 | 0.5429 | 0.9976 |
| 20 | 80 | 72,835 | 0.9089 | 0.8981 | 0.9995 | 0.9461 | 0.5466 | 0.9960 |
| 10 | 90 | 64,740 | 0.9528 | 0.9506 | 0.9995 | 0.9744 | 0.5327 | 0.9908 |
| 0 | 100 | 58,270 | 0.9995 | 1.0000 | 0.9995 | 0.9997 | 0.0000 | 0.0000 |


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
0.15       0.9276   0.7334   0.9201   0.9995   0.9956   0.5805   <--
0.20       0.9276   0.7334   0.9201   0.9995   0.9956   0.5805  
0.25       0.9276   0.7334   0.9201   0.9995   0.9956   0.5805  
0.30       0.9276   0.7334   0.9201   0.9995   0.9956   0.5805  
0.35       0.9276   0.7334   0.9201   0.9995   0.9956   0.5805  
0.40       0.9276   0.7334   0.9201   0.9995   0.9956   0.5805  
0.45       0.9276   0.7334   0.9201   0.9995   0.9956   0.5805  
0.50       0.9276   0.7334   0.9201   0.9995   0.9956   0.5805  
0.55       0.9276   0.7334   0.9201   0.9995   0.9956   0.5805  
0.60       0.9276   0.7334   0.9201   0.9995   0.9956   0.5805  
0.65       0.9276   0.7334   0.9201   0.9995   0.9956   0.5805  
0.70       0.9276   0.7334   0.9201   0.9995   0.9956   0.5805  
0.75       0.9276   0.7334   0.9201   0.9995   0.9956   0.5805  
0.80       0.9276   0.7334   0.9201   0.9995   0.9956   0.5805  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9276, F1=0.7334, Normal Recall=0.9201, Normal Precision=0.9995, Attack Recall=0.9956, Attack Precision=0.5805

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
0.15       0.9350   0.8595   0.9200   0.9986   0.9949   0.7566   <--
0.20       0.9350   0.8595   0.9200   0.9986   0.9949   0.7566  
0.25       0.9350   0.8595   0.9200   0.9986   0.9949   0.7566  
0.30       0.9350   0.8595   0.9200   0.9986   0.9949   0.7566  
0.35       0.9350   0.8595   0.9200   0.9986   0.9949   0.7566  
0.40       0.9350   0.8595   0.9200   0.9986   0.9949   0.7566  
0.45       0.9350   0.8595   0.9200   0.9986   0.9949   0.7566  
0.50       0.9350   0.8595   0.9200   0.9986   0.9949   0.7566  
0.55       0.9350   0.8595   0.9200   0.9986   0.9949   0.7566  
0.60       0.9350   0.8595   0.9200   0.9986   0.9949   0.7566  
0.65       0.9350   0.8595   0.9200   0.9986   0.9949   0.7566  
0.70       0.9350   0.8595   0.9200   0.9986   0.9949   0.7566  
0.75       0.9350   0.8595   0.9200   0.9986   0.9949   0.7566  
0.80       0.9350   0.8595   0.9200   0.9986   0.9949   0.7566  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9350, F1=0.8595, Normal Recall=0.9200, Normal Precision=0.9986, Attack Recall=0.9949, Attack Precision=0.7566

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
0.15       0.9427   0.9124   0.9203   0.9976   0.9949   0.8425   <--
0.20       0.9427   0.9124   0.9203   0.9976   0.9949   0.8425  
0.25       0.9427   0.9124   0.9203   0.9976   0.9949   0.8425  
0.30       0.9427   0.9124   0.9203   0.9976   0.9949   0.8425  
0.35       0.9427   0.9124   0.9203   0.9976   0.9949   0.8425  
0.40       0.9427   0.9124   0.9203   0.9976   0.9949   0.8425  
0.45       0.9427   0.9124   0.9203   0.9976   0.9949   0.8425  
0.50       0.9427   0.9124   0.9203   0.9976   0.9949   0.8425  
0.55       0.9427   0.9124   0.9203   0.9976   0.9949   0.8425  
0.60       0.9427   0.9124   0.9203   0.9976   0.9949   0.8425  
0.65       0.9427   0.9124   0.9203   0.9976   0.9949   0.8425  
0.70       0.9427   0.9124   0.9203   0.9976   0.9949   0.8425  
0.75       0.9427   0.9124   0.9203   0.9976   0.9949   0.8425  
0.80       0.9427   0.9124   0.9203   0.9976   0.9949   0.8425  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9427, F1=0.9124, Normal Recall=0.9203, Normal Precision=0.9976, Attack Recall=0.9949, Attack Precision=0.8425

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
0.15       0.9502   0.9411   0.9204   0.9963   0.9949   0.8928   <--
0.20       0.9502   0.9411   0.9204   0.9963   0.9949   0.8928  
0.25       0.9502   0.9411   0.9204   0.9963   0.9949   0.8928  
0.30       0.9502   0.9411   0.9204   0.9963   0.9949   0.8928  
0.35       0.9502   0.9411   0.9204   0.9963   0.9949   0.8928  
0.40       0.9502   0.9411   0.9204   0.9963   0.9949   0.8928  
0.45       0.9502   0.9411   0.9204   0.9963   0.9949   0.8928  
0.50       0.9502   0.9411   0.9204   0.9963   0.9949   0.8928  
0.55       0.9502   0.9411   0.9204   0.9963   0.9949   0.8928  
0.60       0.9502   0.9411   0.9204   0.9963   0.9949   0.8928  
0.65       0.9502   0.9411   0.9204   0.9963   0.9949   0.8928  
0.70       0.9502   0.9411   0.9204   0.9963   0.9949   0.8928  
0.75       0.9502   0.9411   0.9204   0.9963   0.9949   0.8928  
0.80       0.9502   0.9411   0.9204   0.9963   0.9949   0.8928  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9502, F1=0.9411, Normal Recall=0.9204, Normal Precision=0.9963, Attack Recall=0.9949, Attack Precision=0.8928

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
0.15       0.9577   0.9592   0.9204   0.9945   0.9949   0.9259   <--
0.20       0.9577   0.9592   0.9204   0.9945   0.9949   0.9259  
0.25       0.9577   0.9592   0.9204   0.9945   0.9949   0.9259  
0.30       0.9577   0.9592   0.9204   0.9945   0.9949   0.9259  
0.35       0.9577   0.9592   0.9204   0.9945   0.9949   0.9259  
0.40       0.9577   0.9592   0.9204   0.9945   0.9949   0.9259  
0.45       0.9577   0.9592   0.9204   0.9945   0.9949   0.9259  
0.50       0.9577   0.9592   0.9204   0.9945   0.9949   0.9259  
0.55       0.9577   0.9592   0.9204   0.9945   0.9949   0.9259  
0.60       0.9577   0.9592   0.9204   0.9945   0.9949   0.9259  
0.65       0.9577   0.9592   0.9204   0.9945   0.9949   0.9259  
0.70       0.9577   0.9592   0.9204   0.9945   0.9949   0.9259  
0.75       0.9577   0.9592   0.9204   0.9945   0.9949   0.9259  
0.80       0.9577   0.9592   0.9204   0.9945   0.9949   0.9259  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9577, F1=0.9592, Normal Recall=0.9204, Normal Precision=0.9945, Attack Recall=0.9949, Attack Precision=0.9259

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
0.15       0.5894   0.3274   0.5438   0.9999   0.9995   0.1958   <--
0.20       0.5894   0.3274   0.5438   0.9999   0.9995   0.1958  
0.25       0.5894   0.3274   0.5438   0.9999   0.9995   0.1958  
0.30       0.5894   0.3274   0.5438   0.9999   0.9995   0.1958  
0.35       0.5894   0.3274   0.5438   0.9999   0.9995   0.1958  
0.40       0.5894   0.3274   0.5438   0.9999   0.9995   0.1958  
0.45       0.5894   0.3274   0.5438   0.9999   0.9995   0.1958  
0.50       0.5894   0.3274   0.5438   0.9999   0.9995   0.1958  
0.55       0.5894   0.3274   0.5438   0.9999   0.9995   0.1958  
0.60       0.5894   0.3274   0.5438   0.9999   0.9995   0.1958  
0.65       0.5894   0.3274   0.5438   0.9999   0.9995   0.1958  
0.70       0.5894   0.3274   0.5438   0.9999   0.9995   0.1958  
0.75       0.5894   0.3274   0.5438   0.9999   0.9995   0.1958  
0.80       0.5894   0.3274   0.5438   0.9999   0.9995   0.1958  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5894, F1=0.3274, Normal Recall=0.5438, Normal Precision=0.9999, Attack Recall=0.9995, Attack Precision=0.1958

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
0.15       0.6357   0.5232   0.5447   0.9998   0.9995   0.3544   <--
0.20       0.6357   0.5232   0.5447   0.9998   0.9995   0.3544  
0.25       0.6357   0.5232   0.5447   0.9998   0.9995   0.3544  
0.30       0.6357   0.5232   0.5447   0.9998   0.9995   0.3544  
0.35       0.6357   0.5232   0.5447   0.9998   0.9995   0.3544  
0.40       0.6357   0.5232   0.5447   0.9998   0.9995   0.3544  
0.45       0.6357   0.5232   0.5447   0.9998   0.9995   0.3544  
0.50       0.6357   0.5232   0.5447   0.9998   0.9995   0.3544  
0.55       0.6357   0.5232   0.5447   0.9998   0.9995   0.3544  
0.60       0.6357   0.5232   0.5447   0.9998   0.9995   0.3544  
0.65       0.6357   0.5232   0.5447   0.9998   0.9995   0.3544  
0.70       0.6357   0.5232   0.5447   0.9998   0.9995   0.3544  
0.75       0.6357   0.5232   0.5447   0.9998   0.9995   0.3544  
0.80       0.6357   0.5232   0.5447   0.9998   0.9995   0.3544  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6357, F1=0.5232, Normal Recall=0.5447, Normal Precision=0.9998, Attack Recall=0.9995, Attack Precision=0.3544

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
0.15       0.6806   0.6524   0.5439   0.9996   0.9995   0.4843   <--
0.20       0.6806   0.6524   0.5439   0.9996   0.9995   0.4843  
0.25       0.6806   0.6524   0.5439   0.9996   0.9995   0.4843  
0.30       0.6806   0.6524   0.5439   0.9996   0.9995   0.4843  
0.35       0.6806   0.6524   0.5439   0.9996   0.9995   0.4843  
0.40       0.6806   0.6524   0.5439   0.9996   0.9995   0.4843  
0.45       0.6806   0.6524   0.5439   0.9996   0.9995   0.4843  
0.50       0.6806   0.6524   0.5439   0.9996   0.9995   0.4843  
0.55       0.6806   0.6524   0.5439   0.9996   0.9995   0.4843  
0.60       0.6806   0.6524   0.5439   0.9996   0.9995   0.4843  
0.65       0.6806   0.6524   0.5439   0.9996   0.9995   0.4843  
0.70       0.6806   0.6524   0.5439   0.9996   0.9995   0.4843  
0.75       0.6806   0.6524   0.5439   0.9996   0.9995   0.4843  
0.80       0.6806   0.6524   0.5439   0.9996   0.9995   0.4843  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6806, F1=0.6524, Normal Recall=0.5439, Normal Precision=0.9996, Attack Recall=0.9995, Attack Precision=0.4843

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
0.15       0.7258   0.7446   0.5434   0.9993   0.9995   0.5934   <--
0.20       0.7258   0.7446   0.5434   0.9993   0.9995   0.5934  
0.25       0.7258   0.7446   0.5434   0.9993   0.9995   0.5934  
0.30       0.7258   0.7446   0.5434   0.9993   0.9995   0.5934  
0.35       0.7258   0.7446   0.5434   0.9993   0.9995   0.5934  
0.40       0.7258   0.7446   0.5434   0.9993   0.9995   0.5934  
0.45       0.7258   0.7446   0.5434   0.9993   0.9995   0.5934  
0.50       0.7258   0.7446   0.5434   0.9993   0.9995   0.5934  
0.55       0.7258   0.7446   0.5434   0.9993   0.9995   0.5934  
0.60       0.7258   0.7446   0.5434   0.9993   0.9995   0.5934  
0.65       0.7258   0.7446   0.5434   0.9993   0.9995   0.5934  
0.70       0.7258   0.7446   0.5434   0.9993   0.9995   0.5934  
0.75       0.7258   0.7446   0.5434   0.9993   0.9995   0.5934  
0.80       0.7258   0.7446   0.5434   0.9993   0.9995   0.5934  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7258, F1=0.7446, Normal Recall=0.5434, Normal Precision=0.9993, Attack Recall=0.9995, Attack Precision=0.5934

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
0.15       0.7709   0.8135   0.5424   0.9990   0.9995   0.6859   <--
0.20       0.7709   0.8135   0.5424   0.9990   0.9995   0.6859  
0.25       0.7709   0.8135   0.5424   0.9990   0.9995   0.6859  
0.30       0.7709   0.8135   0.5424   0.9990   0.9995   0.6859  
0.35       0.7709   0.8135   0.5424   0.9990   0.9995   0.6859  
0.40       0.7709   0.8135   0.5424   0.9990   0.9995   0.6859  
0.45       0.7709   0.8135   0.5424   0.9990   0.9995   0.6859  
0.50       0.7709   0.8135   0.5424   0.9990   0.9995   0.6859  
0.55       0.7709   0.8135   0.5424   0.9990   0.9995   0.6859  
0.60       0.7709   0.8135   0.5424   0.9990   0.9995   0.6859  
0.65       0.7709   0.8135   0.5424   0.9990   0.9995   0.6859  
0.70       0.7709   0.8135   0.5424   0.9990   0.9995   0.6859  
0.75       0.7709   0.8135   0.5424   0.9990   0.9995   0.6859  
0.80       0.7709   0.8135   0.5424   0.9990   0.9995   0.6859  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7709, F1=0.8135, Normal Recall=0.5424, Normal Precision=0.9990, Attack Recall=0.9995, Attack Precision=0.6859

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
0.15       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958   <--
0.20       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.25       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.30       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.35       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.40       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.45       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.50       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.55       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.60       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.65       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.70       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.75       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.80       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5895, F1=0.3275, Normal Recall=0.5439, Normal Precision=0.9999, Attack Recall=0.9995, Attack Precision=0.1958

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
0.15       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544   <--
0.20       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.25       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.30       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.35       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.40       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.45       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.50       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.55       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.60       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.65       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.70       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.75       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.80       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6358, F1=0.5233, Normal Recall=0.5449, Normal Precision=0.9997, Attack Recall=0.9995, Attack Precision=0.3544

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
0.15       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843   <--
0.20       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.25       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.30       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.35       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.40       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.45       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.50       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.55       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.60       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.65       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.70       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.75       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.80       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6806, F1=0.6525, Normal Recall=0.5440, Normal Precision=0.9996, Attack Recall=0.9995, Attack Precision=0.4843

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
0.15       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934   <--
0.20       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.25       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.30       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.35       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.40       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.45       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.50       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.55       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.60       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.65       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.70       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.75       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.80       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7259, F1=0.7447, Normal Recall=0.5435, Normal Precision=0.9993, Attack Recall=0.9995, Attack Precision=0.5934

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
0.15       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861   <--
0.20       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.25       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.30       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.35       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.40       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.45       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.50       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.55       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.60       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.65       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.70       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.75       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.80       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7711, F1=0.8137, Normal Recall=0.5427, Normal Precision=0.9990, Attack Recall=0.9995, Attack Precision=0.6861

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
0.15       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958   <--
0.20       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.25       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.30       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.35       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.40       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.45       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.50       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.55       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.60       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.65       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.70       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.75       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
0.80       0.5895   0.3275   0.5439   0.9999   0.9995   0.1958  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5895, F1=0.3275, Normal Recall=0.5439, Normal Precision=0.9999, Attack Recall=0.9995, Attack Precision=0.1958

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
0.15       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544   <--
0.20       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.25       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.30       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.35       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.40       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.45       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.50       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.55       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.60       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.65       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.70       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.75       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
0.80       0.6358   0.5233   0.5449   0.9997   0.9995   0.3544  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6358, F1=0.5233, Normal Recall=0.5449, Normal Precision=0.9997, Attack Recall=0.9995, Attack Precision=0.3544

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
0.15       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843   <--
0.20       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.25       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.30       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.35       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.40       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.45       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.50       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.55       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.60       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.65       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.70       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.75       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
0.80       0.6806   0.6525   0.5440   0.9996   0.9995   0.4843  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6806, F1=0.6525, Normal Recall=0.5440, Normal Precision=0.9996, Attack Recall=0.9995, Attack Precision=0.4843

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
0.15       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934   <--
0.20       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.25       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.30       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.35       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.40       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.45       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.50       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.55       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.60       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.65       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.70       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.75       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
0.80       0.7259   0.7447   0.5435   0.9993   0.9995   0.5934  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7259, F1=0.7447, Normal Recall=0.5435, Normal Precision=0.9993, Attack Recall=0.9995, Attack Precision=0.5934

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
0.15       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861   <--
0.20       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.25       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.30       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.35       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.40       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.45       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.50       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.55       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.60       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.65       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.70       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.75       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
0.80       0.7711   0.8137   0.5427   0.9990   0.9995   0.6861  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7711, F1=0.8137, Normal Recall=0.5427, Normal Precision=0.9990, Attack Recall=0.9995, Attack Precision=0.6861

```

