# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-23 03:53:26 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7768 | 0.7920 | 0.8075 | 0.8238 | 0.8395 | 0.8560 | 0.8706 | 0.8861 | 0.9021 | 0.9186 | 0.9335 |
| QAT+Prune only | 0.6415 | 0.6759 | 0.7108 | 0.7462 | 0.7824 | 0.8163 | 0.8525 | 0.8883 | 0.9243 | 0.9581 | 0.9943 |
| QAT+PTQ | 0.6396 | 0.6746 | 0.7100 | 0.7458 | 0.7820 | 0.8166 | 0.8532 | 0.8894 | 0.9258 | 0.9601 | 0.9967 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6396 | 0.6746 | 0.7100 | 0.7458 | 0.7820 | 0.8166 | 0.8532 | 0.8894 | 0.9258 | 0.9601 | 0.9967 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.4729 | 0.6598 | 0.7607 | 0.8231 | 0.8663 | 0.8964 | 0.9198 | 0.9385 | 0.9538 | 0.9656 |
| QAT+Prune only | 0.0000 | 0.3802 | 0.5790 | 0.7016 | 0.7852 | 0.8440 | 0.8900 | 0.9257 | 0.9546 | 0.9771 | 0.9971 |
| QAT+PTQ | 0.0000 | 0.3799 | 0.5789 | 0.7018 | 0.7853 | 0.8446 | 0.8907 | 0.9266 | 0.9556 | 0.9782 | 0.9983 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3799 | 0.5789 | 0.7018 | 0.7853 | 0.8446 | 0.8907 | 0.9266 | 0.9556 | 0.9782 | 0.9983 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7768 | 0.7763 | 0.7759 | 0.7768 | 0.7768 | 0.7785 | 0.7763 | 0.7754 | 0.7765 | 0.7848 | 0.0000 |
| QAT+Prune only | 0.6415 | 0.6406 | 0.6400 | 0.6399 | 0.6411 | 0.6383 | 0.6399 | 0.6408 | 0.6443 | 0.6327 | 0.0000 |
| QAT+PTQ | 0.6396 | 0.6388 | 0.6383 | 0.6383 | 0.6389 | 0.6365 | 0.6380 | 0.6390 | 0.6423 | 0.6307 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6396 | 0.6388 | 0.6383 | 0.6383 | 0.6389 | 0.6365 | 0.6380 | 0.6390 | 0.6423 | 0.6307 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7768 | 0.0000 | 0.0000 | 0.0000 | 0.7768 | 1.0000 |
| 90 | 10 | 299,940 | 0.7920 | 0.3167 | 0.9333 | 0.4729 | 0.7763 | 0.9905 |
| 80 | 20 | 291,350 | 0.8075 | 0.5102 | 0.9335 | 0.6598 | 0.7759 | 0.9790 |
| 70 | 30 | 194,230 | 0.8238 | 0.6418 | 0.9335 | 0.7607 | 0.7768 | 0.9646 |
| 60 | 40 | 145,675 | 0.8395 | 0.7360 | 0.9335 | 0.8231 | 0.7768 | 0.9460 |
| 50 | 50 | 116,540 | 0.8560 | 0.8082 | 0.9335 | 0.8663 | 0.7785 | 0.9213 |
| 40 | 60 | 97,115 | 0.8706 | 0.8622 | 0.9335 | 0.8964 | 0.7763 | 0.8861 |
| 30 | 70 | 83,240 | 0.8861 | 0.9065 | 0.9335 | 0.9198 | 0.7754 | 0.8332 |
| 20 | 80 | 72,835 | 0.9021 | 0.9435 | 0.9335 | 0.9385 | 0.7765 | 0.7448 |
| 10 | 90 | 64,740 | 0.9186 | 0.9750 | 0.9335 | 0.9538 | 0.7848 | 0.5673 |
| 0 | 100 | 58,270 | 0.9335 | 1.0000 | 0.9335 | 0.9656 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6415 | 0.0000 | 0.0000 | 0.0000 | 0.6415 | 1.0000 |
| 90 | 10 | 299,940 | 0.6759 | 0.2351 | 0.9942 | 0.3802 | 0.6406 | 0.9990 |
| 80 | 20 | 291,350 | 0.7108 | 0.4084 | 0.9943 | 0.5790 | 0.6400 | 0.9978 |
| 70 | 30 | 194,230 | 0.7462 | 0.5420 | 0.9943 | 0.7016 | 0.6399 | 0.9962 |
| 60 | 40 | 145,675 | 0.7824 | 0.6488 | 0.9943 | 0.7852 | 0.6411 | 0.9941 |
| 50 | 50 | 116,540 | 0.8163 | 0.7332 | 0.9943 | 0.8440 | 0.6383 | 0.9912 |
| 40 | 60 | 97,115 | 0.8525 | 0.8055 | 0.9943 | 0.8900 | 0.6399 | 0.9868 |
| 30 | 70 | 83,240 | 0.8883 | 0.8659 | 0.9943 | 0.9257 | 0.6408 | 0.9797 |
| 20 | 80 | 72,835 | 0.9243 | 0.9179 | 0.9943 | 0.9546 | 0.6443 | 0.9658 |
| 10 | 90 | 64,740 | 0.9581 | 0.9606 | 0.9943 | 0.9771 | 0.6327 | 0.9250 |
| 0 | 100 | 58,270 | 0.9943 | 1.0000 | 0.9943 | 0.9971 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6396 | 0.0000 | 0.0000 | 0.0000 | 0.6396 | 1.0000 |
| 90 | 10 | 299,940 | 0.6746 | 0.2347 | 0.9966 | 0.3799 | 0.6388 | 0.9994 |
| 80 | 20 | 291,350 | 0.7100 | 0.4079 | 0.9967 | 0.5789 | 0.6383 | 0.9987 |
| 70 | 30 | 194,230 | 0.7458 | 0.5415 | 0.9967 | 0.7018 | 0.6383 | 0.9978 |
| 60 | 40 | 145,675 | 0.7820 | 0.6479 | 0.9967 | 0.7853 | 0.6389 | 0.9966 |
| 50 | 50 | 116,540 | 0.8166 | 0.7328 | 0.9967 | 0.8446 | 0.6365 | 0.9948 |
| 40 | 60 | 97,115 | 0.8532 | 0.8051 | 0.9967 | 0.8907 | 0.6380 | 0.9923 |
| 30 | 70 | 83,240 | 0.8894 | 0.8656 | 0.9967 | 0.9266 | 0.6390 | 0.9881 |
| 20 | 80 | 72,835 | 0.9258 | 0.9177 | 0.9967 | 0.9556 | 0.6423 | 0.9799 |
| 10 | 90 | 64,740 | 0.9601 | 0.9605 | 0.9967 | 0.9782 | 0.6307 | 0.9551 |
| 0 | 100 | 58,270 | 0.9967 | 1.0000 | 0.9967 | 0.9983 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6396 | 0.0000 | 0.0000 | 0.0000 | 0.6396 | 1.0000 |
| 90 | 10 | 299,940 | 0.6746 | 0.2347 | 0.9966 | 0.3799 | 0.6388 | 0.9994 |
| 80 | 20 | 291,350 | 0.7100 | 0.4079 | 0.9967 | 0.5789 | 0.6383 | 0.9987 |
| 70 | 30 | 194,230 | 0.7458 | 0.5415 | 0.9967 | 0.7018 | 0.6383 | 0.9978 |
| 60 | 40 | 145,675 | 0.7820 | 0.6479 | 0.9967 | 0.7853 | 0.6389 | 0.9966 |
| 50 | 50 | 116,540 | 0.8166 | 0.7328 | 0.9967 | 0.8446 | 0.6365 | 0.9948 |
| 40 | 60 | 97,115 | 0.8532 | 0.8051 | 0.9967 | 0.8907 | 0.6380 | 0.9923 |
| 30 | 70 | 83,240 | 0.8894 | 0.8656 | 0.9967 | 0.9266 | 0.6390 | 0.9881 |
| 20 | 80 | 72,835 | 0.9258 | 0.9177 | 0.9967 | 0.9556 | 0.6423 | 0.9799 |
| 10 | 90 | 64,740 | 0.9601 | 0.9605 | 0.9967 | 0.9782 | 0.6307 | 0.9551 |
| 0 | 100 | 58,270 | 0.9967 | 1.0000 | 0.9967 | 0.9983 | 0.0000 | 0.0000 |


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
0.15       0.7921   0.4733   0.7763   0.9907   0.9342   0.3169   <--
0.20       0.7921   0.4733   0.7763   0.9907   0.9342   0.3169  
0.25       0.7921   0.4733   0.7763   0.9907   0.9342   0.3169  
0.30       0.7921   0.4733   0.7763   0.9907   0.9342   0.3169  
0.35       0.7921   0.4733   0.7763   0.9907   0.9342   0.3169  
0.40       0.7921   0.4733   0.7763   0.9907   0.9342   0.3169  
0.45       0.7921   0.4733   0.7763   0.9907   0.9342   0.3169  
0.50       0.7921   0.4733   0.7763   0.9907   0.9342   0.3169  
0.55       0.7921   0.4733   0.7763   0.9907   0.9342   0.3169  
0.60       0.7921   0.4733   0.7763   0.9907   0.9342   0.3169  
0.65       0.7921   0.4733   0.7763   0.9907   0.9342   0.3169  
0.70       0.7921   0.4733   0.7763   0.9907   0.9342   0.3169  
0.75       0.7921   0.4733   0.7763   0.9907   0.9342   0.3169  
0.80       0.7921   0.4733   0.7763   0.9907   0.9342   0.3169  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7921, F1=0.4733, Normal Recall=0.7763, Normal Precision=0.9907, Attack Recall=0.9342, Attack Precision=0.3169

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
0.15       0.8077   0.6601   0.7763   0.9790   0.9335   0.5105   <--
0.20       0.8077   0.6601   0.7763   0.9790   0.9335   0.5105  
0.25       0.8077   0.6601   0.7763   0.9790   0.9335   0.5105  
0.30       0.8077   0.6601   0.7763   0.9790   0.9335   0.5105  
0.35       0.8077   0.6601   0.7763   0.9790   0.9335   0.5105  
0.40       0.8077   0.6601   0.7763   0.9790   0.9335   0.5105  
0.45       0.8077   0.6601   0.7763   0.9790   0.9335   0.5105  
0.50       0.8077   0.6601   0.7763   0.9790   0.9335   0.5105  
0.55       0.8077   0.6601   0.7763   0.9790   0.9335   0.5105  
0.60       0.8077   0.6601   0.7763   0.9790   0.9335   0.5105  
0.65       0.8077   0.6601   0.7763   0.9790   0.9335   0.5105  
0.70       0.8077   0.6601   0.7763   0.9790   0.9335   0.5105  
0.75       0.8077   0.6601   0.7763   0.9790   0.9335   0.5105  
0.80       0.8077   0.6601   0.7763   0.9790   0.9335   0.5105  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8077, F1=0.6601, Normal Recall=0.7763, Normal Precision=0.9790, Attack Recall=0.9335, Attack Precision=0.5105

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
0.15       0.8239   0.7608   0.7770   0.9646   0.9335   0.6420   <--
0.20       0.8239   0.7608   0.7770   0.9646   0.9335   0.6420  
0.25       0.8239   0.7608   0.7770   0.9646   0.9335   0.6420  
0.30       0.8239   0.7608   0.7770   0.9646   0.9335   0.6420  
0.35       0.8239   0.7608   0.7770   0.9646   0.9335   0.6420  
0.40       0.8239   0.7608   0.7770   0.9646   0.9335   0.6420  
0.45       0.8239   0.7608   0.7770   0.9646   0.9335   0.6420  
0.50       0.8239   0.7608   0.7770   0.9646   0.9335   0.6420  
0.55       0.8239   0.7608   0.7770   0.9646   0.9335   0.6420  
0.60       0.8239   0.7608   0.7770   0.9646   0.9335   0.6420  
0.65       0.8239   0.7608   0.7770   0.9646   0.9335   0.6420  
0.70       0.8239   0.7608   0.7770   0.9646   0.9335   0.6420  
0.75       0.8239   0.7608   0.7770   0.9646   0.9335   0.6420  
0.80       0.8239   0.7608   0.7770   0.9646   0.9335   0.6420  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8239, F1=0.7608, Normal Recall=0.7770, Normal Precision=0.9646, Attack Recall=0.9335, Attack Precision=0.6420

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
0.15       0.8394   0.8230   0.7766   0.9460   0.9335   0.7359   <--
0.20       0.8394   0.8230   0.7766   0.9460   0.9335   0.7359  
0.25       0.8394   0.8230   0.7766   0.9460   0.9335   0.7359  
0.30       0.8394   0.8230   0.7766   0.9460   0.9335   0.7359  
0.35       0.8394   0.8230   0.7766   0.9460   0.9335   0.7359  
0.40       0.8394   0.8230   0.7766   0.9460   0.9335   0.7359  
0.45       0.8394   0.8230   0.7766   0.9460   0.9335   0.7359  
0.50       0.8394   0.8230   0.7766   0.9460   0.9335   0.7359  
0.55       0.8394   0.8230   0.7766   0.9460   0.9335   0.7359  
0.60       0.8394   0.8230   0.7766   0.9460   0.9335   0.7359  
0.65       0.8394   0.8230   0.7766   0.9460   0.9335   0.7359  
0.70       0.8394   0.8230   0.7766   0.9460   0.9335   0.7359  
0.75       0.8394   0.8230   0.7766   0.9460   0.9335   0.7359  
0.80       0.8394   0.8230   0.7766   0.9460   0.9335   0.7359  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8394, F1=0.8230, Normal Recall=0.7766, Normal Precision=0.9460, Attack Recall=0.9335, Attack Precision=0.7359

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
0.15       0.8547   0.8653   0.7760   0.9210   0.9335   0.8065   <--
0.20       0.8547   0.8653   0.7760   0.9210   0.9335   0.8065  
0.25       0.8547   0.8653   0.7760   0.9210   0.9335   0.8065  
0.30       0.8547   0.8653   0.7760   0.9210   0.9335   0.8065  
0.35       0.8547   0.8653   0.7760   0.9210   0.9335   0.8065  
0.40       0.8547   0.8653   0.7760   0.9210   0.9335   0.8065  
0.45       0.8547   0.8653   0.7760   0.9210   0.9335   0.8065  
0.50       0.8547   0.8653   0.7760   0.9210   0.9335   0.8065  
0.55       0.8547   0.8653   0.7760   0.9210   0.9335   0.8065  
0.60       0.8547   0.8653   0.7760   0.9210   0.9335   0.8065  
0.65       0.8547   0.8653   0.7760   0.9210   0.9335   0.8065  
0.70       0.8547   0.8653   0.7760   0.9210   0.9335   0.8065  
0.75       0.8547   0.8653   0.7760   0.9210   0.9335   0.8065  
0.80       0.8547   0.8653   0.7760   0.9210   0.9335   0.8065  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8547, F1=0.8653, Normal Recall=0.7760, Normal Precision=0.9210, Attack Recall=0.9335, Attack Precision=0.8065

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
0.15       0.6759   0.3803   0.6406   0.9990   0.9944   0.2351   <--
0.20       0.6759   0.3803   0.6406   0.9990   0.9944   0.2351  
0.25       0.6759   0.3803   0.6406   0.9990   0.9944   0.2351  
0.30       0.6759   0.3803   0.6406   0.9990   0.9944   0.2351  
0.35       0.6759   0.3803   0.6406   0.9990   0.9944   0.2351  
0.40       0.6759   0.3803   0.6406   0.9990   0.9944   0.2351  
0.45       0.6759   0.3803   0.6406   0.9990   0.9944   0.2351  
0.50       0.6759   0.3803   0.6406   0.9990   0.9944   0.2351  
0.55       0.6759   0.3803   0.6406   0.9990   0.9944   0.2351  
0.60       0.6759   0.3803   0.6406   0.9990   0.9944   0.2351  
0.65       0.6759   0.3803   0.6406   0.9990   0.9944   0.2351  
0.70       0.6759   0.3803   0.6406   0.9990   0.9944   0.2351  
0.75       0.6759   0.3803   0.6406   0.9990   0.9944   0.2351  
0.80       0.6759   0.3803   0.6406   0.9990   0.9944   0.2351  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6759, F1=0.3803, Normal Recall=0.6406, Normal Precision=0.9990, Attack Recall=0.9944, Attack Precision=0.2351

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
0.15       0.7119   0.5799   0.6413   0.9978   0.9943   0.4093   <--
0.20       0.7119   0.5799   0.6413   0.9978   0.9943   0.4093  
0.25       0.7119   0.5799   0.6413   0.9978   0.9943   0.4093  
0.30       0.7119   0.5799   0.6413   0.9978   0.9943   0.4093  
0.35       0.7119   0.5799   0.6413   0.9978   0.9943   0.4093  
0.40       0.7119   0.5799   0.6413   0.9978   0.9943   0.4093  
0.45       0.7119   0.5799   0.6413   0.9978   0.9943   0.4093  
0.50       0.7119   0.5799   0.6413   0.9978   0.9943   0.4093  
0.55       0.7119   0.5799   0.6413   0.9978   0.9943   0.4093  
0.60       0.7119   0.5799   0.6413   0.9978   0.9943   0.4093  
0.65       0.7119   0.5799   0.6413   0.9978   0.9943   0.4093  
0.70       0.7119   0.5799   0.6413   0.9978   0.9943   0.4093  
0.75       0.7119   0.5799   0.6413   0.9978   0.9943   0.4093  
0.80       0.7119   0.5799   0.6413   0.9978   0.9943   0.4093  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7119, F1=0.5799, Normal Recall=0.6413, Normal Precision=0.9978, Attack Recall=0.9943, Attack Precision=0.4093

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
0.15       0.7471   0.7023   0.6412   0.9962   0.9943   0.5429   <--
0.20       0.7471   0.7023   0.6412   0.9962   0.9943   0.5429  
0.25       0.7471   0.7023   0.6412   0.9962   0.9943   0.5429  
0.30       0.7471   0.7023   0.6412   0.9962   0.9943   0.5429  
0.35       0.7471   0.7023   0.6412   0.9962   0.9943   0.5429  
0.40       0.7471   0.7023   0.6412   0.9962   0.9943   0.5429  
0.45       0.7471   0.7023   0.6412   0.9962   0.9943   0.5429  
0.50       0.7471   0.7023   0.6412   0.9962   0.9943   0.5429  
0.55       0.7471   0.7023   0.6412   0.9962   0.9943   0.5429  
0.60       0.7471   0.7023   0.6412   0.9962   0.9943   0.5429  
0.65       0.7471   0.7023   0.6412   0.9962   0.9943   0.5429  
0.70       0.7471   0.7023   0.6412   0.9962   0.9943   0.5429  
0.75       0.7471   0.7023   0.6412   0.9962   0.9943   0.5429  
0.80       0.7471   0.7023   0.6412   0.9962   0.9943   0.5429  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7471, F1=0.7023, Normal Recall=0.6412, Normal Precision=0.9962, Attack Recall=0.9943, Attack Precision=0.5429

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
0.15       0.7825   0.7853   0.6413   0.9941   0.9943   0.6489   <--
0.20       0.7825   0.7853   0.6413   0.9941   0.9943   0.6489  
0.25       0.7825   0.7853   0.6413   0.9941   0.9943   0.6489  
0.30       0.7825   0.7853   0.6413   0.9941   0.9943   0.6489  
0.35       0.7825   0.7853   0.6413   0.9941   0.9943   0.6489  
0.40       0.7825   0.7853   0.6413   0.9941   0.9943   0.6489  
0.45       0.7825   0.7853   0.6413   0.9941   0.9943   0.6489  
0.50       0.7825   0.7853   0.6413   0.9941   0.9943   0.6489  
0.55       0.7825   0.7853   0.6413   0.9941   0.9943   0.6489  
0.60       0.7825   0.7853   0.6413   0.9941   0.9943   0.6489  
0.65       0.7825   0.7853   0.6413   0.9941   0.9943   0.6489  
0.70       0.7825   0.7853   0.6413   0.9941   0.9943   0.6489  
0.75       0.7825   0.7853   0.6413   0.9941   0.9943   0.6489  
0.80       0.7825   0.7853   0.6413   0.9941   0.9943   0.6489  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7825, F1=0.7853, Normal Recall=0.6413, Normal Precision=0.9941, Attack Recall=0.9943, Attack Precision=0.6489

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
0.15       0.8169   0.8445   0.6394   0.9912   0.9943   0.7339   <--
0.20       0.8169   0.8445   0.6394   0.9912   0.9943   0.7339  
0.25       0.8169   0.8445   0.6394   0.9912   0.9943   0.7339  
0.30       0.8169   0.8445   0.6394   0.9912   0.9943   0.7339  
0.35       0.8169   0.8445   0.6394   0.9912   0.9943   0.7339  
0.40       0.8169   0.8445   0.6394   0.9912   0.9943   0.7339  
0.45       0.8169   0.8445   0.6394   0.9912   0.9943   0.7339  
0.50       0.8169   0.8445   0.6394   0.9912   0.9943   0.7339  
0.55       0.8169   0.8445   0.6394   0.9912   0.9943   0.7339  
0.60       0.8169   0.8445   0.6394   0.9912   0.9943   0.7339  
0.65       0.8169   0.8445   0.6394   0.9912   0.9943   0.7339  
0.70       0.8169   0.8445   0.6394   0.9912   0.9943   0.7339  
0.75       0.8169   0.8445   0.6394   0.9912   0.9943   0.7339  
0.80       0.8169   0.8445   0.6394   0.9912   0.9943   0.7339  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8169, F1=0.8445, Normal Recall=0.6394, Normal Precision=0.9912, Attack Recall=0.9943, Attack Precision=0.7339

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
0.15       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347   <--
0.20       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.25       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.30       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.35       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.40       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.45       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.50       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.55       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.60       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.65       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.70       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.75       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.80       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6746, F1=0.3799, Normal Recall=0.6388, Normal Precision=0.9994, Attack Recall=0.9966, Attack Precision=0.2347

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
0.15       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087   <--
0.20       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.25       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.30       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.35       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.40       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.45       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.50       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.55       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.60       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.65       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.70       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.75       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.80       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7110, F1=0.5797, Normal Recall=0.6396, Normal Precision=0.9987, Attack Recall=0.9967, Attack Precision=0.4087

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
0.15       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423   <--
0.20       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.25       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.30       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.35       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.40       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.45       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.50       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.55       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.60       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.65       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.70       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.75       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.80       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7466, F1=0.7024, Normal Recall=0.6395, Normal Precision=0.9978, Attack Recall=0.9967, Attack Precision=0.5423

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
0.15       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482   <--
0.20       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.25       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.30       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.35       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.40       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.45       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.50       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.55       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.60       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.65       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.70       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.75       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.80       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7823, F1=0.7855, Normal Recall=0.6394, Normal Precision=0.9966, Attack Recall=0.9967, Attack Precision=0.6482

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
0.15       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331   <--
0.20       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.25       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.30       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.35       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.40       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.45       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.50       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.55       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.60       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.65       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.70       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.75       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.80       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8169, F1=0.8448, Normal Recall=0.6371, Normal Precision=0.9949, Attack Recall=0.9967, Attack Precision=0.7331

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
0.15       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347   <--
0.20       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.25       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.30       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.35       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.40       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.45       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.50       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.55       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.60       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.65       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.70       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.75       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
0.80       0.6746   0.3799   0.6388   0.9994   0.9966   0.2347  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6746, F1=0.3799, Normal Recall=0.6388, Normal Precision=0.9994, Attack Recall=0.9966, Attack Precision=0.2347

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
0.15       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087   <--
0.20       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.25       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.30       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.35       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.40       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.45       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.50       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.55       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.60       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.65       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.70       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.75       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
0.80       0.7110   0.5797   0.6396   0.9987   0.9967   0.4087  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7110, F1=0.5797, Normal Recall=0.6396, Normal Precision=0.9987, Attack Recall=0.9967, Attack Precision=0.4087

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
0.15       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423   <--
0.20       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.25       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.30       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.35       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.40       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.45       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.50       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.55       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.60       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.65       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.70       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.75       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
0.80       0.7466   0.7024   0.6395   0.9978   0.9967   0.5423  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7466, F1=0.7024, Normal Recall=0.6395, Normal Precision=0.9978, Attack Recall=0.9967, Attack Precision=0.5423

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
0.15       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482   <--
0.20       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.25       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.30       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.35       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.40       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.45       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.50       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.55       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.60       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.65       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.70       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.75       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
0.80       0.7823   0.7855   0.6394   0.9966   0.9967   0.6482  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7823, F1=0.7855, Normal Recall=0.6394, Normal Precision=0.9966, Attack Recall=0.9967, Attack Precision=0.6482

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
0.15       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331   <--
0.20       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.25       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.30       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.35       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.40       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.45       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.50       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.55       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.60       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.65       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.70       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.75       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
0.80       0.8169   0.8448   0.6371   0.9949   0.9967   0.7331  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8169, F1=0.8448, Normal Recall=0.6371, Normal Precision=0.9949, Attack Recall=0.9967, Attack Precision=0.7331

```

