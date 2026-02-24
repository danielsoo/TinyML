# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e2_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-14 22:52:44 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 60 |
| **Local epochs** | 2 |
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2645 | 0.3361 | 0.4082 | 0.4810 | 0.5532 | 0.6257 | 0.6957 | 0.7691 | 0.8405 | 0.9125 | 0.9853 |
| QAT+Prune only | 0.9692 | 0.9428 | 0.9166 | 0.8903 | 0.8634 | 0.8371 | 0.8110 | 0.7849 | 0.7578 | 0.7312 | 0.7051 |
| QAT+PTQ | 0.9681 | 0.9418 | 0.9156 | 0.8895 | 0.8625 | 0.8363 | 0.8104 | 0.7843 | 0.7574 | 0.7308 | 0.7048 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9681 | 0.9418 | 0.9156 | 0.8895 | 0.8625 | 0.8363 | 0.8104 | 0.7843 | 0.7574 | 0.7308 | 0.7048 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2291 | 0.3998 | 0.5325 | 0.6382 | 0.7247 | 0.7953 | 0.8566 | 0.9081 | 0.9530 | 0.9926 |
| QAT+Prune only | 0.0000 | 0.7113 | 0.7718 | 0.7942 | 0.8050 | 0.8123 | 0.8174 | 0.8211 | 0.8233 | 0.8252 | 0.8271 |
| QAT+PTQ | 0.0000 | 0.7074 | 0.7696 | 0.7928 | 0.8040 | 0.8116 | 0.8169 | 0.8206 | 0.8230 | 0.8250 | 0.8269 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.7074 | 0.7696 | 0.7928 | 0.8040 | 0.8116 | 0.8169 | 0.8206 | 0.8230 | 0.8250 | 0.8269 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2645 | 0.2639 | 0.2639 | 0.2649 | 0.2652 | 0.2662 | 0.2612 | 0.2646 | 0.2614 | 0.2576 | 0.0000 |
| QAT+Prune only | 0.9692 | 0.9694 | 0.9695 | 0.9697 | 0.9688 | 0.9691 | 0.9698 | 0.9710 | 0.9687 | 0.9659 | 0.0000 |
| QAT+PTQ | 0.9681 | 0.9682 | 0.9683 | 0.9686 | 0.9677 | 0.9678 | 0.9687 | 0.9698 | 0.9677 | 0.9648 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9681 | 0.9682 | 0.9683 | 0.9686 | 0.9677 | 0.9678 | 0.9687 | 0.9698 | 0.9677 | 0.9648 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2645 | 0.0000 | 0.0000 | 0.0000 | 0.2645 | 1.0000 |
| 90 | 10 | 299,940 | 0.3361 | 0.1296 | 0.9864 | 0.2291 | 0.2639 | 0.9943 |
| 80 | 20 | 291,350 | 0.4082 | 0.2507 | 0.9853 | 0.3998 | 0.2639 | 0.9863 |
| 70 | 30 | 194,230 | 0.4810 | 0.3649 | 0.9853 | 0.5325 | 0.2649 | 0.9768 |
| 60 | 40 | 145,675 | 0.5532 | 0.4720 | 0.9853 | 0.6382 | 0.2652 | 0.9643 |
| 50 | 50 | 116,540 | 0.6257 | 0.5731 | 0.9853 | 0.7247 | 0.2662 | 0.9476 |
| 40 | 60 | 97,115 | 0.6957 | 0.6667 | 0.9853 | 0.7953 | 0.2612 | 0.9221 |
| 30 | 70 | 83,240 | 0.7691 | 0.7576 | 0.9853 | 0.8566 | 0.2646 | 0.8852 |
| 20 | 80 | 72,835 | 0.8405 | 0.8422 | 0.9853 | 0.9081 | 0.2614 | 0.8163 |
| 10 | 90 | 64,740 | 0.9125 | 0.9228 | 0.9853 | 0.9530 | 0.2576 | 0.6606 |
| 0 | 100 | 58,270 | 0.9853 | 1.0000 | 0.9853 | 0.9926 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9692 | 0.0000 | 0.0000 | 0.0000 | 0.9692 | 1.0000 |
| 90 | 10 | 299,940 | 0.9428 | 0.7187 | 0.7041 | 0.7113 | 0.9694 | 0.9672 |
| 80 | 20 | 291,350 | 0.9166 | 0.8523 | 0.7051 | 0.7718 | 0.9695 | 0.9293 |
| 70 | 30 | 194,230 | 0.8903 | 0.9089 | 0.7051 | 0.7942 | 0.9697 | 0.8847 |
| 60 | 40 | 145,675 | 0.8634 | 0.9378 | 0.7051 | 0.8050 | 0.9688 | 0.8313 |
| 50 | 50 | 116,540 | 0.8371 | 0.9580 | 0.7051 | 0.8123 | 0.9691 | 0.7667 |
| 40 | 60 | 97,115 | 0.8110 | 0.9722 | 0.7051 | 0.8174 | 0.9698 | 0.6868 |
| 30 | 70 | 83,240 | 0.7849 | 0.9827 | 0.7051 | 0.8211 | 0.9710 | 0.5853 |
| 20 | 80 | 72,835 | 0.7578 | 0.9890 | 0.7051 | 0.8233 | 0.9687 | 0.4509 |
| 10 | 90 | 64,740 | 0.7312 | 0.9946 | 0.7051 | 0.8252 | 0.9659 | 0.2668 |
| 0 | 100 | 58,270 | 0.7051 | 1.0000 | 0.7051 | 0.8271 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9681 | 0.0000 | 0.0000 | 0.0000 | 0.9681 | 1.0000 |
| 90 | 10 | 299,940 | 0.9418 | 0.7111 | 0.7037 | 0.7074 | 0.9682 | 0.9671 |
| 80 | 20 | 291,350 | 0.9156 | 0.8473 | 0.7048 | 0.7696 | 0.9683 | 0.9292 |
| 70 | 30 | 194,230 | 0.8895 | 0.9059 | 0.7048 | 0.7928 | 0.9686 | 0.8845 |
| 60 | 40 | 145,675 | 0.8625 | 0.9356 | 0.7048 | 0.8040 | 0.9677 | 0.8310 |
| 50 | 50 | 116,540 | 0.8363 | 0.9564 | 0.7048 | 0.8116 | 0.9678 | 0.7663 |
| 40 | 60 | 97,115 | 0.8104 | 0.9713 | 0.7048 | 0.8169 | 0.9687 | 0.6863 |
| 30 | 70 | 83,240 | 0.7843 | 0.9820 | 0.7048 | 0.8206 | 0.9698 | 0.5847 |
| 20 | 80 | 72,835 | 0.7574 | 0.9887 | 0.7048 | 0.8230 | 0.9677 | 0.4504 |
| 10 | 90 | 64,740 | 0.7308 | 0.9945 | 0.7049 | 0.8250 | 0.9648 | 0.2664 |
| 0 | 100 | 58,270 | 0.7048 | 1.0000 | 0.7048 | 0.8269 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9681 | 0.0000 | 0.0000 | 0.0000 | 0.9681 | 1.0000 |
| 90 | 10 | 299,940 | 0.9418 | 0.7111 | 0.7037 | 0.7074 | 0.9682 | 0.9671 |
| 80 | 20 | 291,350 | 0.9156 | 0.8473 | 0.7048 | 0.7696 | 0.9683 | 0.9292 |
| 70 | 30 | 194,230 | 0.8895 | 0.9059 | 0.7048 | 0.7928 | 0.9686 | 0.8845 |
| 60 | 40 | 145,675 | 0.8625 | 0.9356 | 0.7048 | 0.8040 | 0.9677 | 0.8310 |
| 50 | 50 | 116,540 | 0.8363 | 0.9564 | 0.7048 | 0.8116 | 0.9678 | 0.7663 |
| 40 | 60 | 97,115 | 0.8104 | 0.9713 | 0.7048 | 0.8169 | 0.9687 | 0.6863 |
| 30 | 70 | 83,240 | 0.7843 | 0.9820 | 0.7048 | 0.8206 | 0.9698 | 0.5847 |
| 20 | 80 | 72,835 | 0.7574 | 0.9887 | 0.7048 | 0.8230 | 0.9677 | 0.4504 |
| 10 | 90 | 64,740 | 0.7308 | 0.9945 | 0.7049 | 0.8250 | 0.9648 | 0.2664 |
| 0 | 100 | 58,270 | 0.7048 | 1.0000 | 0.7048 | 0.8269 | 0.0000 | 0.0000 |


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
0.15       0.3361   0.2290   0.2639   0.9941   0.9858   0.1295   <--
0.20       0.3361   0.2290   0.2639   0.9941   0.9858   0.1295  
0.25       0.3361   0.2290   0.2639   0.9941   0.9858   0.1295  
0.30       0.3361   0.2290   0.2639   0.9941   0.9858   0.1295  
0.35       0.3361   0.2290   0.2639   0.9941   0.9858   0.1295  
0.40       0.3361   0.2290   0.2639   0.9941   0.9858   0.1295  
0.45       0.3361   0.2290   0.2639   0.9941   0.9858   0.1295  
0.50       0.3361   0.2290   0.2639   0.9941   0.9858   0.1295  
0.55       0.3361   0.2290   0.2639   0.9941   0.9858   0.1295  
0.60       0.3361   0.2290   0.2639   0.9941   0.9858   0.1295  
0.65       0.3361   0.2290   0.2639   0.9941   0.9858   0.1295  
0.70       0.3361   0.2290   0.2639   0.9941   0.9858   0.1295  
0.75       0.3361   0.2290   0.2639   0.9941   0.9858   0.1295  
0.80       0.3361   0.2290   0.2639   0.9941   0.9858   0.1295  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3361, F1=0.2290, Normal Recall=0.2639, Normal Precision=0.9941, Attack Recall=0.9858, Attack Precision=0.1295

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
0.15       0.4079   0.3996   0.2636   0.9862   0.9853   0.2506   <--
0.20       0.4079   0.3996   0.2636   0.9862   0.9853   0.2506  
0.25       0.4079   0.3996   0.2636   0.9862   0.9853   0.2506  
0.30       0.4079   0.3996   0.2636   0.9862   0.9853   0.2506  
0.35       0.4079   0.3996   0.2636   0.9862   0.9853   0.2506  
0.40       0.4079   0.3996   0.2636   0.9862   0.9853   0.2506  
0.45       0.4079   0.3996   0.2636   0.9862   0.9853   0.2506  
0.50       0.4079   0.3996   0.2636   0.9862   0.9853   0.2506  
0.55       0.4079   0.3996   0.2636   0.9862   0.9853   0.2506  
0.60       0.4079   0.3996   0.2636   0.9862   0.9853   0.2506  
0.65       0.4079   0.3996   0.2636   0.9862   0.9853   0.2506  
0.70       0.4079   0.3996   0.2636   0.9862   0.9853   0.2506  
0.75       0.4079   0.3996   0.2636   0.9862   0.9853   0.2506  
0.80       0.4079   0.3996   0.2636   0.9862   0.9853   0.2506  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4079, F1=0.3996, Normal Recall=0.2636, Normal Precision=0.9862, Attack Recall=0.9853, Attack Precision=0.2506

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
0.15       0.4808   0.5324   0.2646   0.9767   0.9853   0.3648   <--
0.20       0.4808   0.5324   0.2646   0.9767   0.9853   0.3648  
0.25       0.4808   0.5324   0.2646   0.9767   0.9853   0.3648  
0.30       0.4808   0.5324   0.2646   0.9767   0.9853   0.3648  
0.35       0.4808   0.5324   0.2646   0.9767   0.9853   0.3648  
0.40       0.4808   0.5324   0.2646   0.9767   0.9853   0.3648  
0.45       0.4808   0.5324   0.2646   0.9767   0.9853   0.3648  
0.50       0.4808   0.5324   0.2646   0.9767   0.9853   0.3648  
0.55       0.4808   0.5324   0.2646   0.9767   0.9853   0.3648  
0.60       0.4808   0.5324   0.2646   0.9767   0.9853   0.3648  
0.65       0.4808   0.5324   0.2646   0.9767   0.9853   0.3648  
0.70       0.4808   0.5324   0.2646   0.9767   0.9853   0.3648  
0.75       0.4808   0.5324   0.2646   0.9767   0.9853   0.3648  
0.80       0.4808   0.5324   0.2646   0.9767   0.9853   0.3648  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4808, F1=0.5324, Normal Recall=0.2646, Normal Precision=0.9767, Attack Recall=0.9853, Attack Precision=0.3648

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
0.15       0.5533   0.6383   0.2654   0.9644   0.9853   0.4721   <--
0.20       0.5533   0.6383   0.2654   0.9644   0.9853   0.4721  
0.25       0.5533   0.6383   0.2654   0.9644   0.9853   0.4721  
0.30       0.5533   0.6383   0.2654   0.9644   0.9853   0.4721  
0.35       0.5533   0.6383   0.2654   0.9644   0.9853   0.4721  
0.40       0.5533   0.6383   0.2654   0.9644   0.9853   0.4721  
0.45       0.5533   0.6383   0.2654   0.9644   0.9853   0.4721  
0.50       0.5533   0.6383   0.2654   0.9644   0.9853   0.4721  
0.55       0.5533   0.6383   0.2654   0.9644   0.9853   0.4721  
0.60       0.5533   0.6383   0.2654   0.9644   0.9853   0.4721  
0.65       0.5533   0.6383   0.2654   0.9644   0.9853   0.4721  
0.70       0.5533   0.6383   0.2654   0.9644   0.9853   0.4721  
0.75       0.5533   0.6383   0.2654   0.9644   0.9853   0.4721  
0.80       0.5533   0.6383   0.2654   0.9644   0.9853   0.4721  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5533, F1=0.6383, Normal Recall=0.2654, Normal Precision=0.9644, Attack Recall=0.9853, Attack Precision=0.4721

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
0.15       0.6256   0.7247   0.2659   0.9476   0.9853   0.5731   <--
0.20       0.6256   0.7247   0.2659   0.9476   0.9853   0.5731  
0.25       0.6256   0.7247   0.2659   0.9476   0.9853   0.5731  
0.30       0.6256   0.7247   0.2659   0.9476   0.9853   0.5731  
0.35       0.6256   0.7247   0.2659   0.9476   0.9853   0.5731  
0.40       0.6256   0.7247   0.2659   0.9476   0.9853   0.5731  
0.45       0.6256   0.7247   0.2659   0.9476   0.9853   0.5731  
0.50       0.6256   0.7247   0.2659   0.9476   0.9853   0.5731  
0.55       0.6256   0.7247   0.2659   0.9476   0.9853   0.5731  
0.60       0.6256   0.7247   0.2659   0.9476   0.9853   0.5731  
0.65       0.6256   0.7247   0.2659   0.9476   0.9853   0.5731  
0.70       0.6256   0.7247   0.2659   0.9476   0.9853   0.5731  
0.75       0.6256   0.7247   0.2659   0.9476   0.9853   0.5731  
0.80       0.6256   0.7247   0.2659   0.9476   0.9853   0.5731  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6256, F1=0.7247, Normal Recall=0.2659, Normal Precision=0.9476, Attack Recall=0.9853, Attack Precision=0.5731

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
0.15       0.9432   0.7134   0.9694   0.9675   0.7072   0.7196   <--
0.20       0.9432   0.7134   0.9694   0.9675   0.7072   0.7196  
0.25       0.9432   0.7134   0.9694   0.9675   0.7072   0.7196  
0.30       0.9432   0.7134   0.9694   0.9675   0.7072   0.7196  
0.35       0.9432   0.7134   0.9694   0.9675   0.7072   0.7196  
0.40       0.9432   0.7134   0.9694   0.9675   0.7072   0.7196  
0.45       0.9432   0.7134   0.9694   0.9675   0.7072   0.7196  
0.50       0.9432   0.7134   0.9694   0.9675   0.7072   0.7196  
0.55       0.9432   0.7134   0.9694   0.9675   0.7072   0.7196  
0.60       0.9432   0.7134   0.9694   0.9675   0.7072   0.7196  
0.65       0.9432   0.7134   0.9694   0.9675   0.7072   0.7196  
0.70       0.9432   0.7134   0.9694   0.9675   0.7072   0.7196  
0.75       0.9432   0.7134   0.9694   0.9675   0.7072   0.7196  
0.80       0.9432   0.7134   0.9694   0.9675   0.7072   0.7196  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9432, F1=0.7134, Normal Recall=0.9694, Normal Precision=0.9675, Attack Recall=0.7072, Attack Precision=0.7196

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
0.15       0.9164   0.7714   0.9692   0.9293   0.7051   0.8513   <--
0.20       0.9164   0.7714   0.9692   0.9293   0.7051   0.8513  
0.25       0.9164   0.7714   0.9692   0.9293   0.7051   0.8513  
0.30       0.9164   0.7714   0.9692   0.9293   0.7051   0.8513  
0.35       0.9164   0.7714   0.9692   0.9293   0.7051   0.8513  
0.40       0.9164   0.7714   0.9692   0.9293   0.7051   0.8513  
0.45       0.9164   0.7714   0.9692   0.9293   0.7051   0.8513  
0.50       0.9164   0.7714   0.9692   0.9293   0.7051   0.8513  
0.55       0.9164   0.7714   0.9692   0.9293   0.7051   0.8513  
0.60       0.9164   0.7714   0.9692   0.9293   0.7051   0.8513  
0.65       0.9164   0.7714   0.9692   0.9293   0.7051   0.8513  
0.70       0.9164   0.7714   0.9692   0.9293   0.7051   0.8513  
0.75       0.9164   0.7714   0.9692   0.9293   0.7051   0.8513  
0.80       0.9164   0.7714   0.9692   0.9293   0.7051   0.8513  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9164, F1=0.7714, Normal Recall=0.9692, Normal Precision=0.9293, Attack Recall=0.7051, Attack Precision=0.8513

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
0.15       0.8900   0.7936   0.9692   0.8846   0.7051   0.9074   <--
0.20       0.8900   0.7936   0.9692   0.8846   0.7051   0.9074  
0.25       0.8900   0.7936   0.9692   0.8846   0.7051   0.9074  
0.30       0.8900   0.7936   0.9692   0.8846   0.7051   0.9074  
0.35       0.8900   0.7936   0.9692   0.8846   0.7051   0.9074  
0.40       0.8900   0.7936   0.9692   0.8846   0.7051   0.9074  
0.45       0.8900   0.7936   0.9692   0.8846   0.7051   0.9074  
0.50       0.8900   0.7936   0.9692   0.8846   0.7051   0.9074  
0.55       0.8900   0.7936   0.9692   0.8846   0.7051   0.9074  
0.60       0.8900   0.7936   0.9692   0.8846   0.7051   0.9074  
0.65       0.8900   0.7936   0.9692   0.8846   0.7051   0.9074  
0.70       0.8900   0.7936   0.9692   0.8846   0.7051   0.9074  
0.75       0.8900   0.7936   0.9692   0.8846   0.7051   0.9074  
0.80       0.8900   0.7936   0.9692   0.8846   0.7051   0.9074  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8900, F1=0.7936, Normal Recall=0.9692, Normal Precision=0.8846, Attack Recall=0.7051, Attack Precision=0.9074

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
0.15       0.8636   0.8053   0.9692   0.8314   0.7051   0.9386   <--
0.20       0.8636   0.8053   0.9692   0.8314   0.7051   0.9386  
0.25       0.8636   0.8053   0.9692   0.8314   0.7051   0.9386  
0.30       0.8636   0.8053   0.9692   0.8314   0.7051   0.9386  
0.35       0.8636   0.8053   0.9692   0.8314   0.7051   0.9386  
0.40       0.8636   0.8053   0.9692   0.8314   0.7051   0.9386  
0.45       0.8636   0.8053   0.9692   0.8314   0.7051   0.9386  
0.50       0.8636   0.8053   0.9692   0.8314   0.7051   0.9386  
0.55       0.8636   0.8053   0.9692   0.8314   0.7051   0.9386  
0.60       0.8636   0.8053   0.9692   0.8314   0.7051   0.9386  
0.65       0.8636   0.8053   0.9692   0.8314   0.7051   0.9386  
0.70       0.8636   0.8053   0.9692   0.8314   0.7051   0.9386  
0.75       0.8636   0.8053   0.9692   0.8314   0.7051   0.9386  
0.80       0.8636   0.8053   0.9692   0.8314   0.7051   0.9386  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8636, F1=0.8053, Normal Recall=0.9692, Normal Precision=0.8314, Attack Recall=0.7051, Attack Precision=0.9386

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
0.15       0.8371   0.8123   0.9690   0.7667   0.7051   0.9579   <--
0.20       0.8371   0.8123   0.9690   0.7667   0.7051   0.9579  
0.25       0.8371   0.8123   0.9690   0.7667   0.7051   0.9579  
0.30       0.8371   0.8123   0.9690   0.7667   0.7051   0.9579  
0.35       0.8371   0.8123   0.9690   0.7667   0.7051   0.9579  
0.40       0.8371   0.8123   0.9690   0.7667   0.7051   0.9579  
0.45       0.8371   0.8123   0.9690   0.7667   0.7051   0.9579  
0.50       0.8371   0.8123   0.9690   0.7667   0.7051   0.9579  
0.55       0.8371   0.8123   0.9690   0.7667   0.7051   0.9579  
0.60       0.8371   0.8123   0.9690   0.7667   0.7051   0.9579  
0.65       0.8371   0.8123   0.9690   0.7667   0.7051   0.9579  
0.70       0.8371   0.8123   0.9690   0.7667   0.7051   0.9579  
0.75       0.8371   0.8123   0.9690   0.7667   0.7051   0.9579  
0.80       0.8371   0.8123   0.9690   0.7667   0.7051   0.9579  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8371, F1=0.8123, Normal Recall=0.9690, Normal Precision=0.7667, Attack Recall=0.7051, Attack Precision=0.9579

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
0.15       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121   <--
0.20       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.25       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.30       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.35       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.40       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.45       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.50       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.55       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.60       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.65       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.70       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.75       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.80       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9421, F1=0.7095, Normal Recall=0.9682, Normal Precision=0.9675, Attack Recall=0.7069, Attack Precision=0.7121

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
0.15       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467   <--
0.20       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.25       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.30       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.35       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.40       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.45       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.50       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.55       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.60       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.65       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.70       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.75       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.80       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9154, F1=0.7693, Normal Recall=0.9681, Normal Precision=0.9292, Attack Recall=0.7048, Attack Precision=0.8467

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
0.15       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042   <--
0.20       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.25       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.30       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.35       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.40       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.45       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.50       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.55       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.60       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.65       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.70       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.75       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.80       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8890, F1=0.7922, Normal Recall=0.9680, Normal Precision=0.8844, Attack Recall=0.7048, Attack Precision=0.9042

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
0.15       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368   <--
0.20       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.25       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.30       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.35       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.40       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.45       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.50       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.55       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.60       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.65       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.70       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.75       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.80       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8629, F1=0.8044, Normal Recall=0.9683, Normal Precision=0.8311, Attack Recall=0.7048, Attack Precision=0.9368

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
0.15       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567   <--
0.20       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.25       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.30       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.35       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.40       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.45       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.50       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.55       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.60       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.65       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.70       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.75       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.80       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8365, F1=0.8117, Normal Recall=0.9681, Normal Precision=0.7664, Attack Recall=0.7048, Attack Precision=0.9567

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
0.15       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121   <--
0.20       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.25       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.30       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.35       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.40       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.45       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.50       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.55       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.60       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.65       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.70       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.75       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
0.80       0.9421   0.7095   0.9682   0.9675   0.7069   0.7121  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9421, F1=0.7095, Normal Recall=0.9682, Normal Precision=0.9675, Attack Recall=0.7069, Attack Precision=0.7121

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
0.15       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467   <--
0.20       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.25       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.30       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.35       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.40       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.45       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.50       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.55       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.60       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.65       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.70       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.75       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
0.80       0.9154   0.7693   0.9681   0.9292   0.7048   0.8467  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9154, F1=0.7693, Normal Recall=0.9681, Normal Precision=0.9292, Attack Recall=0.7048, Attack Precision=0.8467

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
0.15       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042   <--
0.20       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.25       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.30       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.35       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.40       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.45       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.50       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.55       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.60       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.65       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.70       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.75       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
0.80       0.8890   0.7922   0.9680   0.8844   0.7048   0.9042  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8890, F1=0.7922, Normal Recall=0.9680, Normal Precision=0.8844, Attack Recall=0.7048, Attack Precision=0.9042

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
0.15       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368   <--
0.20       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.25       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.30       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.35       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.40       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.45       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.50       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.55       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.60       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.65       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.70       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.75       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
0.80       0.8629   0.8044   0.9683   0.8311   0.7048   0.9368  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8629, F1=0.8044, Normal Recall=0.9683, Normal Precision=0.8311, Attack Recall=0.7048, Attack Precision=0.9368

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
0.15       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567   <--
0.20       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.25       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.30       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.35       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.40       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.45       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.50       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.55       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.60       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.65       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.70       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.75       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
0.80       0.8365   0.8117   0.9681   0.7664   0.7048   0.9567  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8365, F1=0.8117, Normal Recall=0.9681, Normal Precision=0.7664, Attack Recall=0.7048, Attack Precision=0.9567

```

