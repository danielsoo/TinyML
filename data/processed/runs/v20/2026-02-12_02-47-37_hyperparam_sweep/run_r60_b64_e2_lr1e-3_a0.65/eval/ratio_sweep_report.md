# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e2_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-13 03:45:52 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4863 | 0.5223 | 0.5588 | 0.5962 | 0.6324 | 0.6687 | 0.7041 | 0.7429 | 0.7782 | 0.8150 | 0.8516 |
| QAT+Prune only | 0.7111 | 0.7407 | 0.7688 | 0.7973 | 0.8255 | 0.8536 | 0.8834 | 0.9119 | 0.9405 | 0.9685 | 0.9972 |
| QAT+PTQ | 0.7154 | 0.7445 | 0.7722 | 0.8004 | 0.8282 | 0.8557 | 0.8853 | 0.9131 | 0.9412 | 0.9687 | 0.9972 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7154 | 0.7445 | 0.7722 | 0.8004 | 0.8282 | 0.8557 | 0.8853 | 0.9131 | 0.9412 | 0.9687 | 0.9972 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2629 | 0.4357 | 0.5586 | 0.6495 | 0.7199 | 0.7755 | 0.8226 | 0.8600 | 0.8923 | 0.9198 |
| QAT+Prune only | 0.0000 | 0.4348 | 0.6331 | 0.7470 | 0.8205 | 0.8720 | 0.9112 | 0.9406 | 0.9640 | 0.9827 | 0.9986 |
| QAT+PTQ | 0.0000 | 0.4384 | 0.6365 | 0.7499 | 0.8228 | 0.8736 | 0.9125 | 0.9414 | 0.9644 | 0.9828 | 0.9986 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4384 | 0.6365 | 0.7499 | 0.8228 | 0.8736 | 0.9125 | 0.9414 | 0.9644 | 0.9828 | 0.9986 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4863 | 0.4857 | 0.4856 | 0.4867 | 0.4863 | 0.4858 | 0.4830 | 0.4894 | 0.4849 | 0.4859 | 0.0000 |
| QAT+Prune only | 0.7111 | 0.7122 | 0.7117 | 0.7116 | 0.7110 | 0.7101 | 0.7126 | 0.7126 | 0.7133 | 0.7096 | 0.0000 |
| QAT+PTQ | 0.7154 | 0.7164 | 0.7159 | 0.7161 | 0.7156 | 0.7143 | 0.7174 | 0.7170 | 0.7172 | 0.7121 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7154 | 0.7164 | 0.7159 | 0.7161 | 0.7156 | 0.7143 | 0.7174 | 0.7170 | 0.7172 | 0.7121 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4863 | 0.0000 | 0.0000 | 0.0000 | 0.4863 | 1.0000 |
| 90 | 10 | 299,940 | 0.5223 | 0.1555 | 0.8521 | 0.2629 | 0.4857 | 0.9673 |
| 80 | 20 | 291,350 | 0.5588 | 0.2927 | 0.8516 | 0.4357 | 0.4856 | 0.9290 |
| 70 | 30 | 194,230 | 0.5962 | 0.4156 | 0.8516 | 0.5586 | 0.4867 | 0.8844 |
| 60 | 40 | 145,675 | 0.6324 | 0.5250 | 0.8516 | 0.6495 | 0.4863 | 0.8309 |
| 50 | 50 | 116,540 | 0.6687 | 0.6235 | 0.8516 | 0.7199 | 0.4858 | 0.7660 |
| 40 | 60 | 97,115 | 0.7041 | 0.7119 | 0.8516 | 0.7755 | 0.4830 | 0.6845 |
| 30 | 70 | 83,240 | 0.7429 | 0.7956 | 0.8516 | 0.8226 | 0.4894 | 0.5856 |
| 20 | 80 | 72,835 | 0.7782 | 0.8686 | 0.8516 | 0.8600 | 0.4849 | 0.4496 |
| 10 | 90 | 64,740 | 0.8150 | 0.9371 | 0.8516 | 0.8923 | 0.4859 | 0.2668 |
| 0 | 100 | 58,270 | 0.8516 | 1.0000 | 0.8516 | 0.9198 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7111 | 0.0000 | 0.0000 | 0.0000 | 0.7111 | 1.0000 |
| 90 | 10 | 299,940 | 0.7407 | 0.2780 | 0.9974 | 0.4348 | 0.7122 | 0.9996 |
| 80 | 20 | 291,350 | 0.7688 | 0.4638 | 0.9972 | 0.6331 | 0.7117 | 0.9990 |
| 70 | 30 | 194,230 | 0.7973 | 0.5971 | 0.9972 | 0.7470 | 0.7116 | 0.9983 |
| 60 | 40 | 145,675 | 0.8255 | 0.6970 | 0.9972 | 0.8205 | 0.7110 | 0.9974 |
| 50 | 50 | 116,540 | 0.8536 | 0.7747 | 0.9972 | 0.8720 | 0.7101 | 0.9961 |
| 40 | 60 | 97,115 | 0.8834 | 0.8388 | 0.9972 | 0.9112 | 0.7126 | 0.9942 |
| 30 | 70 | 83,240 | 0.9119 | 0.8901 | 0.9972 | 0.9406 | 0.7126 | 0.9910 |
| 20 | 80 | 72,835 | 0.9405 | 0.9330 | 0.9972 | 0.9640 | 0.7133 | 0.9847 |
| 10 | 90 | 64,740 | 0.9685 | 0.9687 | 0.9972 | 0.9827 | 0.7096 | 0.9661 |
| 0 | 100 | 58,270 | 0.9972 | 1.0000 | 0.9972 | 0.9986 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7154 | 0.0000 | 0.0000 | 0.0000 | 0.7154 | 1.0000 |
| 90 | 10 | 299,940 | 0.7445 | 0.2810 | 0.9973 | 0.4384 | 0.7164 | 0.9996 |
| 80 | 20 | 291,350 | 0.7722 | 0.4674 | 0.9972 | 0.6365 | 0.7159 | 0.9990 |
| 70 | 30 | 194,230 | 0.8004 | 0.6008 | 0.9972 | 0.7499 | 0.7161 | 0.9983 |
| 60 | 40 | 145,675 | 0.8282 | 0.7004 | 0.9972 | 0.8228 | 0.7156 | 0.9974 |
| 50 | 50 | 116,540 | 0.8557 | 0.7773 | 0.9972 | 0.8736 | 0.7143 | 0.9961 |
| 40 | 60 | 97,115 | 0.8853 | 0.8411 | 0.9972 | 0.9125 | 0.7174 | 0.9941 |
| 30 | 70 | 83,240 | 0.9131 | 0.8916 | 0.9972 | 0.9414 | 0.7170 | 0.9909 |
| 20 | 80 | 72,835 | 0.9412 | 0.9338 | 0.9972 | 0.9644 | 0.7172 | 0.9845 |
| 10 | 90 | 64,740 | 0.9687 | 0.9689 | 0.9972 | 0.9828 | 0.7121 | 0.9656 |
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
| 100 | 0 | 100,000 | 0.7154 | 0.0000 | 0.0000 | 0.0000 | 0.7154 | 1.0000 |
| 90 | 10 | 299,940 | 0.7445 | 0.2810 | 0.9973 | 0.4384 | 0.7164 | 0.9996 |
| 80 | 20 | 291,350 | 0.7722 | 0.4674 | 0.9972 | 0.6365 | 0.7159 | 0.9990 |
| 70 | 30 | 194,230 | 0.8004 | 0.6008 | 0.9972 | 0.7499 | 0.7161 | 0.9983 |
| 60 | 40 | 145,675 | 0.8282 | 0.7004 | 0.9972 | 0.8228 | 0.7156 | 0.9974 |
| 50 | 50 | 116,540 | 0.8557 | 0.7773 | 0.9972 | 0.8736 | 0.7143 | 0.9961 |
| 40 | 60 | 97,115 | 0.8853 | 0.8411 | 0.9972 | 0.9125 | 0.7174 | 0.9941 |
| 30 | 70 | 83,240 | 0.9131 | 0.8916 | 0.9972 | 0.9414 | 0.7170 | 0.9909 |
| 20 | 80 | 72,835 | 0.9412 | 0.9338 | 0.9972 | 0.9644 | 0.7172 | 0.9845 |
| 10 | 90 | 64,740 | 0.9687 | 0.9689 | 0.9972 | 0.9828 | 0.7121 | 0.9656 |
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
0.15       0.5224   0.2631   0.4857   0.9674   0.8527   0.1556   <--
0.20       0.5224   0.2631   0.4857   0.9674   0.8527   0.1556  
0.25       0.5224   0.2631   0.4857   0.9674   0.8527   0.1556  
0.30       0.5224   0.2631   0.4857   0.9674   0.8527   0.1556  
0.35       0.5224   0.2631   0.4857   0.9674   0.8527   0.1556  
0.40       0.5224   0.2631   0.4857   0.9674   0.8527   0.1556  
0.45       0.5224   0.2631   0.4857   0.9674   0.8527   0.1556  
0.50       0.5224   0.2631   0.4857   0.9674   0.8527   0.1556  
0.55       0.5224   0.2631   0.4857   0.9674   0.8527   0.1556  
0.60       0.5224   0.2631   0.4857   0.9674   0.8527   0.1556  
0.65       0.5224   0.2631   0.4857   0.9674   0.8527   0.1556  
0.70       0.5224   0.2631   0.4857   0.9674   0.8527   0.1556  
0.75       0.5224   0.2631   0.4857   0.9674   0.8527   0.1556  
0.80       0.5224   0.2631   0.4857   0.9674   0.8527   0.1556  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5224, F1=0.2631, Normal Recall=0.4857, Normal Precision=0.9674, Attack Recall=0.8527, Attack Precision=0.1556

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
0.15       0.5585   0.4355   0.4852   0.9290   0.8516   0.2925   <--
0.20       0.5585   0.4355   0.4852   0.9290   0.8516   0.2925  
0.25       0.5585   0.4355   0.4852   0.9290   0.8516   0.2925  
0.30       0.5585   0.4355   0.4852   0.9290   0.8516   0.2925  
0.35       0.5585   0.4355   0.4852   0.9290   0.8516   0.2925  
0.40       0.5585   0.4355   0.4852   0.9290   0.8516   0.2925  
0.45       0.5585   0.4355   0.4852   0.9290   0.8516   0.2925  
0.50       0.5585   0.4355   0.4852   0.9290   0.8516   0.2925  
0.55       0.5585   0.4355   0.4852   0.9290   0.8516   0.2925  
0.60       0.5585   0.4355   0.4852   0.9290   0.8516   0.2925  
0.65       0.5585   0.4355   0.4852   0.9290   0.8516   0.2925  
0.70       0.5585   0.4355   0.4852   0.9290   0.8516   0.2925  
0.75       0.5585   0.4355   0.4852   0.9290   0.8516   0.2925  
0.80       0.5585   0.4355   0.4852   0.9290   0.8516   0.2925  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5585, F1=0.4355, Normal Recall=0.4852, Normal Precision=0.9290, Attack Recall=0.8516, Attack Precision=0.2925

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
0.15       0.5959   0.5584   0.4864   0.8843   0.8516   0.4154   <--
0.20       0.5959   0.5584   0.4864   0.8843   0.8516   0.4154  
0.25       0.5959   0.5584   0.4864   0.8843   0.8516   0.4154  
0.30       0.5959   0.5584   0.4864   0.8843   0.8516   0.4154  
0.35       0.5959   0.5584   0.4864   0.8843   0.8516   0.4154  
0.40       0.5959   0.5584   0.4864   0.8843   0.8516   0.4154  
0.45       0.5959   0.5584   0.4864   0.8843   0.8516   0.4154  
0.50       0.5959   0.5584   0.4864   0.8843   0.8516   0.4154  
0.55       0.5959   0.5584   0.4864   0.8843   0.8516   0.4154  
0.60       0.5959   0.5584   0.4864   0.8843   0.8516   0.4154  
0.65       0.5959   0.5584   0.4864   0.8843   0.8516   0.4154  
0.70       0.5959   0.5584   0.4864   0.8843   0.8516   0.4154  
0.75       0.5959   0.5584   0.4864   0.8843   0.8516   0.4154  
0.80       0.5959   0.5584   0.4864   0.8843   0.8516   0.4154  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5959, F1=0.5584, Normal Recall=0.4864, Normal Precision=0.8843, Attack Recall=0.8516, Attack Precision=0.4154

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
0.15       0.6326   0.6496   0.4866   0.8310   0.8516   0.5251   <--
0.20       0.6326   0.6496   0.4866   0.8310   0.8516   0.5251  
0.25       0.6326   0.6496   0.4866   0.8310   0.8516   0.5251  
0.30       0.6326   0.6496   0.4866   0.8310   0.8516   0.5251  
0.35       0.6326   0.6496   0.4866   0.8310   0.8516   0.5251  
0.40       0.6326   0.6496   0.4866   0.8310   0.8516   0.5251  
0.45       0.6326   0.6496   0.4866   0.8310   0.8516   0.5251  
0.50       0.6326   0.6496   0.4866   0.8310   0.8516   0.5251  
0.55       0.6326   0.6496   0.4866   0.8310   0.8516   0.5251  
0.60       0.6326   0.6496   0.4866   0.8310   0.8516   0.5251  
0.65       0.6326   0.6496   0.4866   0.8310   0.8516   0.5251  
0.70       0.6326   0.6496   0.4866   0.8310   0.8516   0.5251  
0.75       0.6326   0.6496   0.4866   0.8310   0.8516   0.5251  
0.80       0.6326   0.6496   0.4866   0.8310   0.8516   0.5251  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6326, F1=0.6496, Normal Recall=0.4866, Normal Precision=0.8310, Attack Recall=0.8516, Attack Precision=0.5251

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
0.15       0.6691   0.7201   0.4865   0.7663   0.8516   0.6239   <--
0.20       0.6691   0.7201   0.4865   0.7663   0.8516   0.6239  
0.25       0.6691   0.7201   0.4865   0.7663   0.8516   0.6239  
0.30       0.6691   0.7201   0.4865   0.7663   0.8516   0.6239  
0.35       0.6691   0.7201   0.4865   0.7663   0.8516   0.6239  
0.40       0.6691   0.7201   0.4865   0.7663   0.8516   0.6239  
0.45       0.6691   0.7201   0.4865   0.7663   0.8516   0.6239  
0.50       0.6691   0.7201   0.4865   0.7663   0.8516   0.6239  
0.55       0.6691   0.7201   0.4865   0.7663   0.8516   0.6239  
0.60       0.6691   0.7201   0.4865   0.7663   0.8516   0.6239  
0.65       0.6691   0.7201   0.4865   0.7663   0.8516   0.6239  
0.70       0.6691   0.7201   0.4865   0.7663   0.8516   0.6239  
0.75       0.6691   0.7201   0.4865   0.7663   0.8516   0.6239  
0.80       0.6691   0.7201   0.4865   0.7663   0.8516   0.6239  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6691, F1=0.7201, Normal Recall=0.4865, Normal Precision=0.7663, Attack Recall=0.8516, Attack Precision=0.6239

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
0.15       0.7407   0.4348   0.7122   0.9996   0.9974   0.2780   <--
0.20       0.7407   0.4348   0.7122   0.9996   0.9974   0.2780  
0.25       0.7407   0.4348   0.7122   0.9996   0.9974   0.2780  
0.30       0.7407   0.4348   0.7122   0.9996   0.9974   0.2780  
0.35       0.7407   0.4348   0.7122   0.9996   0.9974   0.2780  
0.40       0.7407   0.4348   0.7122   0.9996   0.9974   0.2780  
0.45       0.7407   0.4348   0.7122   0.9996   0.9974   0.2780  
0.50       0.7407   0.4348   0.7122   0.9996   0.9974   0.2780  
0.55       0.7407   0.4348   0.7122   0.9996   0.9974   0.2780  
0.60       0.7407   0.4348   0.7122   0.9996   0.9974   0.2780  
0.65       0.7407   0.4348   0.7122   0.9996   0.9974   0.2780  
0.70       0.7407   0.4348   0.7122   0.9996   0.9974   0.2780  
0.75       0.7407   0.4348   0.7122   0.9996   0.9974   0.2780  
0.80       0.7407   0.4348   0.7122   0.9996   0.9974   0.2780  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7407, F1=0.4348, Normal Recall=0.7122, Normal Precision=0.9996, Attack Recall=0.9974, Attack Precision=0.2780

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
0.15       0.7695   0.6338   0.7126   0.9990   0.9972   0.4645   <--
0.20       0.7695   0.6338   0.7126   0.9990   0.9972   0.4645  
0.25       0.7695   0.6338   0.7126   0.9990   0.9972   0.4645  
0.30       0.7695   0.6338   0.7126   0.9990   0.9972   0.4645  
0.35       0.7695   0.6338   0.7126   0.9990   0.9972   0.4645  
0.40       0.7695   0.6338   0.7126   0.9990   0.9972   0.4645  
0.45       0.7695   0.6338   0.7126   0.9990   0.9972   0.4645  
0.50       0.7695   0.6338   0.7126   0.9990   0.9972   0.4645  
0.55       0.7695   0.6338   0.7126   0.9990   0.9972   0.4645  
0.60       0.7695   0.6338   0.7126   0.9990   0.9972   0.4645  
0.65       0.7695   0.6338   0.7126   0.9990   0.9972   0.4645  
0.70       0.7695   0.6338   0.7126   0.9990   0.9972   0.4645  
0.75       0.7695   0.6338   0.7126   0.9990   0.9972   0.4645  
0.80       0.7695   0.6338   0.7126   0.9990   0.9972   0.4645  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7695, F1=0.6338, Normal Recall=0.7126, Normal Precision=0.9990, Attack Recall=0.9972, Attack Precision=0.4645

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
0.15       0.7973   0.7469   0.7116   0.9983   0.9972   0.5971   <--
0.20       0.7973   0.7469   0.7116   0.9983   0.9972   0.5971  
0.25       0.7973   0.7469   0.7116   0.9983   0.9972   0.5971  
0.30       0.7973   0.7469   0.7116   0.9983   0.9972   0.5971  
0.35       0.7973   0.7469   0.7116   0.9983   0.9972   0.5971  
0.40       0.7973   0.7469   0.7116   0.9983   0.9972   0.5971  
0.45       0.7973   0.7469   0.7116   0.9983   0.9972   0.5971  
0.50       0.7973   0.7469   0.7116   0.9983   0.9972   0.5971  
0.55       0.7973   0.7469   0.7116   0.9983   0.9972   0.5971  
0.60       0.7973   0.7469   0.7116   0.9983   0.9972   0.5971  
0.65       0.7973   0.7469   0.7116   0.9983   0.9972   0.5971  
0.70       0.7973   0.7469   0.7116   0.9983   0.9972   0.5971  
0.75       0.7973   0.7469   0.7116   0.9983   0.9972   0.5971  
0.80       0.7973   0.7469   0.7116   0.9983   0.9972   0.5971  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7973, F1=0.7469, Normal Recall=0.7116, Normal Precision=0.9983, Attack Recall=0.9972, Attack Precision=0.5971

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
0.15       0.8262   0.8211   0.7122   0.9974   0.9972   0.6979   <--
0.20       0.8262   0.8211   0.7122   0.9974   0.9972   0.6979  
0.25       0.8262   0.8211   0.7122   0.9974   0.9972   0.6979  
0.30       0.8262   0.8211   0.7122   0.9974   0.9972   0.6979  
0.35       0.8262   0.8211   0.7122   0.9974   0.9972   0.6979  
0.40       0.8262   0.8211   0.7122   0.9974   0.9972   0.6979  
0.45       0.8262   0.8211   0.7122   0.9974   0.9972   0.6979  
0.50       0.8262   0.8211   0.7122   0.9974   0.9972   0.6979  
0.55       0.8262   0.8211   0.7122   0.9974   0.9972   0.6979  
0.60       0.8262   0.8211   0.7122   0.9974   0.9972   0.6979  
0.65       0.8262   0.8211   0.7122   0.9974   0.9972   0.6979  
0.70       0.8262   0.8211   0.7122   0.9974   0.9972   0.6979  
0.75       0.8262   0.8211   0.7122   0.9974   0.9972   0.6979  
0.80       0.8262   0.8211   0.7122   0.9974   0.9972   0.6979  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8262, F1=0.8211, Normal Recall=0.7122, Normal Precision=0.9974, Attack Recall=0.9972, Attack Precision=0.6979

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
0.15       0.8543   0.8725   0.7113   0.9961   0.9972   0.7755   <--
0.20       0.8543   0.8725   0.7113   0.9961   0.9972   0.7755  
0.25       0.8543   0.8725   0.7113   0.9961   0.9972   0.7755  
0.30       0.8543   0.8725   0.7113   0.9961   0.9972   0.7755  
0.35       0.8543   0.8725   0.7113   0.9961   0.9972   0.7755  
0.40       0.8543   0.8725   0.7113   0.9961   0.9972   0.7755  
0.45       0.8543   0.8725   0.7113   0.9961   0.9972   0.7755  
0.50       0.8543   0.8725   0.7113   0.9961   0.9972   0.7755  
0.55       0.8543   0.8725   0.7113   0.9961   0.9972   0.7755  
0.60       0.8543   0.8725   0.7113   0.9961   0.9972   0.7755  
0.65       0.8543   0.8725   0.7113   0.9961   0.9972   0.7755  
0.70       0.8543   0.8725   0.7113   0.9961   0.9972   0.7755  
0.75       0.8543   0.8725   0.7113   0.9961   0.9972   0.7755  
0.80       0.8543   0.8725   0.7113   0.9961   0.9972   0.7755  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8543, F1=0.8725, Normal Recall=0.7113, Normal Precision=0.9961, Attack Recall=0.9972, Attack Precision=0.7755

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
0.15       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810   <--
0.20       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.25       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.30       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.35       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.40       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.45       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.50       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.55       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.60       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.65       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.70       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.75       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.80       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7445, F1=0.4385, Normal Recall=0.7165, Normal Precision=0.9996, Attack Recall=0.9974, Attack Precision=0.2810

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
0.15       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682   <--
0.20       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.25       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.30       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.35       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.40       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.45       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.50       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.55       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.60       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.65       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.70       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.75       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.80       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7729, F1=0.6372, Normal Recall=0.7168, Normal Precision=0.9990, Attack Recall=0.9972, Attack Precision=0.4682

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
0.15       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007   <--
0.20       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.25       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.30       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.35       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.40       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.45       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.50       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.55       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.60       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.65       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.70       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.75       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.80       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8003, F1=0.7498, Normal Recall=0.7160, Normal Precision=0.9983, Attack Recall=0.9972, Attack Precision=0.6007

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
0.15       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010   <--
0.20       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.25       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.30       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.35       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.40       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.45       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.50       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.55       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.60       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.65       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.70       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.75       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.80       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8287, F1=0.8233, Normal Recall=0.7164, Normal Precision=0.9974, Attack Recall=0.9972, Attack Precision=0.7010

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
0.15       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780   <--
0.20       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.25       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.30       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.35       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.40       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.45       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.50       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.55       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.60       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.65       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.70       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.75       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.80       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8563, F1=0.8741, Normal Recall=0.7154, Normal Precision=0.9961, Attack Recall=0.9972, Attack Precision=0.7780

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
0.15       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810   <--
0.20       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.25       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.30       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.35       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.40       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.45       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.50       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.55       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.60       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.65       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.70       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.75       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
0.80       0.7445   0.4385   0.7165   0.9996   0.9974   0.2810  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7445, F1=0.4385, Normal Recall=0.7165, Normal Precision=0.9996, Attack Recall=0.9974, Attack Precision=0.2810

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
0.15       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682   <--
0.20       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.25       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.30       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.35       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.40       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.45       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.50       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.55       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.60       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.65       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.70       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.75       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
0.80       0.7729   0.6372   0.7168   0.9990   0.9972   0.4682  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7729, F1=0.6372, Normal Recall=0.7168, Normal Precision=0.9990, Attack Recall=0.9972, Attack Precision=0.4682

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
0.15       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007   <--
0.20       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.25       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.30       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.35       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.40       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.45       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.50       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.55       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.60       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.65       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.70       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.75       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
0.80       0.8003   0.7498   0.7160   0.9983   0.9972   0.6007  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8003, F1=0.7498, Normal Recall=0.7160, Normal Precision=0.9983, Attack Recall=0.9972, Attack Precision=0.6007

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
0.15       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010   <--
0.20       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.25       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.30       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.35       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.40       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.45       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.50       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.55       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.60       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.65       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.70       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.75       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
0.80       0.8287   0.8233   0.7164   0.9974   0.9972   0.7010  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8287, F1=0.8233, Normal Recall=0.7164, Normal Precision=0.9974, Attack Recall=0.9972, Attack Precision=0.7010

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
0.15       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780   <--
0.20       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.25       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.30       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.35       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.40       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.45       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.50       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.55       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.60       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.65       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.70       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.75       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
0.80       0.8563   0.8741   0.7154   0.9961   0.9972   0.7780  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8563, F1=0.8741, Normal Recall=0.7154, Normal Precision=0.9961, Attack Recall=0.9972, Attack Precision=0.7780

```

