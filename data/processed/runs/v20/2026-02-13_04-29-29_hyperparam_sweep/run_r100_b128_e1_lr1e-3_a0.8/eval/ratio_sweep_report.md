# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e1_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-21 08:32:29 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 1 |
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2296 | 0.3048 | 0.3800 | 0.4554 | 0.5293 | 0.6059 | 0.6802 | 0.7555 | 0.8311 | 0.9065 | 0.9811 |
| QAT+Prune only | 0.7729 | 0.7936 | 0.8139 | 0.8340 | 0.8557 | 0.8735 | 0.8951 | 0.9160 | 0.9370 | 0.9563 | 0.9776 |
| QAT+PTQ | 0.7628 | 0.7856 | 0.8084 | 0.8313 | 0.8554 | 0.8758 | 0.8998 | 0.9236 | 0.9468 | 0.9686 | 0.9923 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7628 | 0.7856 | 0.8084 | 0.8313 | 0.8554 | 0.8758 | 0.8998 | 0.9236 | 0.9468 | 0.9686 | 0.9923 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2200 | 0.3876 | 0.5194 | 0.6251 | 0.7134 | 0.7864 | 0.8489 | 0.9028 | 0.9497 | 0.9905 |
| QAT+Prune only | 0.0000 | 0.4867 | 0.6776 | 0.7795 | 0.8442 | 0.8854 | 0.9179 | 0.9422 | 0.9613 | 0.9758 | 0.9886 |
| QAT+PTQ | 0.0000 | 0.4808 | 0.6744 | 0.7792 | 0.8459 | 0.8888 | 0.9224 | 0.9479 | 0.9676 | 0.9827 | 0.9961 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4808 | 0.6744 | 0.7792 | 0.8459 | 0.8888 | 0.9224 | 0.9479 | 0.9676 | 0.9827 | 0.9961 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2296 | 0.2297 | 0.2297 | 0.2301 | 0.2281 | 0.2306 | 0.2289 | 0.2290 | 0.2309 | 0.2343 | 0.0000 |
| QAT+Prune only | 0.7729 | 0.7731 | 0.7730 | 0.7725 | 0.7744 | 0.7694 | 0.7713 | 0.7725 | 0.7750 | 0.7654 | 0.0000 |
| QAT+PTQ | 0.7628 | 0.7626 | 0.7624 | 0.7623 | 0.7642 | 0.7594 | 0.7611 | 0.7635 | 0.7647 | 0.7552 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7628 | 0.7626 | 0.7624 | 0.7623 | 0.7642 | 0.7594 | 0.7611 | 0.7635 | 0.7647 | 0.7552 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2296 | 0.0000 | 0.0000 | 0.0000 | 0.2296 | 1.0000 |
| 90 | 10 | 299,940 | 0.3048 | 0.1239 | 0.9806 | 0.2200 | 0.2297 | 0.9907 |
| 80 | 20 | 291,350 | 0.3800 | 0.2415 | 0.9811 | 0.3876 | 0.2297 | 0.9799 |
| 70 | 30 | 194,230 | 0.4554 | 0.3532 | 0.9811 | 0.5194 | 0.2301 | 0.9660 |
| 60 | 40 | 145,675 | 0.5293 | 0.4587 | 0.9811 | 0.6251 | 0.2281 | 0.9477 |
| 50 | 50 | 116,540 | 0.6059 | 0.5605 | 0.9811 | 0.7134 | 0.2306 | 0.9243 |
| 40 | 60 | 97,115 | 0.6802 | 0.6562 | 0.9811 | 0.7864 | 0.2289 | 0.8899 |
| 30 | 70 | 83,240 | 0.7555 | 0.7481 | 0.9811 | 0.8489 | 0.2290 | 0.8387 |
| 20 | 80 | 72,835 | 0.8311 | 0.8361 | 0.9811 | 0.9028 | 0.2309 | 0.7535 |
| 10 | 90 | 64,740 | 0.9065 | 0.9202 | 0.9811 | 0.9497 | 0.2343 | 0.5799 |
| 0 | 100 | 58,270 | 0.9811 | 1.0000 | 0.9811 | 0.9905 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7729 | 0.0000 | 0.0000 | 0.0000 | 0.7729 | 1.0000 |
| 90 | 10 | 299,940 | 0.7936 | 0.3239 | 0.9784 | 0.4867 | 0.7731 | 0.9969 |
| 80 | 20 | 291,350 | 0.8139 | 0.5185 | 0.9776 | 0.6776 | 0.7730 | 0.9928 |
| 70 | 30 | 194,230 | 0.8340 | 0.6481 | 0.9776 | 0.7795 | 0.7725 | 0.9877 |
| 60 | 40 | 145,675 | 0.8557 | 0.7429 | 0.9776 | 0.8442 | 0.7744 | 0.9810 |
| 50 | 50 | 116,540 | 0.8735 | 0.8091 | 0.9776 | 0.8854 | 0.7694 | 0.9717 |
| 40 | 60 | 97,115 | 0.8951 | 0.8651 | 0.9776 | 0.9179 | 0.7713 | 0.9582 |
| 30 | 70 | 83,240 | 0.9160 | 0.9093 | 0.9776 | 0.9422 | 0.7725 | 0.9365 |
| 20 | 80 | 72,835 | 0.9370 | 0.9456 | 0.9776 | 0.9613 | 0.7750 | 0.8962 |
| 10 | 90 | 64,740 | 0.9563 | 0.9740 | 0.9776 | 0.9758 | 0.7654 | 0.7912 |
| 0 | 100 | 58,270 | 0.9776 | 1.0000 | 0.9776 | 0.9886 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7628 | 0.0000 | 0.0000 | 0.0000 | 0.7628 | 1.0000 |
| 90 | 10 | 299,940 | 0.7856 | 0.3172 | 0.9924 | 0.4808 | 0.7626 | 0.9989 |
| 80 | 20 | 291,350 | 0.8084 | 0.5108 | 0.9923 | 0.6744 | 0.7624 | 0.9975 |
| 70 | 30 | 194,230 | 0.8313 | 0.6415 | 0.9923 | 0.7792 | 0.7623 | 0.9957 |
| 60 | 40 | 145,675 | 0.8554 | 0.7372 | 0.9923 | 0.8459 | 0.7642 | 0.9933 |
| 50 | 50 | 116,540 | 0.8758 | 0.8048 | 0.9923 | 0.8888 | 0.7594 | 0.9899 |
| 40 | 60 | 97,115 | 0.8998 | 0.8617 | 0.9923 | 0.9224 | 0.7611 | 0.9850 |
| 30 | 70 | 83,240 | 0.9236 | 0.9073 | 0.9923 | 0.9479 | 0.7635 | 0.9769 |
| 20 | 80 | 72,835 | 0.9468 | 0.9440 | 0.9923 | 0.9676 | 0.7647 | 0.9611 |
| 10 | 90 | 64,740 | 0.9686 | 0.9733 | 0.9923 | 0.9827 | 0.7552 | 0.9155 |
| 0 | 100 | 58,270 | 0.9923 | 1.0000 | 0.9923 | 0.9961 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7628 | 0.0000 | 0.0000 | 0.0000 | 0.7628 | 1.0000 |
| 90 | 10 | 299,940 | 0.7856 | 0.3172 | 0.9924 | 0.4808 | 0.7626 | 0.9989 |
| 80 | 20 | 291,350 | 0.8084 | 0.5108 | 0.9923 | 0.6744 | 0.7624 | 0.9975 |
| 70 | 30 | 194,230 | 0.8313 | 0.6415 | 0.9923 | 0.7792 | 0.7623 | 0.9957 |
| 60 | 40 | 145,675 | 0.8554 | 0.7372 | 0.9923 | 0.8459 | 0.7642 | 0.9933 |
| 50 | 50 | 116,540 | 0.8758 | 0.8048 | 0.9923 | 0.8888 | 0.7594 | 0.9899 |
| 40 | 60 | 97,115 | 0.8998 | 0.8617 | 0.9923 | 0.9224 | 0.7611 | 0.9850 |
| 30 | 70 | 83,240 | 0.9236 | 0.9073 | 0.9923 | 0.9479 | 0.7635 | 0.9769 |
| 20 | 80 | 72,835 | 0.9468 | 0.9440 | 0.9923 | 0.9676 | 0.7647 | 0.9611 |
| 10 | 90 | 64,740 | 0.9686 | 0.9733 | 0.9923 | 0.9827 | 0.7552 | 0.9155 |
| 0 | 100 | 58,270 | 0.9923 | 1.0000 | 0.9923 | 0.9961 | 0.0000 | 0.0000 |


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
0.15       0.3048   0.2199   0.2297   0.9904   0.9799   0.1238   <--
0.20       0.3048   0.2199   0.2297   0.9904   0.9799   0.1238  
0.25       0.3048   0.2199   0.2297   0.9904   0.9799   0.1238  
0.30       0.3048   0.2199   0.2297   0.9904   0.9799   0.1238  
0.35       0.3048   0.2199   0.2297   0.9904   0.9799   0.1238  
0.40       0.3048   0.2199   0.2297   0.9904   0.9799   0.1238  
0.45       0.3048   0.2199   0.2297   0.9904   0.9799   0.1238  
0.50       0.3048   0.2199   0.2297   0.9904   0.9799   0.1238  
0.55       0.3048   0.2199   0.2297   0.9904   0.9799   0.1238  
0.60       0.3048   0.2199   0.2297   0.9904   0.9799   0.1238  
0.65       0.3048   0.2199   0.2297   0.9904   0.9799   0.1238  
0.70       0.3048   0.2199   0.2297   0.9904   0.9799   0.1238  
0.75       0.3048   0.2199   0.2297   0.9904   0.9799   0.1238  
0.80       0.3048   0.2199   0.2297   0.9904   0.9799   0.1238  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3048, F1=0.2199, Normal Recall=0.2297, Normal Precision=0.9904, Attack Recall=0.9799, Attack Precision=0.1238

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
0.15       0.3796   0.3875   0.2292   0.9798   0.9811   0.2414   <--
0.20       0.3796   0.3875   0.2292   0.9798   0.9811   0.2414  
0.25       0.3796   0.3875   0.2292   0.9798   0.9811   0.2414  
0.30       0.3796   0.3875   0.2292   0.9798   0.9811   0.2414  
0.35       0.3796   0.3875   0.2292   0.9798   0.9811   0.2414  
0.40       0.3796   0.3875   0.2292   0.9798   0.9811   0.2414  
0.45       0.3796   0.3875   0.2292   0.9798   0.9811   0.2414  
0.50       0.3796   0.3875   0.2292   0.9798   0.9811   0.2414  
0.55       0.3796   0.3875   0.2292   0.9798   0.9811   0.2414  
0.60       0.3796   0.3875   0.2292   0.9798   0.9811   0.2414  
0.65       0.3796   0.3875   0.2292   0.9798   0.9811   0.2414  
0.70       0.3796   0.3875   0.2292   0.9798   0.9811   0.2414  
0.75       0.3796   0.3875   0.2292   0.9798   0.9811   0.2414  
0.80       0.3796   0.3875   0.2292   0.9798   0.9811   0.2414  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3796, F1=0.3875, Normal Recall=0.2292, Normal Precision=0.9798, Attack Recall=0.9811, Attack Precision=0.2414

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
0.15       0.4557   0.5196   0.2305   0.9661   0.9811   0.3534   <--
0.20       0.4557   0.5196   0.2305   0.9661   0.9811   0.3534  
0.25       0.4557   0.5196   0.2305   0.9661   0.9811   0.3534  
0.30       0.4557   0.5196   0.2305   0.9661   0.9811   0.3534  
0.35       0.4557   0.5196   0.2305   0.9661   0.9811   0.3534  
0.40       0.4557   0.5196   0.2305   0.9661   0.9811   0.3534  
0.45       0.4557   0.5196   0.2305   0.9661   0.9811   0.3534  
0.50       0.4557   0.5196   0.2305   0.9661   0.9811   0.3534  
0.55       0.4557   0.5196   0.2305   0.9661   0.9811   0.3534  
0.60       0.4557   0.5196   0.2305   0.9661   0.9811   0.3534  
0.65       0.4557   0.5196   0.2305   0.9661   0.9811   0.3534  
0.70       0.4557   0.5196   0.2305   0.9661   0.9811   0.3534  
0.75       0.4557   0.5196   0.2305   0.9661   0.9811   0.3534  
0.80       0.4557   0.5196   0.2305   0.9661   0.9811   0.3534  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4557, F1=0.5196, Normal Recall=0.2305, Normal Precision=0.9661, Attack Recall=0.9811, Attack Precision=0.3534

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
0.15       0.5303   0.6256   0.2297   0.9481   0.9811   0.4592   <--
0.20       0.5303   0.6256   0.2297   0.9481   0.9811   0.4592  
0.25       0.5303   0.6256   0.2297   0.9481   0.9811   0.4592  
0.30       0.5303   0.6256   0.2297   0.9481   0.9811   0.4592  
0.35       0.5303   0.6256   0.2297   0.9481   0.9811   0.4592  
0.40       0.5303   0.6256   0.2297   0.9481   0.9811   0.4592  
0.45       0.5303   0.6256   0.2297   0.9481   0.9811   0.4592  
0.50       0.5303   0.6256   0.2297   0.9481   0.9811   0.4592  
0.55       0.5303   0.6256   0.2297   0.9481   0.9811   0.4592  
0.60       0.5303   0.6256   0.2297   0.9481   0.9811   0.4592  
0.65       0.5303   0.6256   0.2297   0.9481   0.9811   0.4592  
0.70       0.5303   0.6256   0.2297   0.9481   0.9811   0.4592  
0.75       0.5303   0.6256   0.2297   0.9481   0.9811   0.4592  
0.80       0.5303   0.6256   0.2297   0.9481   0.9811   0.4592  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5303, F1=0.6256, Normal Recall=0.2297, Normal Precision=0.9481, Attack Recall=0.9811, Attack Precision=0.4592

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
0.15       0.6056   0.7133   0.2301   0.9242   0.9811   0.5603   <--
0.20       0.6056   0.7133   0.2301   0.9242   0.9811   0.5603  
0.25       0.6056   0.7133   0.2301   0.9242   0.9811   0.5603  
0.30       0.6056   0.7133   0.2301   0.9242   0.9811   0.5603  
0.35       0.6056   0.7133   0.2301   0.9242   0.9811   0.5603  
0.40       0.6056   0.7133   0.2301   0.9242   0.9811   0.5603  
0.45       0.6056   0.7133   0.2301   0.9242   0.9811   0.5603  
0.50       0.6056   0.7133   0.2301   0.9242   0.9811   0.5603  
0.55       0.6056   0.7133   0.2301   0.9242   0.9811   0.5603  
0.60       0.6056   0.7133   0.2301   0.9242   0.9811   0.5603  
0.65       0.6056   0.7133   0.2301   0.9242   0.9811   0.5603  
0.70       0.6056   0.7133   0.2301   0.9242   0.9811   0.5603  
0.75       0.6056   0.7133   0.2301   0.9242   0.9811   0.5603  
0.80       0.6056   0.7133   0.2301   0.9242   0.9811   0.5603  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6056, F1=0.7133, Normal Recall=0.2301, Normal Precision=0.9242, Attack Recall=0.9811, Attack Precision=0.5603

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
0.15       0.7936   0.4866   0.7731   0.9969   0.9781   0.3239   <--
0.20       0.7936   0.4866   0.7731   0.9969   0.9781   0.3239  
0.25       0.7936   0.4866   0.7731   0.9969   0.9781   0.3239  
0.30       0.7936   0.4866   0.7731   0.9969   0.9781   0.3239  
0.35       0.7936   0.4866   0.7731   0.9969   0.9781   0.3239  
0.40       0.7936   0.4866   0.7731   0.9969   0.9781   0.3239  
0.45       0.7936   0.4866   0.7731   0.9969   0.9781   0.3239  
0.50       0.7936   0.4866   0.7731   0.9969   0.9781   0.3239  
0.55       0.7936   0.4866   0.7731   0.9969   0.9781   0.3239  
0.60       0.7936   0.4866   0.7731   0.9969   0.9781   0.3239  
0.65       0.7936   0.4866   0.7731   0.9969   0.9781   0.3239  
0.70       0.7936   0.4866   0.7731   0.9969   0.9781   0.3239  
0.75       0.7936   0.4866   0.7731   0.9969   0.9781   0.3239  
0.80       0.7936   0.4866   0.7731   0.9969   0.9781   0.3239  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7936, F1=0.4866, Normal Recall=0.7731, Normal Precision=0.9969, Attack Recall=0.9781, Attack Precision=0.3239

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
0.15       0.8140   0.6777   0.7732   0.9928   0.9776   0.5186   <--
0.20       0.8140   0.6777   0.7732   0.9928   0.9776   0.5186  
0.25       0.8140   0.6777   0.7732   0.9928   0.9776   0.5186  
0.30       0.8140   0.6777   0.7732   0.9928   0.9776   0.5186  
0.35       0.8140   0.6777   0.7732   0.9928   0.9776   0.5186  
0.40       0.8140   0.6777   0.7732   0.9928   0.9776   0.5186  
0.45       0.8140   0.6777   0.7732   0.9928   0.9776   0.5186  
0.50       0.8140   0.6777   0.7732   0.9928   0.9776   0.5186  
0.55       0.8140   0.6777   0.7732   0.9928   0.9776   0.5186  
0.60       0.8140   0.6777   0.7732   0.9928   0.9776   0.5186  
0.65       0.8140   0.6777   0.7732   0.9928   0.9776   0.5186  
0.70       0.8140   0.6777   0.7732   0.9928   0.9776   0.5186  
0.75       0.8140   0.6777   0.7732   0.9928   0.9776   0.5186  
0.80       0.8140   0.6777   0.7732   0.9928   0.9776   0.5186  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8140, F1=0.6777, Normal Recall=0.7732, Normal Precision=0.9928, Attack Recall=0.9776, Attack Precision=0.5186

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
0.15       0.8342   0.7796   0.7728   0.9877   0.9776   0.6483   <--
0.20       0.8342   0.7796   0.7728   0.9877   0.9776   0.6483  
0.25       0.8342   0.7796   0.7728   0.9877   0.9776   0.6483  
0.30       0.8342   0.7796   0.7728   0.9877   0.9776   0.6483  
0.35       0.8342   0.7796   0.7728   0.9877   0.9776   0.6483  
0.40       0.8342   0.7796   0.7728   0.9877   0.9776   0.6483  
0.45       0.8342   0.7796   0.7728   0.9877   0.9776   0.6483  
0.50       0.8342   0.7796   0.7728   0.9877   0.9776   0.6483  
0.55       0.8342   0.7796   0.7728   0.9877   0.9776   0.6483  
0.60       0.8342   0.7796   0.7728   0.9877   0.9776   0.6483  
0.65       0.8342   0.7796   0.7728   0.9877   0.9776   0.6483  
0.70       0.8342   0.7796   0.7728   0.9877   0.9776   0.6483  
0.75       0.8342   0.7796   0.7728   0.9877   0.9776   0.6483  
0.80       0.8342   0.7796   0.7728   0.9877   0.9776   0.6483  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8342, F1=0.7796, Normal Recall=0.7728, Normal Precision=0.9877, Attack Recall=0.9776, Attack Precision=0.6483

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
0.15       0.8548   0.8434   0.7730   0.9810   0.9776   0.7416   <--
0.20       0.8548   0.8434   0.7730   0.9810   0.9776   0.7416  
0.25       0.8548   0.8434   0.7730   0.9810   0.9776   0.7416  
0.30       0.8548   0.8434   0.7730   0.9810   0.9776   0.7416  
0.35       0.8548   0.8434   0.7730   0.9810   0.9776   0.7416  
0.40       0.8548   0.8434   0.7730   0.9810   0.9776   0.7416  
0.45       0.8548   0.8434   0.7730   0.9810   0.9776   0.7416  
0.50       0.8548   0.8434   0.7730   0.9810   0.9776   0.7416  
0.55       0.8548   0.8434   0.7730   0.9810   0.9776   0.7416  
0.60       0.8548   0.8434   0.7730   0.9810   0.9776   0.7416  
0.65       0.8548   0.8434   0.7730   0.9810   0.9776   0.7416  
0.70       0.8548   0.8434   0.7730   0.9810   0.9776   0.7416  
0.75       0.8548   0.8434   0.7730   0.9810   0.9776   0.7416  
0.80       0.8548   0.8434   0.7730   0.9810   0.9776   0.7416  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8548, F1=0.8434, Normal Recall=0.7730, Normal Precision=0.9810, Attack Recall=0.9776, Attack Precision=0.7416

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
0.15       0.8742   0.8860   0.7709   0.9717   0.9776   0.8101   <--
0.20       0.8742   0.8860   0.7709   0.9717   0.9776   0.8101  
0.25       0.8742   0.8860   0.7709   0.9717   0.9776   0.8101  
0.30       0.8742   0.8860   0.7709   0.9717   0.9776   0.8101  
0.35       0.8742   0.8860   0.7709   0.9717   0.9776   0.8101  
0.40       0.8742   0.8860   0.7709   0.9717   0.9776   0.8101  
0.45       0.8742   0.8860   0.7709   0.9717   0.9776   0.8101  
0.50       0.8742   0.8860   0.7709   0.9717   0.9776   0.8101  
0.55       0.8742   0.8860   0.7709   0.9717   0.9776   0.8101  
0.60       0.8742   0.8860   0.7709   0.9717   0.9776   0.8101  
0.65       0.8742   0.8860   0.7709   0.9717   0.9776   0.8101  
0.70       0.8742   0.8860   0.7709   0.9717   0.9776   0.8101  
0.75       0.8742   0.8860   0.7709   0.9717   0.9776   0.8101  
0.80       0.8742   0.8860   0.7709   0.9717   0.9776   0.8101  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8742, F1=0.8860, Normal Recall=0.7709, Normal Precision=0.9717, Attack Recall=0.9776, Attack Precision=0.8101

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
0.15       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173   <--
0.20       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.25       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.30       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.35       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.40       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.45       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.50       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.55       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.60       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.65       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.70       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.75       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.80       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7857, F1=0.4810, Normal Recall=0.7626, Normal Precision=0.9990, Attack Recall=0.9930, Attack Precision=0.3173

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
0.15       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112   <--
0.20       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.25       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.30       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.35       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.40       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.45       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.50       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.55       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.60       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.65       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.70       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.75       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.80       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8087, F1=0.6748, Normal Recall=0.7628, Normal Precision=0.9975, Attack Recall=0.9923, Attack Precision=0.5112

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
0.15       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419   <--
0.20       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.25       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.30       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.35       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.40       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.45       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.50       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.55       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.60       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.65       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.70       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.75       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.80       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8316, F1=0.7795, Normal Recall=0.7628, Normal Precision=0.9957, Attack Recall=0.9923, Attack Precision=0.6419

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
0.15       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362   <--
0.20       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.25       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.30       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.35       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.40       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.45       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.50       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.55       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.60       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.65       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.70       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.75       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.80       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8547, F1=0.8452, Normal Recall=0.7629, Normal Precision=0.9933, Attack Recall=0.9923, Attack Precision=0.7362

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
0.15       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059   <--
0.20       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.25       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.30       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.35       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.40       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.45       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.50       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.55       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.60       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.65       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.70       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.75       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.80       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8767, F1=0.8894, Normal Recall=0.7611, Normal Precision=0.9899, Attack Recall=0.9923, Attack Precision=0.8059

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
0.15       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173   <--
0.20       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.25       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.30       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.35       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.40       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.45       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.50       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.55       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.60       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.65       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.70       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.75       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
0.80       0.7857   0.4810   0.7626   0.9990   0.9930   0.3173  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7857, F1=0.4810, Normal Recall=0.7626, Normal Precision=0.9990, Attack Recall=0.9930, Attack Precision=0.3173

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
0.15       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112   <--
0.20       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.25       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.30       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.35       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.40       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.45       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.50       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.55       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.60       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.65       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.70       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.75       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
0.80       0.8087   0.6748   0.7628   0.9975   0.9923   0.5112  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8087, F1=0.6748, Normal Recall=0.7628, Normal Precision=0.9975, Attack Recall=0.9923, Attack Precision=0.5112

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
0.15       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419   <--
0.20       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.25       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.30       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.35       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.40       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.45       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.50       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.55       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.60       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.65       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.70       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.75       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
0.80       0.8316   0.7795   0.7628   0.9957   0.9923   0.6419  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8316, F1=0.7795, Normal Recall=0.7628, Normal Precision=0.9957, Attack Recall=0.9923, Attack Precision=0.6419

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
0.15       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362   <--
0.20       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.25       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.30       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.35       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.40       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.45       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.50       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.55       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.60       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.65       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.70       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.75       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
0.80       0.8547   0.8452   0.7629   0.9933   0.9923   0.7362  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8547, F1=0.8452, Normal Recall=0.7629, Normal Precision=0.9933, Attack Recall=0.9923, Attack Precision=0.7362

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
0.15       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059   <--
0.20       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.25       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.30       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.35       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.40       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.45       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.50       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.55       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.60       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.65       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.70       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.75       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
0.80       0.8767   0.8894   0.7611   0.9899   0.9923   0.8059  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8767, F1=0.8894, Normal Recall=0.7611, Normal Precision=0.9899, Attack Recall=0.9923, Attack Precision=0.8059

```

