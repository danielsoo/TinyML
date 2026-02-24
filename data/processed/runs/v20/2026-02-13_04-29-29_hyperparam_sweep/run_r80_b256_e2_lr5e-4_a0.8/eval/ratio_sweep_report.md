# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e2_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-19 01:37:51 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 80 |
| **Local epochs** | 2 |
| **Batch size** | 256 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4067 | 0.4654 | 0.5242 | 0.5837 | 0.6419 | 0.7001 | 0.7604 | 0.8191 | 0.8774 | 0.9366 | 0.9956 |
| QAT+Prune only | 0.9861 | 0.9446 | 0.9032 | 0.8619 | 0.8208 | 0.7792 | 0.7385 | 0.6970 | 0.6556 | 0.6142 | 0.5730 |
| QAT+PTQ | 0.9862 | 0.9447 | 0.9033 | 0.8620 | 0.8209 | 0.7793 | 0.7386 | 0.6973 | 0.6557 | 0.6143 | 0.5732 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9862 | 0.9447 | 0.9033 | 0.8620 | 0.8209 | 0.7793 | 0.7386 | 0.6973 | 0.6557 | 0.6143 | 0.5732 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2714 | 0.4556 | 0.5893 | 0.6898 | 0.7685 | 0.8329 | 0.8851 | 0.9285 | 0.9658 | 0.9978 |
| QAT+Prune only | 0.0000 | 0.6747 | 0.7030 | 0.7135 | 0.7190 | 0.7218 | 0.7245 | 0.7259 | 0.7269 | 0.7278 | 0.7286 |
| QAT+PTQ | 0.0000 | 0.6752 | 0.7033 | 0.7136 | 0.7191 | 0.7220 | 0.7247 | 0.7261 | 0.7271 | 0.7279 | 0.7287 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.6752 | 0.7033 | 0.7136 | 0.7191 | 0.7220 | 0.7247 | 0.7261 | 0.7271 | 0.7279 | 0.7287 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4067 | 0.4065 | 0.4064 | 0.4071 | 0.4061 | 0.4045 | 0.4076 | 0.4074 | 0.4047 | 0.4053 | 0.0000 |
| QAT+Prune only | 0.9861 | 0.9858 | 0.9857 | 0.9858 | 0.9860 | 0.9853 | 0.9868 | 0.9863 | 0.9859 | 0.9844 | 0.0000 |
| QAT+PTQ | 0.9862 | 0.9858 | 0.9858 | 0.9858 | 0.9860 | 0.9855 | 0.9868 | 0.9867 | 0.9859 | 0.9847 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9862 | 0.9858 | 0.9858 | 0.9858 | 0.9860 | 0.9855 | 0.9868 | 0.9867 | 0.9859 | 0.9847 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4067 | 0.0000 | 0.0000 | 0.0000 | 0.4067 | 1.0000 |
| 90 | 10 | 299,940 | 0.4654 | 0.1571 | 0.9959 | 0.2714 | 0.4065 | 0.9989 |
| 80 | 20 | 291,350 | 0.5242 | 0.2954 | 0.9956 | 0.4556 | 0.4064 | 0.9973 |
| 70 | 30 | 194,230 | 0.5837 | 0.4185 | 0.9956 | 0.5893 | 0.4071 | 0.9954 |
| 60 | 40 | 145,675 | 0.6419 | 0.5278 | 0.9956 | 0.6898 | 0.4061 | 0.9928 |
| 50 | 50 | 116,540 | 0.7001 | 0.6257 | 0.9956 | 0.7685 | 0.4045 | 0.9892 |
| 40 | 60 | 97,115 | 0.7604 | 0.7160 | 0.9956 | 0.8329 | 0.4076 | 0.9840 |
| 30 | 70 | 83,240 | 0.8191 | 0.7967 | 0.9956 | 0.8851 | 0.4074 | 0.9754 |
| 20 | 80 | 72,835 | 0.8774 | 0.8700 | 0.9956 | 0.9285 | 0.4047 | 0.9582 |
| 10 | 90 | 64,740 | 0.9366 | 0.9378 | 0.9956 | 0.9658 | 0.4053 | 0.9108 |
| 0 | 100 | 58,270 | 0.9956 | 1.0000 | 0.9956 | 0.9978 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9861 | 0.0000 | 0.0000 | 0.0000 | 0.9861 | 1.0000 |
| 90 | 10 | 299,940 | 0.9446 | 0.8179 | 0.5742 | 0.6747 | 0.9858 | 0.9542 |
| 80 | 20 | 291,350 | 0.9032 | 0.9093 | 0.5730 | 0.7030 | 0.9857 | 0.9023 |
| 70 | 30 | 194,230 | 0.8619 | 0.9452 | 0.5730 | 0.7135 | 0.9858 | 0.8434 |
| 60 | 40 | 145,675 | 0.8208 | 0.9646 | 0.5730 | 0.7190 | 0.9860 | 0.7760 |
| 50 | 50 | 116,540 | 0.7792 | 0.9749 | 0.5730 | 0.7218 | 0.9853 | 0.6977 |
| 40 | 60 | 97,115 | 0.7385 | 0.9848 | 0.5730 | 0.7245 | 0.9868 | 0.6064 |
| 30 | 70 | 83,240 | 0.6970 | 0.9899 | 0.5730 | 0.7259 | 0.9863 | 0.4975 |
| 20 | 80 | 72,835 | 0.6556 | 0.9939 | 0.5730 | 0.7269 | 0.9859 | 0.3660 |
| 10 | 90 | 64,740 | 0.6142 | 0.9970 | 0.5730 | 0.7278 | 0.9844 | 0.2039 |
| 0 | 100 | 58,270 | 0.5730 | 1.0000 | 0.5730 | 0.7286 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9862 | 0.0000 | 0.0000 | 0.0000 | 0.9862 | 1.0000 |
| 90 | 10 | 299,940 | 0.9447 | 0.8185 | 0.5745 | 0.6752 | 0.9858 | 0.9542 |
| 80 | 20 | 291,350 | 0.9033 | 0.9097 | 0.5732 | 0.7033 | 0.9858 | 0.9023 |
| 70 | 30 | 194,230 | 0.8620 | 0.9452 | 0.5732 | 0.7136 | 0.9858 | 0.8435 |
| 60 | 40 | 145,675 | 0.8209 | 0.9646 | 0.5732 | 0.7191 | 0.9860 | 0.7760 |
| 50 | 50 | 116,540 | 0.7793 | 0.9753 | 0.5732 | 0.7220 | 0.9855 | 0.6978 |
| 40 | 60 | 97,115 | 0.7386 | 0.9849 | 0.5732 | 0.7247 | 0.9868 | 0.6065 |
| 30 | 70 | 83,240 | 0.6973 | 0.9902 | 0.5732 | 0.7261 | 0.9867 | 0.4977 |
| 20 | 80 | 72,835 | 0.6557 | 0.9939 | 0.5732 | 0.7271 | 0.9859 | 0.3661 |
| 10 | 90 | 64,740 | 0.6143 | 0.9970 | 0.5732 | 0.7279 | 0.9847 | 0.2040 |
| 0 | 100 | 58,270 | 0.5732 | 1.0000 | 0.5732 | 0.7287 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9862 | 0.0000 | 0.0000 | 0.0000 | 0.9862 | 1.0000 |
| 90 | 10 | 299,940 | 0.9447 | 0.8185 | 0.5745 | 0.6752 | 0.9858 | 0.9542 |
| 80 | 20 | 291,350 | 0.9033 | 0.9097 | 0.5732 | 0.7033 | 0.9858 | 0.9023 |
| 70 | 30 | 194,230 | 0.8620 | 0.9452 | 0.5732 | 0.7136 | 0.9858 | 0.8435 |
| 60 | 40 | 145,675 | 0.8209 | 0.9646 | 0.5732 | 0.7191 | 0.9860 | 0.7760 |
| 50 | 50 | 116,540 | 0.7793 | 0.9753 | 0.5732 | 0.7220 | 0.9855 | 0.6978 |
| 40 | 60 | 97,115 | 0.7386 | 0.9849 | 0.5732 | 0.7247 | 0.9868 | 0.6065 |
| 30 | 70 | 83,240 | 0.6973 | 0.9902 | 0.5732 | 0.7261 | 0.9867 | 0.4977 |
| 20 | 80 | 72,835 | 0.6557 | 0.9939 | 0.5732 | 0.7271 | 0.9859 | 0.3661 |
| 10 | 90 | 64,740 | 0.6143 | 0.9970 | 0.5732 | 0.7279 | 0.9847 | 0.2040 |
| 0 | 100 | 58,270 | 0.5732 | 1.0000 | 0.5732 | 0.7287 | 0.0000 | 0.0000 |


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
0.15       0.4654   0.2715   0.4065   0.9989   0.9960   0.1572   <--
0.20       0.4654   0.2715   0.4065   0.9989   0.9960   0.1572  
0.25       0.4654   0.2715   0.4065   0.9989   0.9960   0.1572  
0.30       0.4654   0.2715   0.4065   0.9989   0.9960   0.1572  
0.35       0.4654   0.2715   0.4065   0.9989   0.9960   0.1572  
0.40       0.4654   0.2715   0.4065   0.9989   0.9960   0.1572  
0.45       0.4654   0.2715   0.4065   0.9989   0.9960   0.1572  
0.50       0.4654   0.2715   0.4065   0.9989   0.9960   0.1572  
0.55       0.4654   0.2715   0.4065   0.9989   0.9960   0.1572  
0.60       0.4654   0.2715   0.4065   0.9989   0.9960   0.1572  
0.65       0.4654   0.2715   0.4065   0.9989   0.9960   0.1572  
0.70       0.4654   0.2715   0.4065   0.9989   0.9960   0.1572  
0.75       0.4654   0.2715   0.4065   0.9989   0.9960   0.1572  
0.80       0.4654   0.2715   0.4065   0.9989   0.9960   0.1572  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4654, F1=0.2715, Normal Recall=0.4065, Normal Precision=0.9989, Attack Recall=0.9960, Attack Precision=0.1572

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
0.15       0.5246   0.4558   0.4069   0.9973   0.9956   0.2956   <--
0.20       0.5246   0.4558   0.4069   0.9973   0.9956   0.2956  
0.25       0.5246   0.4558   0.4069   0.9973   0.9956   0.2956  
0.30       0.5246   0.4558   0.4069   0.9973   0.9956   0.2956  
0.35       0.5246   0.4558   0.4069   0.9973   0.9956   0.2956  
0.40       0.5246   0.4558   0.4069   0.9973   0.9956   0.2956  
0.45       0.5246   0.4558   0.4069   0.9973   0.9956   0.2956  
0.50       0.5246   0.4558   0.4069   0.9973   0.9956   0.2956  
0.55       0.5246   0.4558   0.4069   0.9973   0.9956   0.2956  
0.60       0.5246   0.4558   0.4069   0.9973   0.9956   0.2956  
0.65       0.5246   0.4558   0.4069   0.9973   0.9956   0.2956  
0.70       0.5246   0.4558   0.4069   0.9973   0.9956   0.2956  
0.75       0.5246   0.4558   0.4069   0.9973   0.9956   0.2956  
0.80       0.5246   0.4558   0.4069   0.9973   0.9956   0.2956  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5246, F1=0.4558, Normal Recall=0.4069, Normal Precision=0.9973, Attack Recall=0.9956, Attack Precision=0.2956

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
0.15       0.5842   0.5896   0.4079   0.9954   0.9956   0.4188   <--
0.20       0.5842   0.5896   0.4079   0.9954   0.9956   0.4188  
0.25       0.5842   0.5896   0.4079   0.9954   0.9956   0.4188  
0.30       0.5842   0.5896   0.4079   0.9954   0.9956   0.4188  
0.35       0.5842   0.5896   0.4079   0.9954   0.9956   0.4188  
0.40       0.5842   0.5896   0.4079   0.9954   0.9956   0.4188  
0.45       0.5842   0.5896   0.4079   0.9954   0.9956   0.4188  
0.50       0.5842   0.5896   0.4079   0.9954   0.9956   0.4188  
0.55       0.5842   0.5896   0.4079   0.9954   0.9956   0.4188  
0.60       0.5842   0.5896   0.4079   0.9954   0.9956   0.4188  
0.65       0.5842   0.5896   0.4079   0.9954   0.9956   0.4188  
0.70       0.5842   0.5896   0.4079   0.9954   0.9956   0.4188  
0.75       0.5842   0.5896   0.4079   0.9954   0.9956   0.4188  
0.80       0.5842   0.5896   0.4079   0.9954   0.9956   0.4188  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5842, F1=0.5896, Normal Recall=0.4079, Normal Precision=0.9954, Attack Recall=0.9956, Attack Precision=0.4188

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
0.15       0.6428   0.6904   0.4076   0.9928   0.9956   0.5284   <--
0.20       0.6428   0.6904   0.4076   0.9928   0.9956   0.5284  
0.25       0.6428   0.6904   0.4076   0.9928   0.9956   0.5284  
0.30       0.6428   0.6904   0.4076   0.9928   0.9956   0.5284  
0.35       0.6428   0.6904   0.4076   0.9928   0.9956   0.5284  
0.40       0.6428   0.6904   0.4076   0.9928   0.9956   0.5284  
0.45       0.6428   0.6904   0.4076   0.9928   0.9956   0.5284  
0.50       0.6428   0.6904   0.4076   0.9928   0.9956   0.5284  
0.55       0.6428   0.6904   0.4076   0.9928   0.9956   0.5284  
0.60       0.6428   0.6904   0.4076   0.9928   0.9956   0.5284  
0.65       0.6428   0.6904   0.4076   0.9928   0.9956   0.5284  
0.70       0.6428   0.6904   0.4076   0.9928   0.9956   0.5284  
0.75       0.6428   0.6904   0.4076   0.9928   0.9956   0.5284  
0.80       0.6428   0.6904   0.4076   0.9928   0.9956   0.5284  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6428, F1=0.6904, Normal Recall=0.4076, Normal Precision=0.9928, Attack Recall=0.9956, Attack Precision=0.5284

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
0.15       0.7018   0.7695   0.4079   0.9893   0.9956   0.6271   <--
0.20       0.7018   0.7695   0.4079   0.9893   0.9956   0.6271  
0.25       0.7018   0.7695   0.4079   0.9893   0.9956   0.6271  
0.30       0.7018   0.7695   0.4079   0.9893   0.9956   0.6271  
0.35       0.7018   0.7695   0.4079   0.9893   0.9956   0.6271  
0.40       0.7018   0.7695   0.4079   0.9893   0.9956   0.6271  
0.45       0.7018   0.7695   0.4079   0.9893   0.9956   0.6271  
0.50       0.7018   0.7695   0.4079   0.9893   0.9956   0.6271  
0.55       0.7018   0.7695   0.4079   0.9893   0.9956   0.6271  
0.60       0.7018   0.7695   0.4079   0.9893   0.9956   0.6271  
0.65       0.7018   0.7695   0.4079   0.9893   0.9956   0.6271  
0.70       0.7018   0.7695   0.4079   0.9893   0.9956   0.6271  
0.75       0.7018   0.7695   0.4079   0.9893   0.9956   0.6271  
0.80       0.7018   0.7695   0.4079   0.9893   0.9956   0.6271  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7018, F1=0.7695, Normal Recall=0.4079, Normal Precision=0.9893, Attack Recall=0.9956, Attack Precision=0.6271

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
0.15       0.9449   0.6765   0.9858   0.9544   0.5764   0.8185   <--
0.20       0.9449   0.6765   0.9858   0.9544   0.5764   0.8185  
0.25       0.9449   0.6765   0.9858   0.9544   0.5764   0.8185  
0.30       0.9449   0.6765   0.9858   0.9544   0.5764   0.8185  
0.35       0.9449   0.6765   0.9858   0.9544   0.5764   0.8185  
0.40       0.9449   0.6765   0.9858   0.9544   0.5764   0.8185  
0.45       0.9449   0.6765   0.9858   0.9544   0.5764   0.8185  
0.50       0.9449   0.6765   0.9858   0.9544   0.5764   0.8185  
0.55       0.9449   0.6765   0.9858   0.9544   0.5764   0.8185  
0.60       0.9449   0.6765   0.9858   0.9544   0.5764   0.8185  
0.65       0.9449   0.6765   0.9858   0.9544   0.5764   0.8185  
0.70       0.9449   0.6765   0.9858   0.9544   0.5764   0.8185  
0.75       0.9449   0.6765   0.9858   0.9544   0.5764   0.8185  
0.80       0.9449   0.6765   0.9858   0.9544   0.5764   0.8185  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9449, F1=0.6765, Normal Recall=0.9858, Normal Precision=0.9544, Attack Recall=0.5764, Attack Precision=0.8185

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
0.15       0.9034   0.7035   0.9860   0.9023   0.5730   0.9108   <--
0.20       0.9034   0.7035   0.9860   0.9023   0.5730   0.9108  
0.25       0.9034   0.7035   0.9860   0.9023   0.5730   0.9108  
0.30       0.9034   0.7035   0.9860   0.9023   0.5730   0.9108  
0.35       0.9034   0.7035   0.9860   0.9023   0.5730   0.9108  
0.40       0.9034   0.7035   0.9860   0.9023   0.5730   0.9108  
0.45       0.9034   0.7035   0.9860   0.9023   0.5730   0.9108  
0.50       0.9034   0.7035   0.9860   0.9023   0.5730   0.9108  
0.55       0.9034   0.7035   0.9860   0.9023   0.5730   0.9108  
0.60       0.9034   0.7035   0.9860   0.9023   0.5730   0.9108  
0.65       0.9034   0.7035   0.9860   0.9023   0.5730   0.9108  
0.70       0.9034   0.7035   0.9860   0.9023   0.5730   0.9108  
0.75       0.9034   0.7035   0.9860   0.9023   0.5730   0.9108  
0.80       0.9034   0.7035   0.9860   0.9023   0.5730   0.9108  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9034, F1=0.7035, Normal Recall=0.9860, Normal Precision=0.9023, Attack Recall=0.5730, Attack Precision=0.9108

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
0.15       0.8623   0.7140   0.9862   0.8435   0.5730   0.9469   <--
0.20       0.8623   0.7140   0.9862   0.8435   0.5730   0.9469  
0.25       0.8623   0.7140   0.9862   0.8435   0.5730   0.9469  
0.30       0.8623   0.7140   0.9862   0.8435   0.5730   0.9469  
0.35       0.8623   0.7140   0.9862   0.8435   0.5730   0.9469  
0.40       0.8623   0.7140   0.9862   0.8435   0.5730   0.9469  
0.45       0.8623   0.7140   0.9862   0.8435   0.5730   0.9469  
0.50       0.8623   0.7140   0.9862   0.8435   0.5730   0.9469  
0.55       0.8623   0.7140   0.9862   0.8435   0.5730   0.9469  
0.60       0.8623   0.7140   0.9862   0.8435   0.5730   0.9469  
0.65       0.8623   0.7140   0.9862   0.8435   0.5730   0.9469  
0.70       0.8623   0.7140   0.9862   0.8435   0.5730   0.9469  
0.75       0.8623   0.7140   0.9862   0.8435   0.5730   0.9469  
0.80       0.8623   0.7140   0.9862   0.8435   0.5730   0.9469  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8623, F1=0.7140, Normal Recall=0.9862, Normal Precision=0.8435, Attack Recall=0.5730, Attack Precision=0.9469

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
0.15       0.8208   0.7190   0.9861   0.7760   0.5730   0.9648   <--
0.20       0.8208   0.7190   0.9861   0.7760   0.5730   0.9648  
0.25       0.8208   0.7190   0.9861   0.7760   0.5730   0.9648  
0.30       0.8208   0.7190   0.9861   0.7760   0.5730   0.9648  
0.35       0.8208   0.7190   0.9861   0.7760   0.5730   0.9648  
0.40       0.8208   0.7190   0.9861   0.7760   0.5730   0.9648  
0.45       0.8208   0.7190   0.9861   0.7760   0.5730   0.9648  
0.50       0.8208   0.7190   0.9861   0.7760   0.5730   0.9648  
0.55       0.8208   0.7190   0.9861   0.7760   0.5730   0.9648  
0.60       0.8208   0.7190   0.9861   0.7760   0.5730   0.9648  
0.65       0.8208   0.7190   0.9861   0.7760   0.5730   0.9648  
0.70       0.8208   0.7190   0.9861   0.7760   0.5730   0.9648  
0.75       0.8208   0.7190   0.9861   0.7760   0.5730   0.9648  
0.80       0.8208   0.7190   0.9861   0.7760   0.5730   0.9648  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8208, F1=0.7190, Normal Recall=0.9861, Normal Precision=0.7760, Attack Recall=0.5730, Attack Precision=0.9648

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
0.15       0.7796   0.7222   0.9861   0.6978   0.5730   0.9763   <--
0.20       0.7796   0.7222   0.9861   0.6978   0.5730   0.9763  
0.25       0.7796   0.7222   0.9861   0.6978   0.5730   0.9763  
0.30       0.7796   0.7222   0.9861   0.6978   0.5730   0.9763  
0.35       0.7796   0.7222   0.9861   0.6978   0.5730   0.9763  
0.40       0.7796   0.7222   0.9861   0.6978   0.5730   0.9763  
0.45       0.7796   0.7222   0.9861   0.6978   0.5730   0.9763  
0.50       0.7796   0.7222   0.9861   0.6978   0.5730   0.9763  
0.55       0.7796   0.7222   0.9861   0.6978   0.5730   0.9763  
0.60       0.7796   0.7222   0.9861   0.6978   0.5730   0.9763  
0.65       0.7796   0.7222   0.9861   0.6978   0.5730   0.9763  
0.70       0.7796   0.7222   0.9861   0.6978   0.5730   0.9763  
0.75       0.7796   0.7222   0.9861   0.6978   0.5730   0.9763  
0.80       0.7796   0.7222   0.9861   0.6978   0.5730   0.9763  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7796, F1=0.7222, Normal Recall=0.9861, Normal Precision=0.6978, Attack Recall=0.5730, Attack Precision=0.9763

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
0.15       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191   <--
0.20       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.25       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.30       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.35       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.40       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.45       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.50       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.55       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.60       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.65       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.70       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.75       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.80       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9449, F1=0.6768, Normal Recall=0.9858, Normal Precision=0.9545, Attack Recall=0.5766, Attack Precision=0.8191

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
0.15       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110   <--
0.20       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.25       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.30       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.35       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.40       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.45       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.50       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.55       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.60       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.65       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.70       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.75       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.80       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9034, F1=0.7036, Normal Recall=0.9860, Normal Precision=0.9024, Attack Recall=0.5732, Attack Precision=0.9110

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
0.15       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470   <--
0.20       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.25       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.30       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.35       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.40       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.45       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.50       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.55       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.60       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.65       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.70       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.75       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.80       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8623, F1=0.7141, Normal Recall=0.9863, Normal Precision=0.8436, Attack Recall=0.5732, Attack Precision=0.9470

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
0.15       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650   <--
0.20       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.25       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.30       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.35       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.40       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.45       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.50       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.55       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.60       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.65       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.70       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.75       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.80       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8210, F1=0.7192, Normal Recall=0.9861, Normal Precision=0.7761, Attack Recall=0.5732, Attack Precision=0.9650

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
0.15       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763   <--
0.20       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.25       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.30       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.35       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.40       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.45       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.50       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.55       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.60       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.65       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.70       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.75       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.80       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7796, F1=0.7223, Normal Recall=0.9861, Normal Precision=0.6979, Attack Recall=0.5732, Attack Precision=0.9763

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
0.15       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191   <--
0.20       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.25       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.30       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.35       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.40       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.45       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.50       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.55       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.60       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.65       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.70       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.75       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
0.80       0.9449   0.6768   0.9858   0.9545   0.5766   0.8191  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9449, F1=0.6768, Normal Recall=0.9858, Normal Precision=0.9545, Attack Recall=0.5766, Attack Precision=0.8191

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
0.15       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110   <--
0.20       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.25       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.30       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.35       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.40       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.45       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.50       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.55       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.60       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.65       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.70       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.75       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
0.80       0.9034   0.7036   0.9860   0.9024   0.5732   0.9110  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9034, F1=0.7036, Normal Recall=0.9860, Normal Precision=0.9024, Attack Recall=0.5732, Attack Precision=0.9110

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
0.15       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470   <--
0.20       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.25       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.30       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.35       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.40       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.45       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.50       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.55       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.60       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.65       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.70       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.75       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
0.80       0.8623   0.7141   0.9863   0.8436   0.5732   0.9470  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8623, F1=0.7141, Normal Recall=0.9863, Normal Precision=0.8436, Attack Recall=0.5732, Attack Precision=0.9470

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
0.15       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650   <--
0.20       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.25       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.30       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.35       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.40       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.45       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.50       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.55       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.60       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.65       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.70       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.75       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
0.80       0.8210   0.7192   0.9861   0.7761   0.5732   0.9650  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8210, F1=0.7192, Normal Recall=0.9861, Normal Precision=0.7761, Attack Recall=0.5732, Attack Precision=0.9650

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
0.15       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763   <--
0.20       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.25       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.30       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.35       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.40       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.45       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.50       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.55       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.60       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.65       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.70       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.75       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
0.80       0.7796   0.7223   0.9861   0.6979   0.5732   0.9763  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7796, F1=0.7223, Normal Recall=0.9861, Normal Precision=0.6979, Attack Recall=0.5732, Attack Precision=0.9763

```

