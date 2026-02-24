# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e2_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-13 00:28:10 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3875 | 0.4037 | 0.4193 | 0.4354 | 0.4502 | 0.4666 | 0.4831 | 0.4971 | 0.5130 | 0.5293 | 0.5445 |
| QAT+Prune only | 0.6164 | 0.6532 | 0.6913 | 0.7308 | 0.7700 | 0.8069 | 0.8451 | 0.8837 | 0.9210 | 0.9605 | 0.9980 |
| QAT+PTQ | 0.6158 | 0.6528 | 0.6910 | 0.7305 | 0.7698 | 0.8068 | 0.8450 | 0.8835 | 0.9209 | 0.9604 | 0.9980 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6158 | 0.6528 | 0.6910 | 0.7305 | 0.7698 | 0.8068 | 0.8450 | 0.8835 | 0.9209 | 0.9604 | 0.9980 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1544 | 0.2728 | 0.3666 | 0.4421 | 0.5051 | 0.5583 | 0.6026 | 0.6415 | 0.6756 | 0.7051 |
| QAT+Prune only | 0.0000 | 0.3653 | 0.5639 | 0.6898 | 0.7763 | 0.8379 | 0.8855 | 0.9232 | 0.9529 | 0.9785 | 0.9990 |
| QAT+PTQ | 0.0000 | 0.3651 | 0.5637 | 0.6896 | 0.7762 | 0.8378 | 0.8854 | 0.9230 | 0.9528 | 0.9784 | 0.9990 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3651 | 0.5637 | 0.6896 | 0.7762 | 0.8378 | 0.8854 | 0.9230 | 0.9528 | 0.9784 | 0.9990 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3875 | 0.3880 | 0.3880 | 0.3886 | 0.3873 | 0.3886 | 0.3909 | 0.3865 | 0.3869 | 0.3925 | 0.0000 |
| QAT+Prune only | 0.6164 | 0.6149 | 0.6146 | 0.6162 | 0.6179 | 0.6158 | 0.6159 | 0.6170 | 0.6130 | 0.6228 | 0.0000 |
| QAT+PTQ | 0.6158 | 0.6145 | 0.6143 | 0.6159 | 0.6176 | 0.6155 | 0.6155 | 0.6164 | 0.6128 | 0.6220 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6158 | 0.6145 | 0.6143 | 0.6159 | 0.6176 | 0.6155 | 0.6155 | 0.6164 | 0.6128 | 0.6220 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3875 | 0.0000 | 0.0000 | 0.0000 | 0.3875 | 1.0000 |
| 90 | 10 | 299,940 | 0.4037 | 0.0900 | 0.5445 | 0.1544 | 0.3880 | 0.8846 |
| 80 | 20 | 291,350 | 0.4193 | 0.1820 | 0.5445 | 0.2728 | 0.3880 | 0.7731 |
| 70 | 30 | 194,230 | 0.4354 | 0.2763 | 0.5445 | 0.3666 | 0.3886 | 0.6657 |
| 60 | 40 | 145,675 | 0.4502 | 0.3721 | 0.5445 | 0.4421 | 0.3873 | 0.5605 |
| 50 | 50 | 116,540 | 0.4666 | 0.4711 | 0.5445 | 0.5051 | 0.3886 | 0.4604 |
| 40 | 60 | 97,115 | 0.4831 | 0.5728 | 0.5445 | 0.5583 | 0.3909 | 0.3639 |
| 30 | 70 | 83,240 | 0.4971 | 0.6744 | 0.5446 | 0.6026 | 0.3865 | 0.2667 |
| 20 | 80 | 72,835 | 0.5130 | 0.7803 | 0.5445 | 0.6415 | 0.3869 | 0.1752 |
| 10 | 90 | 64,740 | 0.5293 | 0.8897 | 0.5445 | 0.6756 | 0.3925 | 0.0874 |
| 0 | 100 | 58,270 | 0.5445 | 1.0000 | 0.5445 | 0.7051 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6164 | 0.0000 | 0.0000 | 0.0000 | 0.6164 | 1.0000 |
| 90 | 10 | 299,940 | 0.6532 | 0.2236 | 0.9981 | 0.3653 | 0.6149 | 0.9997 |
| 80 | 20 | 291,350 | 0.6913 | 0.3930 | 0.9980 | 0.5639 | 0.6146 | 0.9992 |
| 70 | 30 | 194,230 | 0.7308 | 0.5271 | 0.9980 | 0.6898 | 0.6162 | 0.9986 |
| 60 | 40 | 145,675 | 0.7700 | 0.6352 | 0.9980 | 0.7763 | 0.6179 | 0.9978 |
| 50 | 50 | 116,540 | 0.8069 | 0.7221 | 0.9980 | 0.8379 | 0.6158 | 0.9968 |
| 40 | 60 | 97,115 | 0.8451 | 0.7958 | 0.9980 | 0.8855 | 0.6159 | 0.9951 |
| 30 | 70 | 83,240 | 0.8837 | 0.8588 | 0.9980 | 0.9232 | 0.6170 | 0.9925 |
| 20 | 80 | 72,835 | 0.9210 | 0.9116 | 0.9980 | 0.9529 | 0.6130 | 0.9871 |
| 10 | 90 | 64,740 | 0.9605 | 0.9597 | 0.9980 | 0.9785 | 0.6228 | 0.9718 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6158 | 0.0000 | 0.0000 | 0.0000 | 0.6158 | 1.0000 |
| 90 | 10 | 299,940 | 0.6528 | 0.2234 | 0.9981 | 0.3651 | 0.6145 | 0.9997 |
| 80 | 20 | 291,350 | 0.6910 | 0.3928 | 0.9980 | 0.5637 | 0.6143 | 0.9992 |
| 70 | 30 | 194,230 | 0.7305 | 0.5268 | 0.9980 | 0.6896 | 0.6159 | 0.9986 |
| 60 | 40 | 145,675 | 0.7698 | 0.6350 | 0.9980 | 0.7762 | 0.6176 | 0.9978 |
| 50 | 50 | 116,540 | 0.8068 | 0.7219 | 0.9980 | 0.8378 | 0.6155 | 0.9967 |
| 40 | 60 | 97,115 | 0.8450 | 0.7957 | 0.9980 | 0.8854 | 0.6155 | 0.9951 |
| 30 | 70 | 83,240 | 0.8835 | 0.8586 | 0.9980 | 0.9230 | 0.6164 | 0.9924 |
| 20 | 80 | 72,835 | 0.9209 | 0.9116 | 0.9980 | 0.9528 | 0.6128 | 0.9870 |
| 10 | 90 | 64,740 | 0.9604 | 0.9596 | 0.9980 | 0.9784 | 0.6220 | 0.9715 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6158 | 0.0000 | 0.0000 | 0.0000 | 0.6158 | 1.0000 |
| 90 | 10 | 299,940 | 0.6528 | 0.2234 | 0.9981 | 0.3651 | 0.6145 | 0.9997 |
| 80 | 20 | 291,350 | 0.6910 | 0.3928 | 0.9980 | 0.5637 | 0.6143 | 0.9992 |
| 70 | 30 | 194,230 | 0.7305 | 0.5268 | 0.9980 | 0.6896 | 0.6159 | 0.9986 |
| 60 | 40 | 145,675 | 0.7698 | 0.6350 | 0.9980 | 0.7762 | 0.6176 | 0.9978 |
| 50 | 50 | 116,540 | 0.8068 | 0.7219 | 0.9980 | 0.8378 | 0.6155 | 0.9967 |
| 40 | 60 | 97,115 | 0.8450 | 0.7957 | 0.9980 | 0.8854 | 0.6155 | 0.9951 |
| 30 | 70 | 83,240 | 0.8835 | 0.8586 | 0.9980 | 0.9230 | 0.6164 | 0.9924 |
| 20 | 80 | 72,835 | 0.9209 | 0.9116 | 0.9980 | 0.9528 | 0.6128 | 0.9870 |
| 10 | 90 | 64,740 | 0.9604 | 0.9596 | 0.9980 | 0.9784 | 0.6220 | 0.9715 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |


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
0.15       0.4036   0.1541   0.3880   0.8844   0.5434   0.0898   <--
0.20       0.4036   0.1541   0.3880   0.8844   0.5434   0.0898  
0.25       0.4036   0.1541   0.3880   0.8844   0.5434   0.0898  
0.30       0.4036   0.1541   0.3880   0.8844   0.5434   0.0898  
0.35       0.4036   0.1541   0.3880   0.8844   0.5434   0.0898  
0.40       0.4036   0.1541   0.3880   0.8844   0.5434   0.0898  
0.45       0.4036   0.1541   0.3880   0.8844   0.5434   0.0898  
0.50       0.4036   0.1541   0.3880   0.8844   0.5434   0.0898  
0.55       0.4036   0.1541   0.3880   0.8844   0.5434   0.0898  
0.60       0.4036   0.1541   0.3880   0.8844   0.5434   0.0898  
0.65       0.4036   0.1541   0.3880   0.8844   0.5434   0.0898  
0.70       0.4036   0.1541   0.3880   0.8844   0.5434   0.0898  
0.75       0.4036   0.1541   0.3880   0.8844   0.5434   0.0898  
0.80       0.4036   0.1541   0.3880   0.8844   0.5434   0.0898  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4036, F1=0.1541, Normal Recall=0.3880, Normal Precision=0.8844, Attack Recall=0.5434, Attack Precision=0.0898

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
0.15       0.4192   0.2727   0.3879   0.7730   0.5445   0.1819   <--
0.20       0.4192   0.2727   0.3879   0.7730   0.5445   0.1819  
0.25       0.4192   0.2727   0.3879   0.7730   0.5445   0.1819  
0.30       0.4192   0.2727   0.3879   0.7730   0.5445   0.1819  
0.35       0.4192   0.2727   0.3879   0.7730   0.5445   0.1819  
0.40       0.4192   0.2727   0.3879   0.7730   0.5445   0.1819  
0.45       0.4192   0.2727   0.3879   0.7730   0.5445   0.1819  
0.50       0.4192   0.2727   0.3879   0.7730   0.5445   0.1819  
0.55       0.4192   0.2727   0.3879   0.7730   0.5445   0.1819  
0.60       0.4192   0.2727   0.3879   0.7730   0.5445   0.1819  
0.65       0.4192   0.2727   0.3879   0.7730   0.5445   0.1819  
0.70       0.4192   0.2727   0.3879   0.7730   0.5445   0.1819  
0.75       0.4192   0.2727   0.3879   0.7730   0.5445   0.1819  
0.80       0.4192   0.2727   0.3879   0.7730   0.5445   0.1819  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4192, F1=0.2727, Normal Recall=0.3879, Normal Precision=0.7730, Attack Recall=0.5445, Attack Precision=0.1819

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
0.15       0.4358   0.3667   0.3892   0.6660   0.5445   0.2764   <--
0.20       0.4358   0.3667   0.3892   0.6660   0.5445   0.2764  
0.25       0.4358   0.3667   0.3892   0.6660   0.5445   0.2764  
0.30       0.4358   0.3667   0.3892   0.6660   0.5445   0.2764  
0.35       0.4358   0.3667   0.3892   0.6660   0.5445   0.2764  
0.40       0.4358   0.3667   0.3892   0.6660   0.5445   0.2764  
0.45       0.4358   0.3667   0.3892   0.6660   0.5445   0.2764  
0.50       0.4358   0.3667   0.3892   0.6660   0.5445   0.2764  
0.55       0.4358   0.3667   0.3892   0.6660   0.5445   0.2764  
0.60       0.4358   0.3667   0.3892   0.6660   0.5445   0.2764  
0.65       0.4358   0.3667   0.3892   0.6660   0.5445   0.2764  
0.70       0.4358   0.3667   0.3892   0.6660   0.5445   0.2764  
0.75       0.4358   0.3667   0.3892   0.6660   0.5445   0.2764  
0.80       0.4358   0.3667   0.3892   0.6660   0.5445   0.2764  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4358, F1=0.3667, Normal Recall=0.3892, Normal Precision=0.6660, Attack Recall=0.5445, Attack Precision=0.2764

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
0.15       0.4502   0.4421   0.3874   0.5606   0.5445   0.3721   <--
0.20       0.4502   0.4421   0.3874   0.5606   0.5445   0.3721  
0.25       0.4502   0.4421   0.3874   0.5606   0.5445   0.3721  
0.30       0.4502   0.4421   0.3874   0.5606   0.5445   0.3721  
0.35       0.4502   0.4421   0.3874   0.5606   0.5445   0.3721  
0.40       0.4502   0.4421   0.3874   0.5606   0.5445   0.3721  
0.45       0.4502   0.4421   0.3874   0.5606   0.5445   0.3721  
0.50       0.4502   0.4421   0.3874   0.5606   0.5445   0.3721  
0.55       0.4502   0.4421   0.3874   0.5606   0.5445   0.3721  
0.60       0.4502   0.4421   0.3874   0.5606   0.5445   0.3721  
0.65       0.4502   0.4421   0.3874   0.5606   0.5445   0.3721  
0.70       0.4502   0.4421   0.3874   0.5606   0.5445   0.3721  
0.75       0.4502   0.4421   0.3874   0.5606   0.5445   0.3721  
0.80       0.4502   0.4421   0.3874   0.5606   0.5445   0.3721  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4502, F1=0.4421, Normal Recall=0.3874, Normal Precision=0.5606, Attack Recall=0.5445, Attack Precision=0.3721

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
0.15       0.4670   0.5053   0.3894   0.4609   0.5445   0.4714   <--
0.20       0.4670   0.5053   0.3894   0.4609   0.5445   0.4714  
0.25       0.4670   0.5053   0.3894   0.4609   0.5445   0.4714  
0.30       0.4670   0.5053   0.3894   0.4609   0.5445   0.4714  
0.35       0.4670   0.5053   0.3894   0.4609   0.5445   0.4714  
0.40       0.4670   0.5053   0.3894   0.4609   0.5445   0.4714  
0.45       0.4670   0.5053   0.3894   0.4609   0.5445   0.4714  
0.50       0.4670   0.5053   0.3894   0.4609   0.5445   0.4714  
0.55       0.4670   0.5053   0.3894   0.4609   0.5445   0.4714  
0.60       0.4670   0.5053   0.3894   0.4609   0.5445   0.4714  
0.65       0.4670   0.5053   0.3894   0.4609   0.5445   0.4714  
0.70       0.4670   0.5053   0.3894   0.4609   0.5445   0.4714  
0.75       0.4670   0.5053   0.3894   0.4609   0.5445   0.4714  
0.80       0.4670   0.5053   0.3894   0.4609   0.5445   0.4714  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4670, F1=0.5053, Normal Recall=0.3894, Normal Precision=0.4609, Attack Recall=0.5445, Attack Precision=0.4714

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
0.15       0.6532   0.3653   0.6149   0.9997   0.9981   0.2236   <--
0.20       0.6532   0.3653   0.6149   0.9997   0.9981   0.2236  
0.25       0.6532   0.3653   0.6149   0.9997   0.9981   0.2236  
0.30       0.6532   0.3653   0.6149   0.9997   0.9981   0.2236  
0.35       0.6532   0.3653   0.6149   0.9997   0.9981   0.2236  
0.40       0.6532   0.3653   0.6149   0.9997   0.9981   0.2236  
0.45       0.6532   0.3653   0.6149   0.9997   0.9981   0.2236  
0.50       0.6532   0.3653   0.6149   0.9997   0.9981   0.2236  
0.55       0.6532   0.3653   0.6149   0.9997   0.9981   0.2236  
0.60       0.6532   0.3653   0.6149   0.9997   0.9981   0.2236  
0.65       0.6532   0.3653   0.6149   0.9997   0.9981   0.2236  
0.70       0.6532   0.3653   0.6149   0.9997   0.9981   0.2236  
0.75       0.6532   0.3653   0.6149   0.9997   0.9981   0.2236  
0.80       0.6532   0.3653   0.6149   0.9997   0.9981   0.2236  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6532, F1=0.3653, Normal Recall=0.6149, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.2236

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
0.15       0.6920   0.5645   0.6155   0.9992   0.9980   0.3935   <--
0.20       0.6920   0.5645   0.6155   0.9992   0.9980   0.3935  
0.25       0.6920   0.5645   0.6155   0.9992   0.9980   0.3935  
0.30       0.6920   0.5645   0.6155   0.9992   0.9980   0.3935  
0.35       0.6920   0.5645   0.6155   0.9992   0.9980   0.3935  
0.40       0.6920   0.5645   0.6155   0.9992   0.9980   0.3935  
0.45       0.6920   0.5645   0.6155   0.9992   0.9980   0.3935  
0.50       0.6920   0.5645   0.6155   0.9992   0.9980   0.3935  
0.55       0.6920   0.5645   0.6155   0.9992   0.9980   0.3935  
0.60       0.6920   0.5645   0.6155   0.9992   0.9980   0.3935  
0.65       0.6920   0.5645   0.6155   0.9992   0.9980   0.3935  
0.70       0.6920   0.5645   0.6155   0.9992   0.9980   0.3935  
0.75       0.6920   0.5645   0.6155   0.9992   0.9980   0.3935  
0.80       0.6920   0.5645   0.6155   0.9992   0.9980   0.3935  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6920, F1=0.5645, Normal Recall=0.6155, Normal Precision=0.9992, Attack Recall=0.9980, Attack Precision=0.3935

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
0.15       0.7307   0.6898   0.6162   0.9986   0.9980   0.5270   <--
0.20       0.7307   0.6898   0.6162   0.9986   0.9980   0.5270  
0.25       0.7307   0.6898   0.6162   0.9986   0.9980   0.5270  
0.30       0.7307   0.6898   0.6162   0.9986   0.9980   0.5270  
0.35       0.7307   0.6898   0.6162   0.9986   0.9980   0.5270  
0.40       0.7307   0.6898   0.6162   0.9986   0.9980   0.5270  
0.45       0.7307   0.6898   0.6162   0.9986   0.9980   0.5270  
0.50       0.7307   0.6898   0.6162   0.9986   0.9980   0.5270  
0.55       0.7307   0.6898   0.6162   0.9986   0.9980   0.5270  
0.60       0.7307   0.6898   0.6162   0.9986   0.9980   0.5270  
0.65       0.7307   0.6898   0.6162   0.9986   0.9980   0.5270  
0.70       0.7307   0.6898   0.6162   0.9986   0.9980   0.5270  
0.75       0.7307   0.6898   0.6162   0.9986   0.9980   0.5270  
0.80       0.7307   0.6898   0.6162   0.9986   0.9980   0.5270  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7307, F1=0.6898, Normal Recall=0.6162, Normal Precision=0.9986, Attack Recall=0.9980, Attack Precision=0.5270

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
0.15       0.7689   0.7756   0.6162   0.9978   0.9980   0.6342   <--
0.20       0.7689   0.7756   0.6162   0.9978   0.9980   0.6342  
0.25       0.7689   0.7756   0.6162   0.9978   0.9980   0.6342  
0.30       0.7689   0.7756   0.6162   0.9978   0.9980   0.6342  
0.35       0.7689   0.7756   0.6162   0.9978   0.9980   0.6342  
0.40       0.7689   0.7756   0.6162   0.9978   0.9980   0.6342  
0.45       0.7689   0.7756   0.6162   0.9978   0.9980   0.6342  
0.50       0.7689   0.7756   0.6162   0.9978   0.9980   0.6342  
0.55       0.7689   0.7756   0.6162   0.9978   0.9980   0.6342  
0.60       0.7689   0.7756   0.6162   0.9978   0.9980   0.6342  
0.65       0.7689   0.7756   0.6162   0.9978   0.9980   0.6342  
0.70       0.7689   0.7756   0.6162   0.9978   0.9980   0.6342  
0.75       0.7689   0.7756   0.6162   0.9978   0.9980   0.6342  
0.80       0.7689   0.7756   0.6162   0.9978   0.9980   0.6342  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7689, F1=0.7756, Normal Recall=0.6162, Normal Precision=0.9978, Attack Recall=0.9980, Attack Precision=0.6342

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
0.15       0.8068   0.8378   0.6155   0.9967   0.9980   0.7219   <--
0.20       0.8068   0.8378   0.6155   0.9967   0.9980   0.7219  
0.25       0.8068   0.8378   0.6155   0.9967   0.9980   0.7219  
0.30       0.8068   0.8378   0.6155   0.9967   0.9980   0.7219  
0.35       0.8068   0.8378   0.6155   0.9967   0.9980   0.7219  
0.40       0.8068   0.8378   0.6155   0.9967   0.9980   0.7219  
0.45       0.8068   0.8378   0.6155   0.9967   0.9980   0.7219  
0.50       0.8068   0.8378   0.6155   0.9967   0.9980   0.7219  
0.55       0.8068   0.8378   0.6155   0.9967   0.9980   0.7219  
0.60       0.8068   0.8378   0.6155   0.9967   0.9980   0.7219  
0.65       0.8068   0.8378   0.6155   0.9967   0.9980   0.7219  
0.70       0.8068   0.8378   0.6155   0.9967   0.9980   0.7219  
0.75       0.8068   0.8378   0.6155   0.9967   0.9980   0.7219  
0.80       0.8068   0.8378   0.6155   0.9967   0.9980   0.7219  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8068, F1=0.8378, Normal Recall=0.6155, Normal Precision=0.9967, Attack Recall=0.9980, Attack Precision=0.7219

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
0.15       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234   <--
0.20       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.25       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.30       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.35       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.40       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.45       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.50       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.55       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.60       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.65       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.70       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.75       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.80       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6528, F1=0.3651, Normal Recall=0.6145, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.2234

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
0.15       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933   <--
0.20       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.25       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.30       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.35       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.40       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.45       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.50       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.55       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.60       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.65       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.70       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.75       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.80       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6917, F1=0.5642, Normal Recall=0.6151, Normal Precision=0.9992, Attack Recall=0.9980, Attack Precision=0.3933

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
0.15       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267   <--
0.20       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.25       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.30       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.35       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.40       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.45       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.50       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.55       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.60       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.65       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.70       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.75       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.80       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7304, F1=0.6895, Normal Recall=0.6157, Normal Precision=0.9986, Attack Recall=0.9980, Attack Precision=0.5267

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
0.15       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337   <--
0.20       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.25       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.30       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.35       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.40       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.45       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.50       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.55       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.60       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.65       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.70       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.75       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.80       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7685, F1=0.7752, Normal Recall=0.6155, Normal Precision=0.9978, Attack Recall=0.9980, Attack Precision=0.6337

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
0.15       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214   <--
0.20       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.25       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.30       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.35       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.40       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.45       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.50       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.55       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.60       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.65       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.70       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.75       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.80       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8063, F1=0.8374, Normal Recall=0.6146, Normal Precision=0.9967, Attack Recall=0.9980, Attack Precision=0.7214

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
0.15       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234   <--
0.20       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.25       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.30       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.35       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.40       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.45       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.50       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.55       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.60       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.65       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.70       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.75       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
0.80       0.6528   0.3651   0.6145   0.9997   0.9981   0.2234  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6528, F1=0.3651, Normal Recall=0.6145, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.2234

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
0.15       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933   <--
0.20       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.25       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.30       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.35       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.40       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.45       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.50       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.55       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.60       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.65       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.70       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.75       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
0.80       0.6917   0.5642   0.6151   0.9992   0.9980   0.3933  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6917, F1=0.5642, Normal Recall=0.6151, Normal Precision=0.9992, Attack Recall=0.9980, Attack Precision=0.3933

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
0.15       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267   <--
0.20       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.25       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.30       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.35       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.40       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.45       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.50       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.55       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.60       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.65       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.70       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.75       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
0.80       0.7304   0.6895   0.6157   0.9986   0.9980   0.5267  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7304, F1=0.6895, Normal Recall=0.6157, Normal Precision=0.9986, Attack Recall=0.9980, Attack Precision=0.5267

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
0.15       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337   <--
0.20       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.25       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.30       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.35       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.40       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.45       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.50       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.55       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.60       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.65       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.70       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.75       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
0.80       0.7685   0.7752   0.6155   0.9978   0.9980   0.6337  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7685, F1=0.7752, Normal Recall=0.6155, Normal Precision=0.9978, Attack Recall=0.9980, Attack Precision=0.6337

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
0.15       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214   <--
0.20       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.25       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.30       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.35       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.40       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.45       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.50       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.55       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.60       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.65       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.70       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.75       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
0.80       0.8063   0.8374   0.6146   0.9967   0.9980   0.7214  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8063, F1=0.8374, Normal Recall=0.6146, Normal Precision=0.9967, Attack Recall=0.9980, Attack Precision=0.7214

```

