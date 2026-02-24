# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e2_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-16 23:04:19 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0418 | 0.1376 | 0.2334 | 0.3287 | 0.4246 | 0.5204 | 0.6157 | 0.7113 | 0.8068 | 0.9030 | 0.9985 |
| QAT+Prune only | 0.7888 | 0.8092 | 0.8289 | 0.8503 | 0.8707 | 0.8885 | 0.9108 | 0.9304 | 0.9507 | 0.9711 | 0.9915 |
| QAT+PTQ | 0.7889 | 0.8094 | 0.8291 | 0.8504 | 0.8709 | 0.8887 | 0.9110 | 0.9305 | 0.9509 | 0.9713 | 0.9916 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7889 | 0.8094 | 0.8291 | 0.8504 | 0.8709 | 0.8887 | 0.9110 | 0.9305 | 0.9509 | 0.9713 | 0.9916 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1880 | 0.3426 | 0.4716 | 0.5813 | 0.6755 | 0.7572 | 0.8288 | 0.8921 | 0.9488 | 0.9993 |
| QAT+Prune only | 0.0000 | 0.5095 | 0.6986 | 0.7990 | 0.8598 | 0.8989 | 0.9303 | 0.9522 | 0.9699 | 0.9841 | 0.9957 |
| QAT+PTQ | 0.0000 | 0.5099 | 0.6988 | 0.7991 | 0.8601 | 0.8991 | 0.9304 | 0.9523 | 0.9700 | 0.9842 | 0.9958 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5099 | 0.6988 | 0.7991 | 0.8601 | 0.8991 | 0.9304 | 0.9523 | 0.9700 | 0.9842 | 0.9958 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0418 | 0.0420 | 0.0421 | 0.0416 | 0.0419 | 0.0422 | 0.0415 | 0.0409 | 0.0400 | 0.0431 | 0.0000 |
| QAT+Prune only | 0.7888 | 0.7890 | 0.7882 | 0.7898 | 0.7902 | 0.7856 | 0.7898 | 0.7878 | 0.7874 | 0.7871 | 0.0000 |
| QAT+PTQ | 0.7889 | 0.7892 | 0.7884 | 0.7899 | 0.7904 | 0.7857 | 0.7901 | 0.7878 | 0.7880 | 0.7878 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7889 | 0.7892 | 0.7884 | 0.7899 | 0.7904 | 0.7857 | 0.7901 | 0.7878 | 0.7880 | 0.7878 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0418 | 0.0000 | 0.0000 | 0.0000 | 0.0418 | 1.0000 |
| 90 | 10 | 299,940 | 0.1376 | 0.1038 | 0.9982 | 0.1880 | 0.0420 | 0.9952 |
| 80 | 20 | 291,350 | 0.2334 | 0.2067 | 0.9985 | 0.3426 | 0.0421 | 0.9914 |
| 70 | 30 | 194,230 | 0.3287 | 0.3087 | 0.9985 | 0.4716 | 0.0416 | 0.9852 |
| 60 | 40 | 145,675 | 0.4246 | 0.4100 | 0.9985 | 0.5813 | 0.0419 | 0.9773 |
| 50 | 50 | 116,540 | 0.5204 | 0.5104 | 0.9985 | 0.6755 | 0.0422 | 0.9666 |
| 40 | 60 | 97,115 | 0.6157 | 0.6098 | 0.9985 | 0.7572 | 0.0415 | 0.9499 |
| 30 | 70 | 83,240 | 0.7113 | 0.7084 | 0.9985 | 0.8288 | 0.0409 | 0.9232 |
| 20 | 80 | 72,835 | 0.8068 | 0.8062 | 0.9985 | 0.8921 | 0.0400 | 0.8726 |
| 10 | 90 | 64,740 | 0.9030 | 0.9038 | 0.9985 | 0.9488 | 0.0431 | 0.7665 |
| 0 | 100 | 58,270 | 0.9985 | 1.0000 | 0.9985 | 0.9993 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7888 | 0.0000 | 0.0000 | 0.0000 | 0.7888 | 1.0000 |
| 90 | 10 | 299,940 | 0.8092 | 0.3429 | 0.9912 | 0.5095 | 0.7890 | 0.9988 |
| 80 | 20 | 291,350 | 0.8289 | 0.5392 | 0.9915 | 0.6986 | 0.7882 | 0.9973 |
| 70 | 30 | 194,230 | 0.8503 | 0.6691 | 0.9915 | 0.7990 | 0.7898 | 0.9954 |
| 60 | 40 | 145,675 | 0.8707 | 0.7590 | 0.9915 | 0.8598 | 0.7902 | 0.9929 |
| 50 | 50 | 116,540 | 0.8885 | 0.8222 | 0.9915 | 0.8989 | 0.7856 | 0.9893 |
| 40 | 60 | 97,115 | 0.9108 | 0.8761 | 0.9915 | 0.9303 | 0.7898 | 0.9841 |
| 30 | 70 | 83,240 | 0.9304 | 0.9160 | 0.9915 | 0.9522 | 0.7878 | 0.9755 |
| 20 | 80 | 72,835 | 0.9507 | 0.9491 | 0.9915 | 0.9699 | 0.7874 | 0.9586 |
| 10 | 90 | 64,740 | 0.9711 | 0.9767 | 0.9915 | 0.9841 | 0.7871 | 0.9116 |
| 0 | 100 | 58,270 | 0.9915 | 1.0000 | 0.9915 | 0.9957 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7889 | 0.0000 | 0.0000 | 0.0000 | 0.7889 | 1.0000 |
| 90 | 10 | 299,940 | 0.8094 | 0.3432 | 0.9914 | 0.5099 | 0.7892 | 0.9988 |
| 80 | 20 | 291,350 | 0.8291 | 0.5395 | 0.9916 | 0.6988 | 0.7884 | 0.9974 |
| 70 | 30 | 194,230 | 0.8504 | 0.6692 | 0.9916 | 0.7991 | 0.7899 | 0.9955 |
| 60 | 40 | 145,675 | 0.8709 | 0.7593 | 0.9916 | 0.8601 | 0.7904 | 0.9930 |
| 50 | 50 | 116,540 | 0.8887 | 0.8223 | 0.9916 | 0.8991 | 0.7857 | 0.9895 |
| 40 | 60 | 97,115 | 0.9110 | 0.8763 | 0.9916 | 0.9304 | 0.7901 | 0.9844 |
| 30 | 70 | 83,240 | 0.9305 | 0.9160 | 0.9916 | 0.9523 | 0.7878 | 0.9758 |
| 20 | 80 | 72,835 | 0.9509 | 0.9493 | 0.9916 | 0.9700 | 0.7880 | 0.9593 |
| 10 | 90 | 64,740 | 0.9713 | 0.9768 | 0.9917 | 0.9842 | 0.7878 | 0.9130 |
| 0 | 100 | 58,270 | 0.9916 | 1.0000 | 0.9916 | 0.9958 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7889 | 0.0000 | 0.0000 | 0.0000 | 0.7889 | 1.0000 |
| 90 | 10 | 299,940 | 0.8094 | 0.3432 | 0.9914 | 0.5099 | 0.7892 | 0.9988 |
| 80 | 20 | 291,350 | 0.8291 | 0.5395 | 0.9916 | 0.6988 | 0.7884 | 0.9974 |
| 70 | 30 | 194,230 | 0.8504 | 0.6692 | 0.9916 | 0.7991 | 0.7899 | 0.9955 |
| 60 | 40 | 145,675 | 0.8709 | 0.7593 | 0.9916 | 0.8601 | 0.7904 | 0.9930 |
| 50 | 50 | 116,540 | 0.8887 | 0.8223 | 0.9916 | 0.8991 | 0.7857 | 0.9895 |
| 40 | 60 | 97,115 | 0.9110 | 0.8763 | 0.9916 | 0.9304 | 0.7901 | 0.9844 |
| 30 | 70 | 83,240 | 0.9305 | 0.9160 | 0.9916 | 0.9523 | 0.7878 | 0.9758 |
| 20 | 80 | 72,835 | 0.9509 | 0.9493 | 0.9916 | 0.9700 | 0.7880 | 0.9593 |
| 10 | 90 | 64,740 | 0.9713 | 0.9768 | 0.9917 | 0.9842 | 0.7878 | 0.9130 |
| 0 | 100 | 58,270 | 0.9916 | 1.0000 | 0.9916 | 0.9958 | 0.0000 | 0.0000 |


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
0.15       0.1377   0.1881   0.0420   0.9967   0.9987   0.1038   <--
0.20       0.1377   0.1881   0.0420   0.9967   0.9987   0.1038  
0.25       0.1377   0.1881   0.0420   0.9967   0.9987   0.1038  
0.30       0.1377   0.1881   0.0420   0.9967   0.9987   0.1038  
0.35       0.1377   0.1881   0.0420   0.9967   0.9987   0.1038  
0.40       0.1377   0.1881   0.0420   0.9967   0.9987   0.1038  
0.45       0.1377   0.1881   0.0420   0.9967   0.9987   0.1038  
0.50       0.1377   0.1881   0.0420   0.9967   0.9987   0.1038  
0.55       0.1377   0.1881   0.0420   0.9967   0.9987   0.1038  
0.60       0.1377   0.1881   0.0420   0.9967   0.9987   0.1038  
0.65       0.1377   0.1881   0.0420   0.9967   0.9987   0.1038  
0.70       0.1377   0.1881   0.0420   0.9967   0.9987   0.1038  
0.75       0.1377   0.1881   0.0420   0.9967   0.9987   0.1038  
0.80       0.1377   0.1881   0.0420   0.9967   0.9987   0.1038  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.1377, F1=0.1881, Normal Recall=0.0420, Normal Precision=0.9967, Attack Recall=0.9987, Attack Precision=0.1038

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
0.15       0.2334   0.3425   0.0421   0.9914   0.9985   0.2067   <--
0.20       0.2334   0.3425   0.0421   0.9914   0.9985   0.2067  
0.25       0.2334   0.3425   0.0421   0.9914   0.9985   0.2067  
0.30       0.2334   0.3425   0.0421   0.9914   0.9985   0.2067  
0.35       0.2334   0.3425   0.0421   0.9914   0.9985   0.2067  
0.40       0.2334   0.3425   0.0421   0.9914   0.9985   0.2067  
0.45       0.2334   0.3425   0.0421   0.9914   0.9985   0.2067  
0.50       0.2334   0.3425   0.0421   0.9914   0.9985   0.2067  
0.55       0.2334   0.3425   0.0421   0.9914   0.9985   0.2067  
0.60       0.2334   0.3425   0.0421   0.9914   0.9985   0.2067  
0.65       0.2334   0.3425   0.0421   0.9914   0.9985   0.2067  
0.70       0.2334   0.3425   0.0421   0.9914   0.9985   0.2067  
0.75       0.2334   0.3425   0.0421   0.9914   0.9985   0.2067  
0.80       0.2334   0.3425   0.0421   0.9914   0.9985   0.2067  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2334, F1=0.3425, Normal Recall=0.0421, Normal Precision=0.9914, Attack Recall=0.9985, Attack Precision=0.2067

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
0.15       0.3291   0.4717   0.0422   0.9854   0.9985   0.3088   <--
0.20       0.3291   0.4717   0.0422   0.9854   0.9985   0.3088  
0.25       0.3291   0.4717   0.0422   0.9854   0.9985   0.3088  
0.30       0.3291   0.4717   0.0422   0.9854   0.9985   0.3088  
0.35       0.3291   0.4717   0.0422   0.9854   0.9985   0.3088  
0.40       0.3291   0.4717   0.0422   0.9854   0.9985   0.3088  
0.45       0.3291   0.4717   0.0422   0.9854   0.9985   0.3088  
0.50       0.3291   0.4717   0.0422   0.9854   0.9985   0.3088  
0.55       0.3291   0.4717   0.0422   0.9854   0.9985   0.3088  
0.60       0.3291   0.4717   0.0422   0.9854   0.9985   0.3088  
0.65       0.3291   0.4717   0.0422   0.9854   0.9985   0.3088  
0.70       0.3291   0.4717   0.0422   0.9854   0.9985   0.3088  
0.75       0.3291   0.4717   0.0422   0.9854   0.9985   0.3088  
0.80       0.3291   0.4717   0.0422   0.9854   0.9985   0.3088  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3291, F1=0.4717, Normal Recall=0.0422, Normal Precision=0.9854, Attack Recall=0.9985, Attack Precision=0.3088

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
0.15       0.4246   0.5813   0.0420   0.9774   0.9985   0.4100   <--
0.20       0.4246   0.5813   0.0420   0.9774   0.9985   0.4100  
0.25       0.4246   0.5813   0.0420   0.9774   0.9985   0.4100  
0.30       0.4246   0.5813   0.0420   0.9774   0.9985   0.4100  
0.35       0.4246   0.5813   0.0420   0.9774   0.9985   0.4100  
0.40       0.4246   0.5813   0.0420   0.9774   0.9985   0.4100  
0.45       0.4246   0.5813   0.0420   0.9774   0.9985   0.4100  
0.50       0.4246   0.5813   0.0420   0.9774   0.9985   0.4100  
0.55       0.4246   0.5813   0.0420   0.9774   0.9985   0.4100  
0.60       0.4246   0.5813   0.0420   0.9774   0.9985   0.4100  
0.65       0.4246   0.5813   0.0420   0.9774   0.9985   0.4100  
0.70       0.4246   0.5813   0.0420   0.9774   0.9985   0.4100  
0.75       0.4246   0.5813   0.0420   0.9774   0.9985   0.4100  
0.80       0.4246   0.5813   0.0420   0.9774   0.9985   0.4100  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4246, F1=0.5813, Normal Recall=0.0420, Normal Precision=0.9774, Attack Recall=0.9985, Attack Precision=0.4100

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
0.15       0.5201   0.6754   0.0417   0.9662   0.9985   0.5103   <--
0.20       0.5201   0.6754   0.0417   0.9662   0.9985   0.5103  
0.25       0.5201   0.6754   0.0417   0.9662   0.9985   0.5103  
0.30       0.5201   0.6754   0.0417   0.9662   0.9985   0.5103  
0.35       0.5201   0.6754   0.0417   0.9662   0.9985   0.5103  
0.40       0.5201   0.6754   0.0417   0.9662   0.9985   0.5103  
0.45       0.5201   0.6754   0.0417   0.9662   0.9985   0.5103  
0.50       0.5201   0.6754   0.0417   0.9662   0.9985   0.5103  
0.55       0.5201   0.6754   0.0417   0.9662   0.9985   0.5103  
0.60       0.5201   0.6754   0.0417   0.9662   0.9985   0.5103  
0.65       0.5201   0.6754   0.0417   0.9662   0.9985   0.5103  
0.70       0.5201   0.6754   0.0417   0.9662   0.9985   0.5103  
0.75       0.5201   0.6754   0.0417   0.9662   0.9985   0.5103  
0.80       0.5201   0.6754   0.0417   0.9662   0.9985   0.5103  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5201, F1=0.6754, Normal Recall=0.0417, Normal Precision=0.9662, Attack Recall=0.9985, Attack Precision=0.5103

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
0.15       0.8092   0.5095   0.7890   0.9987   0.9910   0.3429   <--
0.20       0.8092   0.5095   0.7890   0.9987   0.9910   0.3429  
0.25       0.8092   0.5095   0.7890   0.9987   0.9910   0.3429  
0.30       0.8092   0.5095   0.7890   0.9987   0.9910   0.3429  
0.35       0.8092   0.5095   0.7890   0.9987   0.9910   0.3429  
0.40       0.8092   0.5095   0.7890   0.9987   0.9910   0.3429  
0.45       0.8092   0.5095   0.7890   0.9987   0.9910   0.3429  
0.50       0.8092   0.5095   0.7890   0.9987   0.9910   0.3429  
0.55       0.8092   0.5095   0.7890   0.9987   0.9910   0.3429  
0.60       0.8092   0.5095   0.7890   0.9987   0.9910   0.3429  
0.65       0.8092   0.5095   0.7890   0.9987   0.9910   0.3429  
0.70       0.8092   0.5095   0.7890   0.9987   0.9910   0.3429  
0.75       0.8092   0.5095   0.7890   0.9987   0.9910   0.3429  
0.80       0.8092   0.5095   0.7890   0.9987   0.9910   0.3429  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8092, F1=0.5095, Normal Recall=0.7890, Normal Precision=0.9987, Attack Recall=0.9910, Attack Precision=0.3429

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
0.15       0.8298   0.6998   0.7894   0.9973   0.9915   0.5407   <--
0.20       0.8298   0.6998   0.7894   0.9973   0.9915   0.5407  
0.25       0.8298   0.6998   0.7894   0.9973   0.9915   0.5407  
0.30       0.8298   0.6998   0.7894   0.9973   0.9915   0.5407  
0.35       0.8298   0.6998   0.7894   0.9973   0.9915   0.5407  
0.40       0.8298   0.6998   0.7894   0.9973   0.9915   0.5407  
0.45       0.8298   0.6998   0.7894   0.9973   0.9915   0.5407  
0.50       0.8298   0.6998   0.7894   0.9973   0.9915   0.5407  
0.55       0.8298   0.6998   0.7894   0.9973   0.9915   0.5407  
0.60       0.8298   0.6998   0.7894   0.9973   0.9915   0.5407  
0.65       0.8298   0.6998   0.7894   0.9973   0.9915   0.5407  
0.70       0.8298   0.6998   0.7894   0.9973   0.9915   0.5407  
0.75       0.8298   0.6998   0.7894   0.9973   0.9915   0.5407  
0.80       0.8298   0.6998   0.7894   0.9973   0.9915   0.5407  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8298, F1=0.6998, Normal Recall=0.7894, Normal Precision=0.9973, Attack Recall=0.9915, Attack Precision=0.5407

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
0.15       0.8499   0.7985   0.7892   0.9954   0.9915   0.6684   <--
0.20       0.8499   0.7985   0.7892   0.9954   0.9915   0.6684  
0.25       0.8499   0.7985   0.7892   0.9954   0.9915   0.6684  
0.30       0.8499   0.7985   0.7892   0.9954   0.9915   0.6684  
0.35       0.8499   0.7985   0.7892   0.9954   0.9915   0.6684  
0.40       0.8499   0.7985   0.7892   0.9954   0.9915   0.6684  
0.45       0.8499   0.7985   0.7892   0.9954   0.9915   0.6684  
0.50       0.8499   0.7985   0.7892   0.9954   0.9915   0.6684  
0.55       0.8499   0.7985   0.7892   0.9954   0.9915   0.6684  
0.60       0.8499   0.7985   0.7892   0.9954   0.9915   0.6684  
0.65       0.8499   0.7985   0.7892   0.9954   0.9915   0.6684  
0.70       0.8499   0.7985   0.7892   0.9954   0.9915   0.6684  
0.75       0.8499   0.7985   0.7892   0.9954   0.9915   0.6684  
0.80       0.8499   0.7985   0.7892   0.9954   0.9915   0.6684  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8499, F1=0.7985, Normal Recall=0.7892, Normal Precision=0.9954, Attack Recall=0.9915, Attack Precision=0.6684

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
0.15       0.8703   0.8594   0.7894   0.9929   0.9915   0.7584   <--
0.20       0.8703   0.8594   0.7894   0.9929   0.9915   0.7584  
0.25       0.8703   0.8594   0.7894   0.9929   0.9915   0.7584  
0.30       0.8703   0.8594   0.7894   0.9929   0.9915   0.7584  
0.35       0.8703   0.8594   0.7894   0.9929   0.9915   0.7584  
0.40       0.8703   0.8594   0.7894   0.9929   0.9915   0.7584  
0.45       0.8703   0.8594   0.7894   0.9929   0.9915   0.7584  
0.50       0.8703   0.8594   0.7894   0.9929   0.9915   0.7584  
0.55       0.8703   0.8594   0.7894   0.9929   0.9915   0.7584  
0.60       0.8703   0.8594   0.7894   0.9929   0.9915   0.7584  
0.65       0.8703   0.8594   0.7894   0.9929   0.9915   0.7584  
0.70       0.8703   0.8594   0.7894   0.9929   0.9915   0.7584  
0.75       0.8703   0.8594   0.7894   0.9929   0.9915   0.7584  
0.80       0.8703   0.8594   0.7894   0.9929   0.9915   0.7584  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8703, F1=0.8594, Normal Recall=0.7894, Normal Precision=0.9929, Attack Recall=0.9915, Attack Precision=0.7584

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
0.15       0.8897   0.8999   0.7878   0.9893   0.9915   0.8237   <--
0.20       0.8897   0.8999   0.7878   0.9893   0.9915   0.8237  
0.25       0.8897   0.8999   0.7878   0.9893   0.9915   0.8237  
0.30       0.8897   0.8999   0.7878   0.9893   0.9915   0.8237  
0.35       0.8897   0.8999   0.7878   0.9893   0.9915   0.8237  
0.40       0.8897   0.8999   0.7878   0.9893   0.9915   0.8237  
0.45       0.8897   0.8999   0.7878   0.9893   0.9915   0.8237  
0.50       0.8897   0.8999   0.7878   0.9893   0.9915   0.8237  
0.55       0.8897   0.8999   0.7878   0.9893   0.9915   0.8237  
0.60       0.8897   0.8999   0.7878   0.9893   0.9915   0.8237  
0.65       0.8897   0.8999   0.7878   0.9893   0.9915   0.8237  
0.70       0.8897   0.8999   0.7878   0.9893   0.9915   0.8237  
0.75       0.8897   0.8999   0.7878   0.9893   0.9915   0.8237  
0.80       0.8897   0.8999   0.7878   0.9893   0.9915   0.8237  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8897, F1=0.8999, Normal Recall=0.7878, Normal Precision=0.9893, Attack Recall=0.9915, Attack Precision=0.8237

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
0.15       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432   <--
0.20       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.25       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.30       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.35       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.40       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.45       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.50       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.55       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.60       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.65       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.70       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.75       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.80       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8094, F1=0.5098, Normal Recall=0.7892, Normal Precision=0.9988, Attack Recall=0.9911, Attack Precision=0.3432

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
0.15       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409   <--
0.20       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.25       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.30       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.35       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.40       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.45       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.50       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.55       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.60       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.65       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.70       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.75       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.80       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8300, F1=0.7000, Normal Recall=0.7896, Normal Precision=0.9974, Attack Recall=0.9916, Attack Precision=0.5409

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
0.15       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685   <--
0.20       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.25       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.30       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.35       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.40       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.45       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.50       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.55       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.60       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.65       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.70       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.75       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.80       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8500, F1=0.7986, Normal Recall=0.7893, Normal Precision=0.9955, Attack Recall=0.9916, Attack Precision=0.6685

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
0.15       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585   <--
0.20       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.25       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.30       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.35       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.40       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.45       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.50       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.55       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.60       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.65       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.70       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.75       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.80       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8704, F1=0.8595, Normal Recall=0.7895, Normal Precision=0.9930, Attack Recall=0.9916, Attack Precision=0.7585

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
0.15       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240   <--
0.20       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.25       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.30       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.35       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.40       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.45       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.50       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.55       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.60       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.65       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.70       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.75       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.80       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8899, F1=0.9001, Normal Recall=0.7881, Normal Precision=0.9895, Attack Recall=0.9916, Attack Precision=0.8240

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
0.15       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432   <--
0.20       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.25       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.30       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.35       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.40       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.45       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.50       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.55       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.60       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.65       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.70       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.75       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
0.80       0.8094   0.5098   0.7892   0.9988   0.9911   0.3432  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8094, F1=0.5098, Normal Recall=0.7892, Normal Precision=0.9988, Attack Recall=0.9911, Attack Precision=0.3432

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
0.15       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409   <--
0.20       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.25       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.30       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.35       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.40       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.45       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.50       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.55       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.60       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.65       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.70       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.75       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
0.80       0.8300   0.7000   0.7896   0.9974   0.9916   0.5409  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8300, F1=0.7000, Normal Recall=0.7896, Normal Precision=0.9974, Attack Recall=0.9916, Attack Precision=0.5409

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
0.15       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685   <--
0.20       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.25       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.30       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.35       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.40       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.45       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.50       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.55       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.60       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.65       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.70       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.75       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
0.80       0.8500   0.7986   0.7893   0.9955   0.9916   0.6685  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8500, F1=0.7986, Normal Recall=0.7893, Normal Precision=0.9955, Attack Recall=0.9916, Attack Precision=0.6685

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
0.15       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585   <--
0.20       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.25       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.30       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.35       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.40       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.45       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.50       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.55       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.60       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.65       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.70       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.75       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
0.80       0.8704   0.8595   0.7895   0.9930   0.9916   0.7585  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8704, F1=0.8595, Normal Recall=0.7895, Normal Precision=0.9930, Attack Recall=0.9916, Attack Precision=0.7585

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
0.15       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240   <--
0.20       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.25       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.30       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.35       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.40       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.45       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.50       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.55       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.60       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.65       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.70       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.75       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
0.80       0.8899   0.9001   0.7881   0.9895   0.9916   0.8240  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8899, F1=0.9001, Normal Recall=0.7881, Normal Precision=0.9895, Attack Recall=0.9916, Attack Precision=0.8240

```

