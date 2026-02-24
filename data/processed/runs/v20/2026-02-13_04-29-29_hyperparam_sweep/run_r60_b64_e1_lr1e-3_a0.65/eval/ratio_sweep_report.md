# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-13 13:31:00 |

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
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0573 | 0.1513 | 0.2456 | 0.3395 | 0.4335 | 0.5278 | 0.6220 | 0.7159 | 0.8100 | 0.9047 | 0.9986 |
| QAT+Prune only | 0.7731 | 0.7951 | 0.8161 | 0.8383 | 0.8596 | 0.8800 | 0.9026 | 0.9237 | 0.9451 | 0.9663 | 0.9882 |
| QAT+PTQ | 0.7727 | 0.7949 | 0.8158 | 0.8382 | 0.8594 | 0.8795 | 0.9022 | 0.9237 | 0.9447 | 0.9659 | 0.9878 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7727 | 0.7949 | 0.8158 | 0.8382 | 0.8594 | 0.8795 | 0.9022 | 0.9237 | 0.9447 | 0.9659 | 0.9878 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1905 | 0.3462 | 0.4756 | 0.5851 | 0.6789 | 0.7602 | 0.8311 | 0.8937 | 0.9496 | 0.9993 |
| QAT+Prune only | 0.0000 | 0.4909 | 0.6825 | 0.7857 | 0.8492 | 0.8917 | 0.9241 | 0.9477 | 0.9664 | 0.9814 | 0.9941 |
| QAT+PTQ | 0.0000 | 0.4905 | 0.6820 | 0.7856 | 0.8489 | 0.8912 | 0.9238 | 0.9477 | 0.9662 | 0.9812 | 0.9939 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4905 | 0.6820 | 0.7856 | 0.8489 | 0.8912 | 0.9238 | 0.9477 | 0.9662 | 0.9812 | 0.9939 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0573 | 0.0572 | 0.0573 | 0.0570 | 0.0568 | 0.0569 | 0.0572 | 0.0563 | 0.0557 | 0.0593 | 0.0000 |
| QAT+Prune only | 0.7731 | 0.7737 | 0.7731 | 0.7741 | 0.7739 | 0.7719 | 0.7742 | 0.7732 | 0.7727 | 0.7691 | 0.0000 |
| QAT+PTQ | 0.7727 | 0.7734 | 0.7728 | 0.7741 | 0.7737 | 0.7711 | 0.7738 | 0.7739 | 0.7720 | 0.7688 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7727 | 0.7734 | 0.7728 | 0.7741 | 0.7737 | 0.7711 | 0.7738 | 0.7739 | 0.7720 | 0.7688 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0573 | 0.0000 | 0.0000 | 0.0000 | 0.0573 | 1.0000 |
| 90 | 10 | 299,940 | 0.1513 | 0.1053 | 0.9985 | 0.1905 | 0.0572 | 0.9970 |
| 80 | 20 | 291,350 | 0.2456 | 0.2094 | 0.9986 | 0.3462 | 0.0573 | 0.9940 |
| 70 | 30 | 194,230 | 0.3395 | 0.3122 | 0.9986 | 0.4756 | 0.0570 | 0.9897 |
| 60 | 40 | 145,675 | 0.4335 | 0.4138 | 0.9986 | 0.5851 | 0.0568 | 0.9840 |
| 50 | 50 | 116,540 | 0.5278 | 0.5143 | 0.9986 | 0.6789 | 0.0569 | 0.9762 |
| 40 | 60 | 97,115 | 0.6220 | 0.6137 | 0.9986 | 0.7602 | 0.0572 | 0.9648 |
| 30 | 70 | 83,240 | 0.7159 | 0.7117 | 0.9986 | 0.8311 | 0.0563 | 0.9455 |
| 20 | 80 | 72,835 | 0.8100 | 0.8088 | 0.9986 | 0.8937 | 0.0557 | 0.9093 |
| 10 | 90 | 64,740 | 0.9047 | 0.9053 | 0.9986 | 0.9496 | 0.0593 | 0.8258 |
| 0 | 100 | 58,270 | 0.9986 | 1.0000 | 0.9986 | 0.9993 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7731 | 0.0000 | 0.0000 | 0.0000 | 0.7731 | 1.0000 |
| 90 | 10 | 299,940 | 0.7951 | 0.3266 | 0.9879 | 0.4909 | 0.7737 | 0.9983 |
| 80 | 20 | 291,350 | 0.8161 | 0.5212 | 0.9882 | 0.6825 | 0.7731 | 0.9962 |
| 70 | 30 | 194,230 | 0.8383 | 0.6522 | 0.9882 | 0.7857 | 0.7741 | 0.9935 |
| 60 | 40 | 145,675 | 0.8596 | 0.7445 | 0.9882 | 0.8492 | 0.7739 | 0.9899 |
| 50 | 50 | 116,540 | 0.8800 | 0.8124 | 0.9882 | 0.8917 | 0.7719 | 0.9849 |
| 40 | 60 | 97,115 | 0.9026 | 0.8678 | 0.9882 | 0.9241 | 0.7742 | 0.9776 |
| 30 | 70 | 83,240 | 0.9237 | 0.9104 | 0.9882 | 0.9477 | 0.7732 | 0.9655 |
| 20 | 80 | 72,835 | 0.9451 | 0.9456 | 0.9882 | 0.9664 | 0.7727 | 0.9423 |
| 10 | 90 | 64,740 | 0.9663 | 0.9747 | 0.9882 | 0.9814 | 0.7691 | 0.8784 |
| 0 | 100 | 58,270 | 0.9882 | 1.0000 | 0.9882 | 0.9941 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7727 | 0.0000 | 0.0000 | 0.0000 | 0.7727 | 1.0000 |
| 90 | 10 | 299,940 | 0.7949 | 0.3263 | 0.9876 | 0.4905 | 0.7734 | 0.9982 |
| 80 | 20 | 291,350 | 0.8158 | 0.5208 | 0.9878 | 0.6820 | 0.7728 | 0.9961 |
| 70 | 30 | 194,230 | 0.8382 | 0.6521 | 0.9878 | 0.7856 | 0.7741 | 0.9933 |
| 60 | 40 | 145,675 | 0.8594 | 0.7442 | 0.9878 | 0.8489 | 0.7737 | 0.9896 |
| 50 | 50 | 116,540 | 0.8795 | 0.8119 | 0.9878 | 0.8912 | 0.7711 | 0.9845 |
| 40 | 60 | 97,115 | 0.9022 | 0.8675 | 0.9878 | 0.9238 | 0.7738 | 0.9770 |
| 30 | 70 | 83,240 | 0.9237 | 0.9107 | 0.9878 | 0.9477 | 0.7739 | 0.9646 |
| 20 | 80 | 72,835 | 0.9447 | 0.9454 | 0.9878 | 0.9662 | 0.7720 | 0.9407 |
| 10 | 90 | 64,740 | 0.9659 | 0.9747 | 0.9878 | 0.9812 | 0.7688 | 0.8753 |
| 0 | 100 | 58,270 | 0.9878 | 1.0000 | 0.9878 | 0.9939 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7727 | 0.0000 | 0.0000 | 0.0000 | 0.7727 | 1.0000 |
| 90 | 10 | 299,940 | 0.7949 | 0.3263 | 0.9876 | 0.4905 | 0.7734 | 0.9982 |
| 80 | 20 | 291,350 | 0.8158 | 0.5208 | 0.9878 | 0.6820 | 0.7728 | 0.9961 |
| 70 | 30 | 194,230 | 0.8382 | 0.6521 | 0.9878 | 0.7856 | 0.7741 | 0.9933 |
| 60 | 40 | 145,675 | 0.8594 | 0.7442 | 0.9878 | 0.8489 | 0.7737 | 0.9896 |
| 50 | 50 | 116,540 | 0.8795 | 0.8119 | 0.9878 | 0.8912 | 0.7711 | 0.9845 |
| 40 | 60 | 97,115 | 0.9022 | 0.8675 | 0.9878 | 0.9238 | 0.7738 | 0.9770 |
| 30 | 70 | 83,240 | 0.9237 | 0.9107 | 0.9878 | 0.9477 | 0.7739 | 0.9646 |
| 20 | 80 | 72,835 | 0.9447 | 0.9454 | 0.9878 | 0.9662 | 0.7720 | 0.9407 |
| 10 | 90 | 64,740 | 0.9659 | 0.9747 | 0.9878 | 0.9812 | 0.7688 | 0.8753 |
| 0 | 100 | 58,270 | 0.9878 | 1.0000 | 0.9878 | 0.9939 | 0.0000 | 0.0000 |


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
0.15       0.1513   0.1905   0.0572   0.9974   0.9986   0.1053   <--
0.20       0.1513   0.1905   0.0572   0.9974   0.9986   0.1053  
0.25       0.1513   0.1905   0.0572   0.9974   0.9986   0.1053  
0.30       0.1513   0.1905   0.0572   0.9974   0.9986   0.1053  
0.35       0.1513   0.1905   0.0572   0.9974   0.9986   0.1053  
0.40       0.1513   0.1905   0.0572   0.9974   0.9986   0.1053  
0.45       0.1513   0.1905   0.0572   0.9974   0.9986   0.1053  
0.50       0.1513   0.1905   0.0572   0.9974   0.9986   0.1053  
0.55       0.1513   0.1905   0.0572   0.9974   0.9986   0.1053  
0.60       0.1513   0.1905   0.0572   0.9974   0.9986   0.1053  
0.65       0.1513   0.1905   0.0572   0.9974   0.9986   0.1053  
0.70       0.1513   0.1905   0.0572   0.9974   0.9986   0.1053  
0.75       0.1513   0.1905   0.0572   0.9974   0.9986   0.1053  
0.80       0.1513   0.1905   0.0572   0.9974   0.9986   0.1053  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.1513, F1=0.1905, Normal Recall=0.0572, Normal Precision=0.9974, Attack Recall=0.9986, Attack Precision=0.1053

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
0.15       0.2454   0.3461   0.0571   0.9939   0.9986   0.2093   <--
0.20       0.2454   0.3461   0.0571   0.9939   0.9986   0.2093  
0.25       0.2454   0.3461   0.0571   0.9939   0.9986   0.2093  
0.30       0.2454   0.3461   0.0571   0.9939   0.9986   0.2093  
0.35       0.2454   0.3461   0.0571   0.9939   0.9986   0.2093  
0.40       0.2454   0.3461   0.0571   0.9939   0.9986   0.2093  
0.45       0.2454   0.3461   0.0571   0.9939   0.9986   0.2093  
0.50       0.2454   0.3461   0.0571   0.9939   0.9986   0.2093  
0.55       0.2454   0.3461   0.0571   0.9939   0.9986   0.2093  
0.60       0.2454   0.3461   0.0571   0.9939   0.9986   0.2093  
0.65       0.2454   0.3461   0.0571   0.9939   0.9986   0.2093  
0.70       0.2454   0.3461   0.0571   0.9939   0.9986   0.2093  
0.75       0.2454   0.3461   0.0571   0.9939   0.9986   0.2093  
0.80       0.2454   0.3461   0.0571   0.9939   0.9986   0.2093  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2454, F1=0.3461, Normal Recall=0.0571, Normal Precision=0.9939, Attack Recall=0.9986, Attack Precision=0.2093

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
0.15       0.3401   0.4759   0.0579   0.9898   0.9986   0.3124   <--
0.20       0.3401   0.4759   0.0579   0.9898   0.9986   0.3124  
0.25       0.3401   0.4759   0.0579   0.9898   0.9986   0.3124  
0.30       0.3401   0.4759   0.0579   0.9898   0.9986   0.3124  
0.35       0.3401   0.4759   0.0579   0.9898   0.9986   0.3124  
0.40       0.3401   0.4759   0.0579   0.9898   0.9986   0.3124  
0.45       0.3401   0.4759   0.0579   0.9898   0.9986   0.3124  
0.50       0.3401   0.4759   0.0579   0.9898   0.9986   0.3124  
0.55       0.3401   0.4759   0.0579   0.9898   0.9986   0.3124  
0.60       0.3401   0.4759   0.0579   0.9898   0.9986   0.3124  
0.65       0.3401   0.4759   0.0579   0.9898   0.9986   0.3124  
0.70       0.3401   0.4759   0.0579   0.9898   0.9986   0.3124  
0.75       0.3401   0.4759   0.0579   0.9898   0.9986   0.3124  
0.80       0.3401   0.4759   0.0579   0.9898   0.9986   0.3124  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3401, F1=0.4759, Normal Recall=0.0579, Normal Precision=0.9898, Attack Recall=0.9986, Attack Precision=0.3124

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
0.15       0.4338   0.5852   0.0573   0.9841   0.9986   0.4139   <--
0.20       0.4338   0.5852   0.0573   0.9841   0.9986   0.4139  
0.25       0.4338   0.5852   0.0573   0.9841   0.9986   0.4139  
0.30       0.4338   0.5852   0.0573   0.9841   0.9986   0.4139  
0.35       0.4338   0.5852   0.0573   0.9841   0.9986   0.4139  
0.40       0.4338   0.5852   0.0573   0.9841   0.9986   0.4139  
0.45       0.4338   0.5852   0.0573   0.9841   0.9986   0.4139  
0.50       0.4338   0.5852   0.0573   0.9841   0.9986   0.4139  
0.55       0.4338   0.5852   0.0573   0.9841   0.9986   0.4139  
0.60       0.4338   0.5852   0.0573   0.9841   0.9986   0.4139  
0.65       0.4338   0.5852   0.0573   0.9841   0.9986   0.4139  
0.70       0.4338   0.5852   0.0573   0.9841   0.9986   0.4139  
0.75       0.4338   0.5852   0.0573   0.9841   0.9986   0.4139  
0.80       0.4338   0.5852   0.0573   0.9841   0.9986   0.4139  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4338, F1=0.5852, Normal Recall=0.0573, Normal Precision=0.9841, Attack Recall=0.9986, Attack Precision=0.4139

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
0.15       0.5279   0.6790   0.0572   0.9763   0.9986   0.5144   <--
0.20       0.5279   0.6790   0.0572   0.9763   0.9986   0.5144  
0.25       0.5279   0.6790   0.0572   0.9763   0.9986   0.5144  
0.30       0.5279   0.6790   0.0572   0.9763   0.9986   0.5144  
0.35       0.5279   0.6790   0.0572   0.9763   0.9986   0.5144  
0.40       0.5279   0.6790   0.0572   0.9763   0.9986   0.5144  
0.45       0.5279   0.6790   0.0572   0.9763   0.9986   0.5144  
0.50       0.5279   0.6790   0.0572   0.9763   0.9986   0.5144  
0.55       0.5279   0.6790   0.0572   0.9763   0.9986   0.5144  
0.60       0.5279   0.6790   0.0572   0.9763   0.9986   0.5144  
0.65       0.5279   0.6790   0.0572   0.9763   0.9986   0.5144  
0.70       0.5279   0.6790   0.0572   0.9763   0.9986   0.5144  
0.75       0.5279   0.6790   0.0572   0.9763   0.9986   0.5144  
0.80       0.5279   0.6790   0.0572   0.9763   0.9986   0.5144  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5279, F1=0.6790, Normal Recall=0.0572, Normal Precision=0.9763, Attack Recall=0.9986, Attack Precision=0.5144

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
0.15       0.7952   0.4910   0.7737   0.9983   0.9882   0.3267   <--
0.20       0.7952   0.4910   0.7737   0.9983   0.9882   0.3267  
0.25       0.7952   0.4910   0.7737   0.9983   0.9882   0.3267  
0.30       0.7952   0.4910   0.7737   0.9983   0.9882   0.3267  
0.35       0.7952   0.4910   0.7737   0.9983   0.9882   0.3267  
0.40       0.7952   0.4910   0.7737   0.9983   0.9882   0.3267  
0.45       0.7952   0.4910   0.7737   0.9983   0.9882   0.3267  
0.50       0.7952   0.4910   0.7737   0.9983   0.9882   0.3267  
0.55       0.7952   0.4910   0.7737   0.9983   0.9882   0.3267  
0.60       0.7952   0.4910   0.7737   0.9983   0.9882   0.3267  
0.65       0.7952   0.4910   0.7737   0.9983   0.9882   0.3267  
0.70       0.7952   0.4910   0.7737   0.9983   0.9882   0.3267  
0.75       0.7952   0.4910   0.7737   0.9983   0.9882   0.3267  
0.80       0.7952   0.4910   0.7737   0.9983   0.9882   0.3267  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7952, F1=0.4910, Normal Recall=0.7737, Normal Precision=0.9983, Attack Recall=0.9882, Attack Precision=0.3267

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
0.15       0.8166   0.6831   0.7737   0.9962   0.9882   0.5220   <--
0.20       0.8166   0.6831   0.7737   0.9962   0.9882   0.5220  
0.25       0.8166   0.6831   0.7737   0.9962   0.9882   0.5220  
0.30       0.8166   0.6831   0.7737   0.9962   0.9882   0.5220  
0.35       0.8166   0.6831   0.7737   0.9962   0.9882   0.5220  
0.40       0.8166   0.6831   0.7737   0.9962   0.9882   0.5220  
0.45       0.8166   0.6831   0.7737   0.9962   0.9882   0.5220  
0.50       0.8166   0.6831   0.7737   0.9962   0.9882   0.5220  
0.55       0.8166   0.6831   0.7737   0.9962   0.9882   0.5220  
0.60       0.8166   0.6831   0.7737   0.9962   0.9882   0.5220  
0.65       0.8166   0.6831   0.7737   0.9962   0.9882   0.5220  
0.70       0.8166   0.6831   0.7737   0.9962   0.9882   0.5220  
0.75       0.8166   0.6831   0.7737   0.9962   0.9882   0.5220  
0.80       0.8166   0.6831   0.7737   0.9962   0.9882   0.5220  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8166, F1=0.6831, Normal Recall=0.7737, Normal Precision=0.9962, Attack Recall=0.9882, Attack Precision=0.5220

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
0.15       0.8380   0.7854   0.7737   0.9935   0.9882   0.6517   <--
0.20       0.8380   0.7854   0.7737   0.9935   0.9882   0.6517  
0.25       0.8380   0.7854   0.7737   0.9935   0.9882   0.6517  
0.30       0.8380   0.7854   0.7737   0.9935   0.9882   0.6517  
0.35       0.8380   0.7854   0.7737   0.9935   0.9882   0.6517  
0.40       0.8380   0.7854   0.7737   0.9935   0.9882   0.6517  
0.45       0.8380   0.7854   0.7737   0.9935   0.9882   0.6517  
0.50       0.8380   0.7854   0.7737   0.9935   0.9882   0.6517  
0.55       0.8380   0.7854   0.7737   0.9935   0.9882   0.6517  
0.60       0.8380   0.7854   0.7737   0.9935   0.9882   0.6517  
0.65       0.8380   0.7854   0.7737   0.9935   0.9882   0.6517  
0.70       0.8380   0.7854   0.7737   0.9935   0.9882   0.6517  
0.75       0.8380   0.7854   0.7737   0.9935   0.9882   0.6517  
0.80       0.8380   0.7854   0.7737   0.9935   0.9882   0.6517  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8380, F1=0.7854, Normal Recall=0.7737, Normal Precision=0.9935, Attack Recall=0.9882, Attack Precision=0.6517

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
0.15       0.8592   0.8488   0.7732   0.9899   0.9882   0.7439   <--
0.20       0.8592   0.8488   0.7732   0.9899   0.9882   0.7439  
0.25       0.8592   0.8488   0.7732   0.9899   0.9882   0.7439  
0.30       0.8592   0.8488   0.7732   0.9899   0.9882   0.7439  
0.35       0.8592   0.8488   0.7732   0.9899   0.9882   0.7439  
0.40       0.8592   0.8488   0.7732   0.9899   0.9882   0.7439  
0.45       0.8592   0.8488   0.7732   0.9899   0.9882   0.7439  
0.50       0.8592   0.8488   0.7732   0.9899   0.9882   0.7439  
0.55       0.8592   0.8488   0.7732   0.9899   0.9882   0.7439  
0.60       0.8592   0.8488   0.7732   0.9899   0.9882   0.7439  
0.65       0.8592   0.8488   0.7732   0.9899   0.9882   0.7439  
0.70       0.8592   0.8488   0.7732   0.9899   0.9882   0.7439  
0.75       0.8592   0.8488   0.7732   0.9899   0.9882   0.7439  
0.80       0.8592   0.8488   0.7732   0.9899   0.9882   0.7439  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8592, F1=0.8488, Normal Recall=0.7732, Normal Precision=0.9899, Attack Recall=0.9882, Attack Precision=0.7439

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
0.15       0.8798   0.8915   0.7713   0.9849   0.9882   0.8121   <--
0.20       0.8798   0.8915   0.7713   0.9849   0.9882   0.8121  
0.25       0.8798   0.8915   0.7713   0.9849   0.9882   0.8121  
0.30       0.8798   0.8915   0.7713   0.9849   0.9882   0.8121  
0.35       0.8798   0.8915   0.7713   0.9849   0.9882   0.8121  
0.40       0.8798   0.8915   0.7713   0.9849   0.9882   0.8121  
0.45       0.8798   0.8915   0.7713   0.9849   0.9882   0.8121  
0.50       0.8798   0.8915   0.7713   0.9849   0.9882   0.8121  
0.55       0.8798   0.8915   0.7713   0.9849   0.9882   0.8121  
0.60       0.8798   0.8915   0.7713   0.9849   0.9882   0.8121  
0.65       0.8798   0.8915   0.7713   0.9849   0.9882   0.8121  
0.70       0.8798   0.8915   0.7713   0.9849   0.9882   0.8121  
0.75       0.8798   0.8915   0.7713   0.9849   0.9882   0.8121  
0.80       0.8798   0.8915   0.7713   0.9849   0.9882   0.8121  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8798, F1=0.8915, Normal Recall=0.7713, Normal Precision=0.9849, Attack Recall=0.9882, Attack Precision=0.8121

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
0.15       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264   <--
0.20       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.25       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.30       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.35       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.40       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.45       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.50       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.55       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.60       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.65       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.70       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.75       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.80       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7949, F1=0.4906, Normal Recall=0.7734, Normal Precision=0.9983, Attack Recall=0.9878, Attack Precision=0.3264

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
0.15       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216   <--
0.20       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.25       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.30       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.35       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.40       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.45       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.50       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.55       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.60       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.65       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.70       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.75       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.80       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8163, F1=0.6827, Normal Recall=0.7735, Normal Precision=0.9961, Attack Recall=0.9878, Attack Precision=0.5216

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
0.15       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511   <--
0.20       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.25       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.30       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.35       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.40       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.45       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.50       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.55       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.60       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.65       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.70       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.75       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.80       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8376, F1=0.7849, Normal Recall=0.7732, Normal Precision=0.9933, Attack Recall=0.9878, Attack Precision=0.6511

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
0.15       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436   <--
0.20       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.25       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.30       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.35       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.40       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.45       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.50       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.55       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.60       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.65       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.70       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.75       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.80       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8589, F1=0.8485, Normal Recall=0.7730, Normal Precision=0.9896, Attack Recall=0.9878, Attack Precision=0.7436

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
0.15       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117   <--
0.20       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.25       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.30       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.35       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.40       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.45       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.50       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.55       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.60       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.65       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.70       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.75       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.80       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8793, F1=0.8912, Normal Recall=0.7709, Normal Precision=0.9845, Attack Recall=0.9878, Attack Precision=0.8117

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
0.15       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264   <--
0.20       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.25       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.30       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.35       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.40       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.45       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.50       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.55       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.60       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.65       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.70       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.75       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
0.80       0.7949   0.4906   0.7734   0.9983   0.9878   0.3264  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7949, F1=0.4906, Normal Recall=0.7734, Normal Precision=0.9983, Attack Recall=0.9878, Attack Precision=0.3264

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
0.15       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216   <--
0.20       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.25       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.30       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.35       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.40       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.45       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.50       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.55       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.60       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.65       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.70       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.75       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
0.80       0.8163   0.6827   0.7735   0.9961   0.9878   0.5216  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8163, F1=0.6827, Normal Recall=0.7735, Normal Precision=0.9961, Attack Recall=0.9878, Attack Precision=0.5216

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
0.15       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511   <--
0.20       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.25       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.30       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.35       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.40       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.45       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.50       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.55       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.60       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.65       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.70       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.75       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
0.80       0.8376   0.7849   0.7732   0.9933   0.9878   0.6511  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8376, F1=0.7849, Normal Recall=0.7732, Normal Precision=0.9933, Attack Recall=0.9878, Attack Precision=0.6511

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
0.15       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436   <--
0.20       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.25       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.30       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.35       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.40       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.45       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.50       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.55       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.60       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.65       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.70       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.75       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
0.80       0.8589   0.8485   0.7730   0.9896   0.9878   0.7436  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8589, F1=0.8485, Normal Recall=0.7730, Normal Precision=0.9896, Attack Recall=0.9878, Attack Precision=0.7436

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
0.15       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117   <--
0.20       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.25       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.30       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.35       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.40       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.45       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.50       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.55       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.60       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.65       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.70       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.75       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
0.80       0.8793   0.8912   0.7709   0.9845   0.9878   0.8117  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8793, F1=0.8912, Normal Recall=0.7709, Normal Precision=0.9845, Attack Recall=0.9878, Attack Precision=0.8117

```

