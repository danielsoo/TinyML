# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e2_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-15 20:59:11 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1442 | 0.2212 | 0.2983 | 0.3749 | 0.4527 | 0.5303 | 0.6075 | 0.6839 | 0.7608 | 0.8385 | 0.9155 |
| QAT+Prune only | 0.9809 | 0.9514 | 0.9228 | 0.8941 | 0.8649 | 0.8365 | 0.8077 | 0.7787 | 0.7501 | 0.7213 | 0.6926 |
| QAT+PTQ | 0.9799 | 0.9503 | 0.9216 | 0.8930 | 0.8635 | 0.8351 | 0.8061 | 0.7771 | 0.7482 | 0.7193 | 0.6905 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9799 | 0.9503 | 0.9216 | 0.8930 | 0.8635 | 0.8351 | 0.8061 | 0.7771 | 0.7482 | 0.7193 | 0.6905 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1903 | 0.3429 | 0.4677 | 0.5723 | 0.6609 | 0.7368 | 0.8022 | 0.8596 | 0.9107 | 0.9559 |
| QAT+Prune only | 0.0000 | 0.7399 | 0.7820 | 0.7970 | 0.8040 | 0.8091 | 0.8121 | 0.8142 | 0.8160 | 0.8173 | 0.8184 |
| QAT+PTQ | 0.0000 | 0.7351 | 0.7789 | 0.7947 | 0.8019 | 0.8072 | 0.8104 | 0.8126 | 0.8144 | 0.8158 | 0.8169 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.7351 | 0.7789 | 0.7947 | 0.8019 | 0.8072 | 0.8104 | 0.8126 | 0.8144 | 0.8158 | 0.8169 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1442 | 0.1441 | 0.1440 | 0.1432 | 0.1441 | 0.1450 | 0.1455 | 0.1434 | 0.1418 | 0.1455 | 0.0000 |
| QAT+Prune only | 0.9809 | 0.9802 | 0.9803 | 0.9805 | 0.9798 | 0.9805 | 0.9802 | 0.9797 | 0.9799 | 0.9793 | 0.0000 |
| QAT+PTQ | 0.9799 | 0.9793 | 0.9794 | 0.9797 | 0.9788 | 0.9796 | 0.9795 | 0.9789 | 0.9789 | 0.9782 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9799 | 0.9793 | 0.9794 | 0.9797 | 0.9788 | 0.9796 | 0.9795 | 0.9789 | 0.9789 | 0.9782 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1442 | 0.0000 | 0.0000 | 0.0000 | 0.1442 | 1.0000 |
| 90 | 10 | 299,940 | 0.2212 | 0.1062 | 0.9151 | 0.1903 | 0.1441 | 0.9386 |
| 80 | 20 | 291,350 | 0.2983 | 0.2110 | 0.9155 | 0.3429 | 0.1440 | 0.8721 |
| 70 | 30 | 194,230 | 0.3749 | 0.3141 | 0.9155 | 0.4677 | 0.1432 | 0.7982 |
| 60 | 40 | 145,675 | 0.4527 | 0.4163 | 0.9155 | 0.5723 | 0.1441 | 0.7190 |
| 50 | 50 | 116,540 | 0.5303 | 0.5171 | 0.9155 | 0.6609 | 0.1450 | 0.6319 |
| 40 | 60 | 97,115 | 0.6075 | 0.6164 | 0.9155 | 0.7368 | 0.1455 | 0.5344 |
| 30 | 70 | 83,240 | 0.6839 | 0.7138 | 0.9155 | 0.8022 | 0.1434 | 0.4211 |
| 20 | 80 | 72,835 | 0.7608 | 0.8101 | 0.9155 | 0.8596 | 0.1418 | 0.2956 |
| 10 | 90 | 64,740 | 0.8385 | 0.9060 | 0.9155 | 0.9107 | 0.1455 | 0.1606 |
| 0 | 100 | 58,270 | 0.9155 | 1.0000 | 0.9155 | 0.9559 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9809 | 0.0000 | 0.0000 | 0.0000 | 0.9809 | 1.0000 |
| 90 | 10 | 299,940 | 0.9514 | 0.7952 | 0.6918 | 0.7399 | 0.9802 | 0.9662 |
| 80 | 20 | 291,350 | 0.9228 | 0.8978 | 0.6926 | 0.7820 | 0.9803 | 0.9273 |
| 70 | 30 | 194,230 | 0.8941 | 0.9384 | 0.6926 | 0.7970 | 0.9805 | 0.8816 |
| 60 | 40 | 145,675 | 0.8649 | 0.9581 | 0.6926 | 0.8040 | 0.9798 | 0.8270 |
| 50 | 50 | 116,540 | 0.8365 | 0.9726 | 0.6926 | 0.8091 | 0.9805 | 0.7613 |
| 40 | 60 | 97,115 | 0.8077 | 0.9813 | 0.6926 | 0.8121 | 0.9802 | 0.6801 |
| 30 | 70 | 83,240 | 0.7787 | 0.9876 | 0.6926 | 0.8142 | 0.9797 | 0.5773 |
| 20 | 80 | 72,835 | 0.7501 | 0.9928 | 0.6926 | 0.8160 | 0.9799 | 0.4435 |
| 10 | 90 | 64,740 | 0.7213 | 0.9967 | 0.6926 | 0.8173 | 0.9793 | 0.2614 |
| 0 | 100 | 58,270 | 0.6926 | 1.0000 | 0.6926 | 0.8184 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9799 | 0.0000 | 0.0000 | 0.0000 | 0.9799 | 1.0000 |
| 90 | 10 | 299,940 | 0.9503 | 0.7872 | 0.6895 | 0.7351 | 0.9793 | 0.9660 |
| 80 | 20 | 291,350 | 0.9216 | 0.8932 | 0.6905 | 0.7789 | 0.9794 | 0.9268 |
| 70 | 30 | 194,230 | 0.8930 | 0.9359 | 0.6905 | 0.7947 | 0.9797 | 0.8808 |
| 60 | 40 | 145,675 | 0.8635 | 0.9560 | 0.6905 | 0.8019 | 0.9788 | 0.8259 |
| 50 | 50 | 116,540 | 0.8351 | 0.9713 | 0.6905 | 0.8072 | 0.9796 | 0.7599 |
| 40 | 60 | 97,115 | 0.8061 | 0.9806 | 0.6906 | 0.8104 | 0.9795 | 0.6785 |
| 30 | 70 | 83,240 | 0.7771 | 0.9871 | 0.6905 | 0.8126 | 0.9789 | 0.5755 |
| 20 | 80 | 72,835 | 0.7482 | 0.9924 | 0.6906 | 0.8144 | 0.9789 | 0.4416 |
| 10 | 90 | 64,740 | 0.7193 | 0.9965 | 0.6905 | 0.8158 | 0.9782 | 0.2599 |
| 0 | 100 | 58,270 | 0.6905 | 1.0000 | 0.6905 | 0.8169 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9799 | 0.0000 | 0.0000 | 0.0000 | 0.9799 | 1.0000 |
| 90 | 10 | 299,940 | 0.9503 | 0.7872 | 0.6895 | 0.7351 | 0.9793 | 0.9660 |
| 80 | 20 | 291,350 | 0.9216 | 0.8932 | 0.6905 | 0.7789 | 0.9794 | 0.9268 |
| 70 | 30 | 194,230 | 0.8930 | 0.9359 | 0.6905 | 0.7947 | 0.9797 | 0.8808 |
| 60 | 40 | 145,675 | 0.8635 | 0.9560 | 0.6905 | 0.8019 | 0.9788 | 0.8259 |
| 50 | 50 | 116,540 | 0.8351 | 0.9713 | 0.6905 | 0.8072 | 0.9796 | 0.7599 |
| 40 | 60 | 97,115 | 0.8061 | 0.9806 | 0.6906 | 0.8104 | 0.9795 | 0.6785 |
| 30 | 70 | 83,240 | 0.7771 | 0.9871 | 0.6905 | 0.8126 | 0.9789 | 0.5755 |
| 20 | 80 | 72,835 | 0.7482 | 0.9924 | 0.6906 | 0.8144 | 0.9789 | 0.4416 |
| 10 | 90 | 64,740 | 0.7193 | 0.9965 | 0.6905 | 0.8158 | 0.9782 | 0.2599 |
| 0 | 100 | 58,270 | 0.6905 | 1.0000 | 0.6905 | 0.8169 | 0.0000 | 0.0000 |


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
0.15       0.2211   0.1901   0.1441   0.9380   0.9143   0.1061   <--
0.20       0.2211   0.1901   0.1441   0.9380   0.9143   0.1061  
0.25       0.2211   0.1901   0.1441   0.9380   0.9143   0.1061  
0.30       0.2211   0.1901   0.1441   0.9380   0.9143   0.1061  
0.35       0.2211   0.1901   0.1441   0.9380   0.9143   0.1061  
0.40       0.2211   0.1901   0.1441   0.9380   0.9143   0.1061  
0.45       0.2211   0.1901   0.1441   0.9380   0.9143   0.1061  
0.50       0.2211   0.1901   0.1441   0.9380   0.9143   0.1061  
0.55       0.2211   0.1901   0.1441   0.9380   0.9143   0.1061  
0.60       0.2211   0.1901   0.1441   0.9380   0.9143   0.1061  
0.65       0.2211   0.1901   0.1441   0.9380   0.9143   0.1061  
0.70       0.2211   0.1901   0.1441   0.9380   0.9143   0.1061  
0.75       0.2211   0.1901   0.1441   0.9380   0.9143   0.1061  
0.80       0.2211   0.1901   0.1441   0.9380   0.9143   0.1061  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2211, F1=0.1901, Normal Recall=0.1441, Normal Precision=0.9380, Attack Recall=0.9143, Attack Precision=0.1061

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
0.15       0.2982   0.3429   0.1438   0.8719   0.9155   0.2109   <--
0.20       0.2982   0.3429   0.1438   0.8719   0.9155   0.2109  
0.25       0.2982   0.3429   0.1438   0.8719   0.9155   0.2109  
0.30       0.2982   0.3429   0.1438   0.8719   0.9155   0.2109  
0.35       0.2982   0.3429   0.1438   0.8719   0.9155   0.2109  
0.40       0.2982   0.3429   0.1438   0.8719   0.9155   0.2109  
0.45       0.2982   0.3429   0.1438   0.8719   0.9155   0.2109  
0.50       0.2982   0.3429   0.1438   0.8719   0.9155   0.2109  
0.55       0.2982   0.3429   0.1438   0.8719   0.9155   0.2109  
0.60       0.2982   0.3429   0.1438   0.8719   0.9155   0.2109  
0.65       0.2982   0.3429   0.1438   0.8719   0.9155   0.2109  
0.70       0.2982   0.3429   0.1438   0.8719   0.9155   0.2109  
0.75       0.2982   0.3429   0.1438   0.8719   0.9155   0.2109  
0.80       0.2982   0.3429   0.1438   0.8719   0.9155   0.2109  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2982, F1=0.3429, Normal Recall=0.1438, Normal Precision=0.8719, Attack Recall=0.9155, Attack Precision=0.2109

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
0.15       0.3763   0.4683   0.1453   0.8005   0.9155   0.3146   <--
0.20       0.3763   0.4683   0.1453   0.8005   0.9155   0.3146  
0.25       0.3763   0.4683   0.1453   0.8005   0.9155   0.3146  
0.30       0.3763   0.4683   0.1453   0.8005   0.9155   0.3146  
0.35       0.3763   0.4683   0.1453   0.8005   0.9155   0.3146  
0.40       0.3763   0.4683   0.1453   0.8005   0.9155   0.3146  
0.45       0.3763   0.4683   0.1453   0.8005   0.9155   0.3146  
0.50       0.3763   0.4683   0.1453   0.8005   0.9155   0.3146  
0.55       0.3763   0.4683   0.1453   0.8005   0.9155   0.3146  
0.60       0.3763   0.4683   0.1453   0.8005   0.9155   0.3146  
0.65       0.3763   0.4683   0.1453   0.8005   0.9155   0.3146  
0.70       0.3763   0.4683   0.1453   0.8005   0.9155   0.3146  
0.75       0.3763   0.4683   0.1453   0.8005   0.9155   0.3146  
0.80       0.3763   0.4683   0.1453   0.8005   0.9155   0.3146  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3763, F1=0.4683, Normal Recall=0.1453, Normal Precision=0.8005, Attack Recall=0.9155, Attack Precision=0.3146

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
0.15       0.4528   0.5724   0.1443   0.7192   0.9155   0.4163   <--
0.20       0.4528   0.5724   0.1443   0.7192   0.9155   0.4163  
0.25       0.4528   0.5724   0.1443   0.7192   0.9155   0.4163  
0.30       0.4528   0.5724   0.1443   0.7192   0.9155   0.4163  
0.35       0.4528   0.5724   0.1443   0.7192   0.9155   0.4163  
0.40       0.4528   0.5724   0.1443   0.7192   0.9155   0.4163  
0.45       0.4528   0.5724   0.1443   0.7192   0.9155   0.4163  
0.50       0.4528   0.5724   0.1443   0.7192   0.9155   0.4163  
0.55       0.4528   0.5724   0.1443   0.7192   0.9155   0.4163  
0.60       0.4528   0.5724   0.1443   0.7192   0.9155   0.4163  
0.65       0.4528   0.5724   0.1443   0.7192   0.9155   0.4163  
0.70       0.4528   0.5724   0.1443   0.7192   0.9155   0.4163  
0.75       0.4528   0.5724   0.1443   0.7192   0.9155   0.4163  
0.80       0.4528   0.5724   0.1443   0.7192   0.9155   0.4163  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4528, F1=0.5724, Normal Recall=0.1443, Normal Precision=0.7192, Attack Recall=0.9155, Attack Precision=0.4163

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
0.15       0.5300   0.6608   0.1444   0.6309   0.9155   0.5169   <--
0.20       0.5300   0.6608   0.1444   0.6309   0.9155   0.5169  
0.25       0.5300   0.6608   0.1444   0.6309   0.9155   0.5169  
0.30       0.5300   0.6608   0.1444   0.6309   0.9155   0.5169  
0.35       0.5300   0.6608   0.1444   0.6309   0.9155   0.5169  
0.40       0.5300   0.6608   0.1444   0.6309   0.9155   0.5169  
0.45       0.5300   0.6608   0.1444   0.6309   0.9155   0.5169  
0.50       0.5300   0.6608   0.1444   0.6309   0.9155   0.5169  
0.55       0.5300   0.6608   0.1444   0.6309   0.9155   0.5169  
0.60       0.5300   0.6608   0.1444   0.6309   0.9155   0.5169  
0.65       0.5300   0.6608   0.1444   0.6309   0.9155   0.5169  
0.70       0.5300   0.6608   0.1444   0.6309   0.9155   0.5169  
0.75       0.5300   0.6608   0.1444   0.6309   0.9155   0.5169  
0.80       0.5300   0.6608   0.1444   0.6309   0.9155   0.5169  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5300, F1=0.6608, Normal Recall=0.1444, Normal Precision=0.6309, Attack Recall=0.9155, Attack Precision=0.5169

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
0.15       0.9518   0.7429   0.9802   0.9667   0.6963   0.7963   <--
0.20       0.9518   0.7429   0.9802   0.9667   0.6963   0.7963  
0.25       0.9518   0.7429   0.9802   0.9667   0.6963   0.7963  
0.30       0.9518   0.7429   0.9802   0.9667   0.6963   0.7963  
0.35       0.9518   0.7429   0.9802   0.9667   0.6963   0.7963  
0.40       0.9518   0.7429   0.9802   0.9667   0.6963   0.7963  
0.45       0.9518   0.7429   0.9802   0.9667   0.6963   0.7963  
0.50       0.9518   0.7429   0.9802   0.9667   0.6963   0.7963  
0.55       0.9518   0.7429   0.9802   0.9667   0.6963   0.7963  
0.60       0.9518   0.7429   0.9802   0.9667   0.6963   0.7963  
0.65       0.9518   0.7429   0.9802   0.9667   0.6963   0.7963  
0.70       0.9518   0.7429   0.9802   0.9667   0.6963   0.7963  
0.75       0.9518   0.7429   0.9802   0.9667   0.6963   0.7963  
0.80       0.9518   0.7429   0.9802   0.9667   0.6963   0.7963  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9518, F1=0.7429, Normal Recall=0.9802, Normal Precision=0.9667, Attack Recall=0.6963, Attack Precision=0.7963

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
0.15       0.9227   0.7819   0.9802   0.9273   0.6926   0.8976   <--
0.20       0.9227   0.7819   0.9802   0.9273   0.6926   0.8976  
0.25       0.9227   0.7819   0.9802   0.9273   0.6926   0.8976  
0.30       0.9227   0.7819   0.9802   0.9273   0.6926   0.8976  
0.35       0.9227   0.7819   0.9802   0.9273   0.6926   0.8976  
0.40       0.9227   0.7819   0.9802   0.9273   0.6926   0.8976  
0.45       0.9227   0.7819   0.9802   0.9273   0.6926   0.8976  
0.50       0.9227   0.7819   0.9802   0.9273   0.6926   0.8976  
0.55       0.9227   0.7819   0.9802   0.9273   0.6926   0.8976  
0.60       0.9227   0.7819   0.9802   0.9273   0.6926   0.8976  
0.65       0.9227   0.7819   0.9802   0.9273   0.6926   0.8976  
0.70       0.9227   0.7819   0.9802   0.9273   0.6926   0.8976  
0.75       0.9227   0.7819   0.9802   0.9273   0.6926   0.8976  
0.80       0.9227   0.7819   0.9802   0.9273   0.6926   0.8976  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9227, F1=0.7819, Normal Recall=0.9802, Normal Precision=0.9273, Attack Recall=0.6926, Attack Precision=0.8976

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
0.15       0.8942   0.7971   0.9806   0.8816   0.6926   0.9388   <--
0.20       0.8942   0.7971   0.9806   0.8816   0.6926   0.9388  
0.25       0.8942   0.7971   0.9806   0.8816   0.6926   0.9388  
0.30       0.8942   0.7971   0.9806   0.8816   0.6926   0.9388  
0.35       0.8942   0.7971   0.9806   0.8816   0.6926   0.9388  
0.40       0.8942   0.7971   0.9806   0.8816   0.6926   0.9388  
0.45       0.8942   0.7971   0.9806   0.8816   0.6926   0.9388  
0.50       0.8942   0.7971   0.9806   0.8816   0.6926   0.9388  
0.55       0.8942   0.7971   0.9806   0.8816   0.6926   0.9388  
0.60       0.8942   0.7971   0.9806   0.8816   0.6926   0.9388  
0.65       0.8942   0.7971   0.9806   0.8816   0.6926   0.9388  
0.70       0.8942   0.7971   0.9806   0.8816   0.6926   0.9388  
0.75       0.8942   0.7971   0.9806   0.8816   0.6926   0.9388  
0.80       0.8942   0.7971   0.9806   0.8816   0.6926   0.9388  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8942, F1=0.7971, Normal Recall=0.9806, Normal Precision=0.8816, Attack Recall=0.6926, Attack Precision=0.9388

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
0.15       0.8656   0.8048   0.9810   0.8272   0.6926   0.9605   <--
0.20       0.8656   0.8048   0.9810   0.8272   0.6926   0.9605  
0.25       0.8656   0.8048   0.9810   0.8272   0.6926   0.9605  
0.30       0.8656   0.8048   0.9810   0.8272   0.6926   0.9605  
0.35       0.8656   0.8048   0.9810   0.8272   0.6926   0.9605  
0.40       0.8656   0.8048   0.9810   0.8272   0.6926   0.9605  
0.45       0.8656   0.8048   0.9810   0.8272   0.6926   0.9605  
0.50       0.8656   0.8048   0.9810   0.8272   0.6926   0.9605  
0.55       0.8656   0.8048   0.9810   0.8272   0.6926   0.9605  
0.60       0.8656   0.8048   0.9810   0.8272   0.6926   0.9605  
0.65       0.8656   0.8048   0.9810   0.8272   0.6926   0.9605  
0.70       0.8656   0.8048   0.9810   0.8272   0.6926   0.9605  
0.75       0.8656   0.8048   0.9810   0.8272   0.6926   0.9605  
0.80       0.8656   0.8048   0.9810   0.8272   0.6926   0.9605  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8656, F1=0.8048, Normal Recall=0.9810, Normal Precision=0.8272, Attack Recall=0.6926, Attack Precision=0.9605

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
0.15       0.8367   0.8092   0.9807   0.7614   0.6926   0.9730   <--
0.20       0.8367   0.8092   0.9807   0.7614   0.6926   0.9730  
0.25       0.8367   0.8092   0.9807   0.7614   0.6926   0.9730  
0.30       0.8367   0.8092   0.9807   0.7614   0.6926   0.9730  
0.35       0.8367   0.8092   0.9807   0.7614   0.6926   0.9730  
0.40       0.8367   0.8092   0.9807   0.7614   0.6926   0.9730  
0.45       0.8367   0.8092   0.9807   0.7614   0.6926   0.9730  
0.50       0.8367   0.8092   0.9807   0.7614   0.6926   0.9730  
0.55       0.8367   0.8092   0.9807   0.7614   0.6926   0.9730  
0.60       0.8367   0.8092   0.9807   0.7614   0.6926   0.9730  
0.65       0.8367   0.8092   0.9807   0.7614   0.6926   0.9730  
0.70       0.8367   0.8092   0.9807   0.7614   0.6926   0.9730  
0.75       0.8367   0.8092   0.9807   0.7614   0.6926   0.9730  
0.80       0.8367   0.8092   0.9807   0.7614   0.6926   0.9730  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8367, F1=0.8092, Normal Recall=0.9807, Normal Precision=0.7614, Attack Recall=0.6926, Attack Precision=0.9730

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
0.15       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883   <--
0.20       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.25       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.30       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.35       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.40       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.45       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.50       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.55       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.60       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.65       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.70       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.75       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.80       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9507, F1=0.7379, Normal Recall=0.9793, Normal Precision=0.9664, Attack Recall=0.6937, Attack Precision=0.7883

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
0.15       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931   <--
0.20       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.25       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.30       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.35       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.40       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.45       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.50       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.55       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.60       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.65       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.70       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.75       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.80       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9216, F1=0.7789, Normal Recall=0.9793, Normal Precision=0.9268, Attack Recall=0.6905, Attack Precision=0.8931

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
0.15       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358   <--
0.20       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.25       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.30       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.35       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.40       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.45       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.50       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.55       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.60       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.65       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.70       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.75       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.80       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8929, F1=0.7947, Normal Recall=0.9797, Normal Precision=0.8808, Attack Recall=0.6906, Attack Precision=0.9358

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
0.15       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583   <--
0.20       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.25       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.30       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.35       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.40       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.45       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.50       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.55       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.60       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.65       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.70       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.75       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.80       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8642, F1=0.8027, Normal Recall=0.9800, Normal Precision=0.8261, Attack Recall=0.6905, Attack Precision=0.9583

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
0.15       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714   <--
0.20       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.25       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.30       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.35       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.40       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.45       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.50       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.55       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.60       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.65       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.70       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.75       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.80       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8351, F1=0.8072, Normal Recall=0.9797, Normal Precision=0.7600, Attack Recall=0.6905, Attack Precision=0.9714

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
0.15       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883   <--
0.20       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.25       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.30       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.35       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.40       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.45       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.50       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.55       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.60       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.65       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.70       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.75       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
0.80       0.9507   0.7379   0.9793   0.9664   0.6937   0.7883  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9507, F1=0.7379, Normal Recall=0.9793, Normal Precision=0.9664, Attack Recall=0.6937, Attack Precision=0.7883

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
0.15       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931   <--
0.20       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.25       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.30       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.35       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.40       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.45       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.50       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.55       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.60       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.65       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.70       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.75       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
0.80       0.9216   0.7789   0.9793   0.9268   0.6905   0.8931  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9216, F1=0.7789, Normal Recall=0.9793, Normal Precision=0.9268, Attack Recall=0.6905, Attack Precision=0.8931

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
0.15       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358   <--
0.20       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.25       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.30       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.35       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.40       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.45       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.50       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.55       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.60       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.65       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.70       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.75       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
0.80       0.8929   0.7947   0.9797   0.8808   0.6906   0.9358  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8929, F1=0.7947, Normal Recall=0.9797, Normal Precision=0.8808, Attack Recall=0.6906, Attack Precision=0.9358

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
0.15       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583   <--
0.20       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.25       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.30       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.35       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.40       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.45       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.50       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.55       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.60       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.65       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.70       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.75       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
0.80       0.8642   0.8027   0.9800   0.8261   0.6905   0.9583  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8642, F1=0.8027, Normal Recall=0.9800, Normal Precision=0.8261, Attack Recall=0.6905, Attack Precision=0.9583

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
0.15       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714   <--
0.20       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.25       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.30       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.35       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.40       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.45       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.50       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.55       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.60       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.65       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.70       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.75       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
0.80       0.8351   0.8072   0.9797   0.7600   0.6905   0.9714  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8351, F1=0.8072, Normal Recall=0.9797, Normal Precision=0.7600, Attack Recall=0.6905, Attack Precision=0.9714

```

