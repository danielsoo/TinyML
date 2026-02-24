# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e2_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-14 17:22:09 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6251 | 0.6409 | 0.6571 | 0.6746 | 0.6911 | 0.7077 | 0.7244 | 0.7390 | 0.7545 | 0.7720 | 0.7882 |
| QAT+Prune only | 0.6741 | 0.7032 | 0.7333 | 0.7634 | 0.7943 | 0.8241 | 0.8561 | 0.8860 | 0.9180 | 0.9470 | 0.9789 |
| QAT+PTQ | 0.6739 | 0.7029 | 0.7330 | 0.7630 | 0.7940 | 0.8236 | 0.8555 | 0.8853 | 0.9172 | 0.9464 | 0.9781 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6739 | 0.7029 | 0.7330 | 0.7630 | 0.7940 | 0.8236 | 0.8555 | 0.8853 | 0.9172 | 0.9464 | 0.9781 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3051 | 0.4790 | 0.5924 | 0.6712 | 0.7295 | 0.7743 | 0.8087 | 0.8370 | 0.8616 | 0.8815 |
| QAT+Prune only | 0.0000 | 0.3975 | 0.5949 | 0.7129 | 0.7920 | 0.8477 | 0.8909 | 0.9232 | 0.9503 | 0.9708 | 0.9893 |
| QAT+PTQ | 0.0000 | 0.3971 | 0.5944 | 0.7123 | 0.7916 | 0.8472 | 0.8904 | 0.9227 | 0.9497 | 0.9704 | 0.9889 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3971 | 0.5944 | 0.7123 | 0.7916 | 0.8472 | 0.8904 | 0.9227 | 0.9497 | 0.9704 | 0.9889 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6251 | 0.6245 | 0.6243 | 0.6260 | 0.6264 | 0.6272 | 0.6286 | 0.6242 | 0.6197 | 0.6267 | 0.0000 |
| QAT+Prune only | 0.6741 | 0.6725 | 0.6719 | 0.6711 | 0.6713 | 0.6693 | 0.6719 | 0.6692 | 0.6746 | 0.6602 | 0.0000 |
| QAT+PTQ | 0.6739 | 0.6722 | 0.6717 | 0.6708 | 0.6712 | 0.6690 | 0.6715 | 0.6687 | 0.6734 | 0.6603 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6739 | 0.6722 | 0.6717 | 0.6708 | 0.6712 | 0.6690 | 0.6715 | 0.6687 | 0.6734 | 0.6603 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6251 | 0.0000 | 0.0000 | 0.0000 | 0.6251 | 1.0000 |
| 90 | 10 | 299,940 | 0.6409 | 0.1892 | 0.7885 | 0.3051 | 0.6245 | 0.9637 |
| 80 | 20 | 291,350 | 0.6571 | 0.3440 | 0.7882 | 0.4790 | 0.6243 | 0.9218 |
| 70 | 30 | 194,230 | 0.6746 | 0.4746 | 0.7882 | 0.5924 | 0.6260 | 0.8733 |
| 60 | 40 | 145,675 | 0.6911 | 0.5845 | 0.7882 | 0.6712 | 0.6264 | 0.8160 |
| 50 | 50 | 116,540 | 0.7077 | 0.6789 | 0.7882 | 0.7295 | 0.6272 | 0.7475 |
| 40 | 60 | 97,115 | 0.7244 | 0.7610 | 0.7882 | 0.7743 | 0.6286 | 0.6643 |
| 30 | 70 | 83,240 | 0.7390 | 0.8303 | 0.7882 | 0.8087 | 0.6242 | 0.5581 |
| 20 | 80 | 72,835 | 0.7545 | 0.8924 | 0.7882 | 0.8370 | 0.6197 | 0.4224 |
| 10 | 90 | 64,740 | 0.7720 | 0.9500 | 0.7882 | 0.8616 | 0.6267 | 0.2474 |
| 0 | 100 | 58,270 | 0.7882 | 1.0000 | 0.7882 | 0.8815 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6741 | 0.0000 | 0.0000 | 0.0000 | 0.6741 | 1.0000 |
| 90 | 10 | 299,940 | 0.7032 | 0.2494 | 0.9793 | 0.3975 | 0.6725 | 0.9966 |
| 80 | 20 | 291,350 | 0.7333 | 0.4272 | 0.9789 | 0.5949 | 0.6719 | 0.9922 |
| 70 | 30 | 194,230 | 0.7634 | 0.5605 | 0.9789 | 0.7129 | 0.6711 | 0.9867 |
| 60 | 40 | 145,675 | 0.7943 | 0.6650 | 0.9789 | 0.7920 | 0.6713 | 0.9795 |
| 50 | 50 | 116,540 | 0.8241 | 0.7475 | 0.9789 | 0.8477 | 0.6693 | 0.9694 |
| 40 | 60 | 97,115 | 0.8561 | 0.8174 | 0.9789 | 0.8909 | 0.6719 | 0.9550 |
| 30 | 70 | 83,240 | 0.8860 | 0.8735 | 0.9789 | 0.9232 | 0.6692 | 0.9314 |
| 20 | 80 | 72,835 | 0.9180 | 0.9233 | 0.9789 | 0.9503 | 0.6746 | 0.8888 |
| 10 | 90 | 64,740 | 0.9470 | 0.9629 | 0.9789 | 0.9708 | 0.6602 | 0.7767 |
| 0 | 100 | 58,270 | 0.9789 | 1.0000 | 0.9789 | 0.9893 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6739 | 0.0000 | 0.0000 | 0.0000 | 0.6739 | 1.0000 |
| 90 | 10 | 299,940 | 0.7029 | 0.2491 | 0.9785 | 0.3971 | 0.6722 | 0.9965 |
| 80 | 20 | 291,350 | 0.7330 | 0.4269 | 0.9781 | 0.5944 | 0.6717 | 0.9919 |
| 70 | 30 | 194,230 | 0.7630 | 0.5601 | 0.9781 | 0.7123 | 0.6708 | 0.9862 |
| 60 | 40 | 145,675 | 0.7940 | 0.6648 | 0.9781 | 0.7916 | 0.6712 | 0.9787 |
| 50 | 50 | 116,540 | 0.8236 | 0.7472 | 0.9781 | 0.8472 | 0.6690 | 0.9684 |
| 40 | 60 | 97,115 | 0.8555 | 0.8171 | 0.9781 | 0.8904 | 0.6715 | 0.9534 |
| 30 | 70 | 83,240 | 0.8853 | 0.8732 | 0.9781 | 0.9227 | 0.6687 | 0.9291 |
| 20 | 80 | 72,835 | 0.9172 | 0.9230 | 0.9781 | 0.9497 | 0.6734 | 0.8851 |
| 10 | 90 | 64,740 | 0.9464 | 0.9628 | 0.9781 | 0.9704 | 0.6603 | 0.7704 |
| 0 | 100 | 58,270 | 0.9781 | 1.0000 | 0.9781 | 0.9889 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6739 | 0.0000 | 0.0000 | 0.0000 | 0.6739 | 1.0000 |
| 90 | 10 | 299,940 | 0.7029 | 0.2491 | 0.9785 | 0.3971 | 0.6722 | 0.9965 |
| 80 | 20 | 291,350 | 0.7330 | 0.4269 | 0.9781 | 0.5944 | 0.6717 | 0.9919 |
| 70 | 30 | 194,230 | 0.7630 | 0.5601 | 0.9781 | 0.7123 | 0.6708 | 0.9862 |
| 60 | 40 | 145,675 | 0.7940 | 0.6648 | 0.9781 | 0.7916 | 0.6712 | 0.9787 |
| 50 | 50 | 116,540 | 0.8236 | 0.7472 | 0.9781 | 0.8472 | 0.6690 | 0.9684 |
| 40 | 60 | 97,115 | 0.8555 | 0.8171 | 0.9781 | 0.8904 | 0.6715 | 0.9534 |
| 30 | 70 | 83,240 | 0.8853 | 0.8732 | 0.9781 | 0.9227 | 0.6687 | 0.9291 |
| 20 | 80 | 72,835 | 0.9172 | 0.9230 | 0.9781 | 0.9497 | 0.6734 | 0.8851 |
| 10 | 90 | 64,740 | 0.9464 | 0.9628 | 0.9781 | 0.9704 | 0.6603 | 0.7704 |
| 0 | 100 | 58,270 | 0.9781 | 1.0000 | 0.9781 | 0.9889 | 0.0000 | 0.0000 |


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
0.15       0.6412   0.3059   0.6245   0.9641   0.7909   0.1896   <--
0.20       0.6412   0.3059   0.6245   0.9641   0.7909   0.1896  
0.25       0.6412   0.3059   0.6245   0.9641   0.7909   0.1896  
0.30       0.6412   0.3059   0.6245   0.9641   0.7909   0.1896  
0.35       0.6412   0.3059   0.6245   0.9641   0.7909   0.1896  
0.40       0.6412   0.3059   0.6245   0.9641   0.7909   0.1896  
0.45       0.6412   0.3059   0.6245   0.9641   0.7909   0.1896  
0.50       0.6412   0.3059   0.6245   0.9641   0.7909   0.1896  
0.55       0.6412   0.3059   0.6245   0.9641   0.7909   0.1896  
0.60       0.6412   0.3059   0.6245   0.9641   0.7909   0.1896  
0.65       0.6412   0.3059   0.6245   0.9641   0.7909   0.1896  
0.70       0.6412   0.3059   0.6245   0.9641   0.7909   0.1896  
0.75       0.6412   0.3059   0.6245   0.9641   0.7909   0.1896  
0.80       0.6412   0.3059   0.6245   0.9641   0.7909   0.1896  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6412, F1=0.3059, Normal Recall=0.6245, Normal Precision=0.9641, Attack Recall=0.7909, Attack Precision=0.1896

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
0.15       0.6573   0.4791   0.6246   0.9218   0.7882   0.3442   <--
0.20       0.6573   0.4791   0.6246   0.9218   0.7882   0.3442  
0.25       0.6573   0.4791   0.6246   0.9218   0.7882   0.3442  
0.30       0.6573   0.4791   0.6246   0.9218   0.7882   0.3442  
0.35       0.6573   0.4791   0.6246   0.9218   0.7882   0.3442  
0.40       0.6573   0.4791   0.6246   0.9218   0.7882   0.3442  
0.45       0.6573   0.4791   0.6246   0.9218   0.7882   0.3442  
0.50       0.6573   0.4791   0.6246   0.9218   0.7882   0.3442  
0.55       0.6573   0.4791   0.6246   0.9218   0.7882   0.3442  
0.60       0.6573   0.4791   0.6246   0.9218   0.7882   0.3442  
0.65       0.6573   0.4791   0.6246   0.9218   0.7882   0.3442  
0.70       0.6573   0.4791   0.6246   0.9218   0.7882   0.3442  
0.75       0.6573   0.4791   0.6246   0.9218   0.7882   0.3442  
0.80       0.6573   0.4791   0.6246   0.9218   0.7882   0.3442  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6573, F1=0.4791, Normal Recall=0.6246, Normal Precision=0.9218, Attack Recall=0.7882, Attack Precision=0.3442

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
0.15       0.6744   0.5923   0.6257   0.8733   0.7882   0.4743   <--
0.20       0.6744   0.5923   0.6257   0.8733   0.7882   0.4743  
0.25       0.6744   0.5923   0.6257   0.8733   0.7882   0.4743  
0.30       0.6744   0.5923   0.6257   0.8733   0.7882   0.4743  
0.35       0.6744   0.5923   0.6257   0.8733   0.7882   0.4743  
0.40       0.6744   0.5923   0.6257   0.8733   0.7882   0.4743  
0.45       0.6744   0.5923   0.6257   0.8733   0.7882   0.4743  
0.50       0.6744   0.5923   0.6257   0.8733   0.7882   0.4743  
0.55       0.6744   0.5923   0.6257   0.8733   0.7882   0.4743  
0.60       0.6744   0.5923   0.6257   0.8733   0.7882   0.4743  
0.65       0.6744   0.5923   0.6257   0.8733   0.7882   0.4743  
0.70       0.6744   0.5923   0.6257   0.8733   0.7882   0.4743  
0.75       0.6744   0.5923   0.6257   0.8733   0.7882   0.4743  
0.80       0.6744   0.5923   0.6257   0.8733   0.7882   0.4743  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6744, F1=0.5923, Normal Recall=0.6257, Normal Precision=0.8733, Attack Recall=0.7882, Attack Precision=0.4743

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
0.15       0.6908   0.6710   0.6259   0.8159   0.7882   0.5841   <--
0.20       0.6908   0.6710   0.6259   0.8159   0.7882   0.5841  
0.25       0.6908   0.6710   0.6259   0.8159   0.7882   0.5841  
0.30       0.6908   0.6710   0.6259   0.8159   0.7882   0.5841  
0.35       0.6908   0.6710   0.6259   0.8159   0.7882   0.5841  
0.40       0.6908   0.6710   0.6259   0.8159   0.7882   0.5841  
0.45       0.6908   0.6710   0.6259   0.8159   0.7882   0.5841  
0.50       0.6908   0.6710   0.6259   0.8159   0.7882   0.5841  
0.55       0.6908   0.6710   0.6259   0.8159   0.7882   0.5841  
0.60       0.6908   0.6710   0.6259   0.8159   0.7882   0.5841  
0.65       0.6908   0.6710   0.6259   0.8159   0.7882   0.5841  
0.70       0.6908   0.6710   0.6259   0.8159   0.7882   0.5841  
0.75       0.6908   0.6710   0.6259   0.8159   0.7882   0.5841  
0.80       0.6908   0.6710   0.6259   0.8159   0.7882   0.5841  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6908, F1=0.6710, Normal Recall=0.6259, Normal Precision=0.8159, Attack Recall=0.7882, Attack Precision=0.5841

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
0.15       0.7067   0.7288   0.6252   0.7469   0.7882   0.6777   <--
0.20       0.7067   0.7288   0.6252   0.7469   0.7882   0.6777  
0.25       0.7067   0.7288   0.6252   0.7469   0.7882   0.6777  
0.30       0.7067   0.7288   0.6252   0.7469   0.7882   0.6777  
0.35       0.7067   0.7288   0.6252   0.7469   0.7882   0.6777  
0.40       0.7067   0.7288   0.6252   0.7469   0.7882   0.6777  
0.45       0.7067   0.7288   0.6252   0.7469   0.7882   0.6777  
0.50       0.7067   0.7288   0.6252   0.7469   0.7882   0.6777  
0.55       0.7067   0.7288   0.6252   0.7469   0.7882   0.6777  
0.60       0.7067   0.7288   0.6252   0.7469   0.7882   0.6777  
0.65       0.7067   0.7288   0.6252   0.7469   0.7882   0.6777  
0.70       0.7067   0.7288   0.6252   0.7469   0.7882   0.6777  
0.75       0.7067   0.7288   0.6252   0.7469   0.7882   0.6777  
0.80       0.7067   0.7288   0.6252   0.7469   0.7882   0.6777  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7067, F1=0.7288, Normal Recall=0.6252, Normal Precision=0.7469, Attack Recall=0.7882, Attack Precision=0.6777

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
0.15       0.7031   0.3974   0.6725   0.9965   0.9789   0.2493   <--
0.20       0.7031   0.3974   0.6725   0.9965   0.9789   0.2493  
0.25       0.7031   0.3974   0.6725   0.9965   0.9789   0.2493  
0.30       0.7031   0.3974   0.6725   0.9965   0.9789   0.2493  
0.35       0.7031   0.3974   0.6725   0.9965   0.9789   0.2493  
0.40       0.7031   0.3974   0.6725   0.9965   0.9789   0.2493  
0.45       0.7031   0.3974   0.6725   0.9965   0.9789   0.2493  
0.50       0.7031   0.3974   0.6725   0.9965   0.9789   0.2493  
0.55       0.7031   0.3974   0.6725   0.9965   0.9789   0.2493  
0.60       0.7031   0.3974   0.6725   0.9965   0.9789   0.2493  
0.65       0.7031   0.3974   0.6725   0.9965   0.9789   0.2493  
0.70       0.7031   0.3974   0.6725   0.9965   0.9789   0.2493  
0.75       0.7031   0.3974   0.6725   0.9965   0.9789   0.2493  
0.80       0.7031   0.3974   0.6725   0.9965   0.9789   0.2493  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7031, F1=0.3974, Normal Recall=0.6725, Normal Precision=0.9965, Attack Recall=0.9789, Attack Precision=0.2493

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
0.15       0.7341   0.5956   0.6730   0.9922   0.9789   0.4280   <--
0.20       0.7341   0.5956   0.6730   0.9922   0.9789   0.4280  
0.25       0.7341   0.5956   0.6730   0.9922   0.9789   0.4280  
0.30       0.7341   0.5956   0.6730   0.9922   0.9789   0.4280  
0.35       0.7341   0.5956   0.6730   0.9922   0.9789   0.4280  
0.40       0.7341   0.5956   0.6730   0.9922   0.9789   0.4280  
0.45       0.7341   0.5956   0.6730   0.9922   0.9789   0.4280  
0.50       0.7341   0.5956   0.6730   0.9922   0.9789   0.4280  
0.55       0.7341   0.5956   0.6730   0.9922   0.9789   0.4280  
0.60       0.7341   0.5956   0.6730   0.9922   0.9789   0.4280  
0.65       0.7341   0.5956   0.6730   0.9922   0.9789   0.4280  
0.70       0.7341   0.5956   0.6730   0.9922   0.9789   0.4280  
0.75       0.7341   0.5956   0.6730   0.9922   0.9789   0.4280  
0.80       0.7341   0.5956   0.6730   0.9922   0.9789   0.4280  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7341, F1=0.5956, Normal Recall=0.6730, Normal Precision=0.9922, Attack Recall=0.9789, Attack Precision=0.4280

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
0.15       0.7653   0.7145   0.6738   0.9868   0.9789   0.5626   <--
0.20       0.7653   0.7145   0.6738   0.9868   0.9789   0.5626  
0.25       0.7653   0.7145   0.6738   0.9868   0.9789   0.5626  
0.30       0.7653   0.7145   0.6738   0.9868   0.9789   0.5626  
0.35       0.7653   0.7145   0.6738   0.9868   0.9789   0.5626  
0.40       0.7653   0.7145   0.6738   0.9868   0.9789   0.5626  
0.45       0.7653   0.7145   0.6738   0.9868   0.9789   0.5626  
0.50       0.7653   0.7145   0.6738   0.9868   0.9789   0.5626  
0.55       0.7653   0.7145   0.6738   0.9868   0.9789   0.5626  
0.60       0.7653   0.7145   0.6738   0.9868   0.9789   0.5626  
0.65       0.7653   0.7145   0.6738   0.9868   0.9789   0.5626  
0.70       0.7653   0.7145   0.6738   0.9868   0.9789   0.5626  
0.75       0.7653   0.7145   0.6738   0.9868   0.9789   0.5626  
0.80       0.7653   0.7145   0.6738   0.9868   0.9789   0.5626  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7653, F1=0.7145, Normal Recall=0.6738, Normal Precision=0.9868, Attack Recall=0.9789, Attack Precision=0.5626

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
0.15       0.7959   0.7932   0.6738   0.9795   0.9789   0.6668   <--
0.20       0.7959   0.7932   0.6738   0.9795   0.9789   0.6668  
0.25       0.7959   0.7932   0.6738   0.9795   0.9789   0.6668  
0.30       0.7959   0.7932   0.6738   0.9795   0.9789   0.6668  
0.35       0.7959   0.7932   0.6738   0.9795   0.9789   0.6668  
0.40       0.7959   0.7932   0.6738   0.9795   0.9789   0.6668  
0.45       0.7959   0.7932   0.6738   0.9795   0.9789   0.6668  
0.50       0.7959   0.7932   0.6738   0.9795   0.9789   0.6668  
0.55       0.7959   0.7932   0.6738   0.9795   0.9789   0.6668  
0.60       0.7959   0.7932   0.6738   0.9795   0.9789   0.6668  
0.65       0.7959   0.7932   0.6738   0.9795   0.9789   0.6668  
0.70       0.7959   0.7932   0.6738   0.9795   0.9789   0.6668  
0.75       0.7959   0.7932   0.6738   0.9795   0.9789   0.6668  
0.80       0.7959   0.7932   0.6738   0.9795   0.9789   0.6668  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7959, F1=0.7932, Normal Recall=0.6738, Normal Precision=0.9795, Attack Recall=0.9789, Attack Precision=0.6668

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
0.15       0.8253   0.8486   0.6718   0.9695   0.9789   0.7489   <--
0.20       0.8253   0.8486   0.6718   0.9695   0.9789   0.7489  
0.25       0.8253   0.8486   0.6718   0.9695   0.9789   0.7489  
0.30       0.8253   0.8486   0.6718   0.9695   0.9789   0.7489  
0.35       0.8253   0.8486   0.6718   0.9695   0.9789   0.7489  
0.40       0.8253   0.8486   0.6718   0.9695   0.9789   0.7489  
0.45       0.8253   0.8486   0.6718   0.9695   0.9789   0.7489  
0.50       0.8253   0.8486   0.6718   0.9695   0.9789   0.7489  
0.55       0.8253   0.8486   0.6718   0.9695   0.9789   0.7489  
0.60       0.8253   0.8486   0.6718   0.9695   0.9789   0.7489  
0.65       0.8253   0.8486   0.6718   0.9695   0.9789   0.7489  
0.70       0.8253   0.8486   0.6718   0.9695   0.9789   0.7489  
0.75       0.8253   0.8486   0.6718   0.9695   0.9789   0.7489  
0.80       0.8253   0.8486   0.6718   0.9695   0.9789   0.7489  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8253, F1=0.8486, Normal Recall=0.6718, Normal Precision=0.9695, Attack Recall=0.9789, Attack Precision=0.7489

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
0.15       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490   <--
0.20       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.25       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.30       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.35       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.40       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.45       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.50       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.55       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.60       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.65       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.70       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.75       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.80       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7028, F1=0.3970, Normal Recall=0.6722, Normal Precision=0.9964, Attack Recall=0.9781, Attack Precision=0.2490

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
0.15       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276   <--
0.20       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.25       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.30       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.35       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.40       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.45       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.50       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.55       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.60       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.65       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.70       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.75       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.80       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7338, F1=0.5951, Normal Recall=0.6727, Normal Precision=0.9919, Attack Recall=0.9781, Attack Precision=0.4276

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
0.15       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623   <--
0.20       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.25       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.30       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.35       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.40       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.45       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.50       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.55       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.60       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.65       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.70       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.75       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.80       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7650, F1=0.7141, Normal Recall=0.6736, Normal Precision=0.9863, Attack Recall=0.9781, Attack Precision=0.5623

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
0.15       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664   <--
0.20       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.25       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.30       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.35       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.40       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.45       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.50       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.55       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.60       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.65       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.70       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.75       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.80       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7954, F1=0.7927, Normal Recall=0.6736, Normal Precision=0.9788, Attack Recall=0.9781, Attack Precision=0.6664

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
0.15       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487   <--
0.20       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.25       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.30       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.35       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.40       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.45       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.50       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.55       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.60       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.65       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.70       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.75       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.80       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8249, F1=0.8481, Normal Recall=0.6716, Normal Precision=0.9685, Attack Recall=0.9781, Attack Precision=0.7487

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
0.15       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490   <--
0.20       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.25       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.30       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.35       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.40       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.45       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.50       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.55       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.60       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.65       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.70       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.75       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
0.80       0.7028   0.3970   0.6722   0.9964   0.9781   0.2490  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7028, F1=0.3970, Normal Recall=0.6722, Normal Precision=0.9964, Attack Recall=0.9781, Attack Precision=0.2490

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
0.15       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276   <--
0.20       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.25       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.30       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.35       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.40       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.45       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.50       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.55       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.60       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.65       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.70       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.75       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
0.80       0.7338   0.5951   0.6727   0.9919   0.9781   0.4276  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7338, F1=0.5951, Normal Recall=0.6727, Normal Precision=0.9919, Attack Recall=0.9781, Attack Precision=0.4276

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
0.15       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623   <--
0.20       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.25       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.30       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.35       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.40       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.45       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.50       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.55       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.60       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.65       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.70       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.75       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
0.80       0.7650   0.7141   0.6736   0.9863   0.9781   0.5623  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7650, F1=0.7141, Normal Recall=0.6736, Normal Precision=0.9863, Attack Recall=0.9781, Attack Precision=0.5623

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
0.15       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664   <--
0.20       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.25       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.30       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.35       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.40       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.45       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.50       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.55       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.60       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.65       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.70       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.75       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
0.80       0.7954   0.7927   0.6736   0.9788   0.9781   0.6664  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7954, F1=0.7927, Normal Recall=0.6736, Normal Precision=0.9788, Attack Recall=0.9781, Attack Precision=0.6664

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
0.15       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487   <--
0.20       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.25       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.30       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.35       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.40       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.45       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.50       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.55       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.60       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.65       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.70       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.75       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
0.80       0.8249   0.8481   0.6716   0.9685   0.9781   0.7487  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8249, F1=0.8481, Normal Recall=0.6716, Normal Precision=0.9685, Attack Recall=0.9781, Attack Precision=0.7487

```

