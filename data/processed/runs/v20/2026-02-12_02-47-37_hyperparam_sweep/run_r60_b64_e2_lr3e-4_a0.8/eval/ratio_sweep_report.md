# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e2_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-12 19:31:43 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7570 | 0.7230 | 0.6890 | 0.6556 | 0.6207 | 0.5885 | 0.5539 | 0.5216 | 0.4858 | 0.4521 | 0.4177 |
| QAT+Prune only | 0.9545 | 0.9390 | 0.9232 | 0.9081 | 0.8919 | 0.8758 | 0.8608 | 0.8449 | 0.8289 | 0.8134 | 0.7979 |
| QAT+PTQ | 0.9527 | 0.9378 | 0.9224 | 0.9076 | 0.8917 | 0.8763 | 0.8614 | 0.8460 | 0.8304 | 0.8151 | 0.8000 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9527 | 0.9378 | 0.9224 | 0.9076 | 0.8917 | 0.8763 | 0.8614 | 0.8460 | 0.8304 | 0.8151 | 0.8000 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2302 | 0.3495 | 0.4212 | 0.4684 | 0.5037 | 0.5291 | 0.5500 | 0.5652 | 0.5785 | 0.5893 |
| QAT+Prune only | 0.0000 | 0.7238 | 0.8059 | 0.8389 | 0.8552 | 0.8653 | 0.8731 | 0.8781 | 0.8818 | 0.8850 | 0.8876 |
| QAT+PTQ | 0.0000 | 0.7202 | 0.8048 | 0.8386 | 0.8553 | 0.8661 | 0.8738 | 0.8791 | 0.8830 | 0.8862 | 0.8889 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.7202 | 0.8048 | 0.8386 | 0.8553 | 0.8661 | 0.8738 | 0.8791 | 0.8830 | 0.8862 | 0.8889 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7570 | 0.7573 | 0.7569 | 0.7575 | 0.7560 | 0.7592 | 0.7582 | 0.7638 | 0.7579 | 0.7610 | 0.0000 |
| QAT+Prune only | 0.9545 | 0.9546 | 0.9545 | 0.9553 | 0.9546 | 0.9538 | 0.9551 | 0.9545 | 0.9528 | 0.9524 | 0.0000 |
| QAT+PTQ | 0.9527 | 0.9531 | 0.9530 | 0.9537 | 0.9529 | 0.9526 | 0.9534 | 0.9533 | 0.9517 | 0.9509 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9527 | 0.9531 | 0.9530 | 0.9537 | 0.9529 | 0.9526 | 0.9534 | 0.9533 | 0.9517 | 0.9509 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7570 | 0.0000 | 0.0000 | 0.0000 | 0.7570 | 1.0000 |
| 90 | 10 | 299,940 | 0.7230 | 0.1594 | 0.4142 | 0.2302 | 0.7573 | 0.9209 |
| 80 | 20 | 291,350 | 0.6890 | 0.3005 | 0.4177 | 0.3495 | 0.7569 | 0.8387 |
| 70 | 30 | 194,230 | 0.6556 | 0.4247 | 0.4177 | 0.4212 | 0.7575 | 0.7522 |
| 60 | 40 | 145,675 | 0.6207 | 0.5330 | 0.4177 | 0.4684 | 0.7560 | 0.6607 |
| 50 | 50 | 116,540 | 0.5885 | 0.6344 | 0.4177 | 0.5037 | 0.7592 | 0.5660 |
| 40 | 60 | 97,115 | 0.5539 | 0.7215 | 0.4177 | 0.5291 | 0.7582 | 0.4647 |
| 30 | 70 | 83,240 | 0.5216 | 0.8049 | 0.4177 | 0.5500 | 0.7638 | 0.3599 |
| 20 | 80 | 72,835 | 0.4858 | 0.8735 | 0.4177 | 0.5652 | 0.7579 | 0.2455 |
| 10 | 90 | 64,740 | 0.4521 | 0.9402 | 0.4178 | 0.5785 | 0.7610 | 0.1268 |
| 0 | 100 | 58,270 | 0.4177 | 1.0000 | 0.4177 | 0.5893 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9545 | 0.0000 | 0.0000 | 0.0000 | 0.9545 | 1.0000 |
| 90 | 10 | 299,940 | 0.9390 | 0.6616 | 0.7988 | 0.7238 | 0.9546 | 0.9771 |
| 80 | 20 | 291,350 | 0.9232 | 0.8141 | 0.7979 | 0.8059 | 0.9545 | 0.9497 |
| 70 | 30 | 194,230 | 0.9081 | 0.8844 | 0.7979 | 0.8389 | 0.9553 | 0.9169 |
| 60 | 40 | 145,675 | 0.8919 | 0.9214 | 0.7979 | 0.8552 | 0.9546 | 0.8763 |
| 50 | 50 | 116,540 | 0.8758 | 0.9452 | 0.7979 | 0.8653 | 0.9538 | 0.8252 |
| 40 | 60 | 97,115 | 0.8608 | 0.9638 | 0.7979 | 0.8731 | 0.9551 | 0.7591 |
| 30 | 70 | 83,240 | 0.8449 | 0.9762 | 0.7979 | 0.8781 | 0.9545 | 0.6693 |
| 20 | 80 | 72,835 | 0.8289 | 0.9854 | 0.7979 | 0.8818 | 0.9528 | 0.5410 |
| 10 | 90 | 64,740 | 0.8134 | 0.9934 | 0.7979 | 0.8850 | 0.9524 | 0.3437 |
| 0 | 100 | 58,270 | 0.7979 | 1.0000 | 0.7979 | 0.8876 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9527 | 0.0000 | 0.0000 | 0.0000 | 0.9527 | 1.0000 |
| 90 | 10 | 299,940 | 0.9378 | 0.6546 | 0.8006 | 0.7202 | 0.9531 | 0.9773 |
| 80 | 20 | 291,350 | 0.9224 | 0.8096 | 0.8000 | 0.8048 | 0.9530 | 0.9502 |
| 70 | 30 | 194,230 | 0.9076 | 0.8811 | 0.8000 | 0.8386 | 0.9537 | 0.9175 |
| 60 | 40 | 145,675 | 0.8917 | 0.9188 | 0.8000 | 0.8553 | 0.9529 | 0.8773 |
| 50 | 50 | 116,540 | 0.8763 | 0.9441 | 0.8000 | 0.8661 | 0.9526 | 0.8265 |
| 40 | 60 | 97,115 | 0.8614 | 0.9626 | 0.8000 | 0.8738 | 0.9534 | 0.7607 |
| 30 | 70 | 83,240 | 0.8460 | 0.9756 | 0.8000 | 0.8791 | 0.9533 | 0.6714 |
| 20 | 80 | 72,835 | 0.8304 | 0.9851 | 0.8000 | 0.8830 | 0.9517 | 0.5434 |
| 10 | 90 | 64,740 | 0.8151 | 0.9932 | 0.8000 | 0.8862 | 0.9509 | 0.3457 |
| 0 | 100 | 58,270 | 0.8000 | 1.0000 | 0.8000 | 0.8889 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9527 | 0.0000 | 0.0000 | 0.0000 | 0.9527 | 1.0000 |
| 90 | 10 | 299,940 | 0.9378 | 0.6546 | 0.8006 | 0.7202 | 0.9531 | 0.9773 |
| 80 | 20 | 291,350 | 0.9224 | 0.8096 | 0.8000 | 0.8048 | 0.9530 | 0.9502 |
| 70 | 30 | 194,230 | 0.9076 | 0.8811 | 0.8000 | 0.8386 | 0.9537 | 0.9175 |
| 60 | 40 | 145,675 | 0.8917 | 0.9188 | 0.8000 | 0.8553 | 0.9529 | 0.8773 |
| 50 | 50 | 116,540 | 0.8763 | 0.9441 | 0.8000 | 0.8661 | 0.9526 | 0.8265 |
| 40 | 60 | 97,115 | 0.8614 | 0.9626 | 0.8000 | 0.8738 | 0.9534 | 0.7607 |
| 30 | 70 | 83,240 | 0.8460 | 0.9756 | 0.8000 | 0.8791 | 0.9533 | 0.6714 |
| 20 | 80 | 72,835 | 0.8304 | 0.9851 | 0.8000 | 0.8830 | 0.9517 | 0.5434 |
| 10 | 90 | 64,740 | 0.8151 | 0.9932 | 0.8000 | 0.8862 | 0.9509 | 0.3457 |
| 0 | 100 | 58,270 | 0.8000 | 1.0000 | 0.8000 | 0.8889 | 0.0000 | 0.0000 |


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
0.15       0.7232   0.2312   0.7573   0.9211   0.4162   0.1601   <--
0.20       0.7232   0.2312   0.7573   0.9211   0.4162   0.1601  
0.25       0.7232   0.2312   0.7573   0.9211   0.4162   0.1601  
0.30       0.7232   0.2312   0.7573   0.9211   0.4162   0.1601  
0.35       0.7232   0.2312   0.7573   0.9211   0.4162   0.1601  
0.40       0.7232   0.2312   0.7573   0.9211   0.4162   0.1601  
0.45       0.7232   0.2312   0.7573   0.9211   0.4162   0.1601  
0.50       0.7232   0.2312   0.7573   0.9211   0.4162   0.1601  
0.55       0.7232   0.2312   0.7573   0.9211   0.4162   0.1601  
0.60       0.7232   0.2312   0.7573   0.9211   0.4162   0.1601  
0.65       0.7232   0.2312   0.7573   0.9211   0.4162   0.1601  
0.70       0.7232   0.2312   0.7573   0.9211   0.4162   0.1601  
0.75       0.7232   0.2312   0.7573   0.9211   0.4162   0.1601  
0.80       0.7232   0.2312   0.7573   0.9211   0.4162   0.1601  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7232, F1=0.2312, Normal Recall=0.7573, Normal Precision=0.9211, Attack Recall=0.4162, Attack Precision=0.1601

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
0.15       0.6894   0.3498   0.7573   0.8388   0.4177   0.3008   <--
0.20       0.6894   0.3498   0.7573   0.8388   0.4177   0.3008  
0.25       0.6894   0.3498   0.7573   0.8388   0.4177   0.3008  
0.30       0.6894   0.3498   0.7573   0.8388   0.4177   0.3008  
0.35       0.6894   0.3498   0.7573   0.8388   0.4177   0.3008  
0.40       0.6894   0.3498   0.7573   0.8388   0.4177   0.3008  
0.45       0.6894   0.3498   0.7573   0.8388   0.4177   0.3008  
0.50       0.6894   0.3498   0.7573   0.8388   0.4177   0.3008  
0.55       0.6894   0.3498   0.7573   0.8388   0.4177   0.3008  
0.60       0.6894   0.3498   0.7573   0.8388   0.4177   0.3008  
0.65       0.6894   0.3498   0.7573   0.8388   0.4177   0.3008  
0.70       0.6894   0.3498   0.7573   0.8388   0.4177   0.3008  
0.75       0.6894   0.3498   0.7573   0.8388   0.4177   0.3008  
0.80       0.6894   0.3498   0.7573   0.8388   0.4177   0.3008  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6894, F1=0.3498, Normal Recall=0.7573, Normal Precision=0.8388, Attack Recall=0.4177, Attack Precision=0.3008

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
0.15       0.6552   0.4209   0.7570   0.7521   0.4177   0.4242   <--
0.20       0.6552   0.4209   0.7570   0.7521   0.4177   0.4242  
0.25       0.6552   0.4209   0.7570   0.7521   0.4177   0.4242  
0.30       0.6552   0.4209   0.7570   0.7521   0.4177   0.4242  
0.35       0.6552   0.4209   0.7570   0.7521   0.4177   0.4242  
0.40       0.6552   0.4209   0.7570   0.7521   0.4177   0.4242  
0.45       0.6552   0.4209   0.7570   0.7521   0.4177   0.4242  
0.50       0.6552   0.4209   0.7570   0.7521   0.4177   0.4242  
0.55       0.6552   0.4209   0.7570   0.7521   0.4177   0.4242  
0.60       0.6552   0.4209   0.7570   0.7521   0.4177   0.4242  
0.65       0.6552   0.4209   0.7570   0.7521   0.4177   0.4242  
0.70       0.6552   0.4209   0.7570   0.7521   0.4177   0.4242  
0.75       0.6552   0.4209   0.7570   0.7521   0.4177   0.4242  
0.80       0.6552   0.4209   0.7570   0.7521   0.4177   0.4242  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6552, F1=0.4209, Normal Recall=0.7570, Normal Precision=0.7521, Attack Recall=0.4177, Attack Precision=0.4242

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
0.15       0.6215   0.4689   0.7573   0.6611   0.4177   0.5343   <--
0.20       0.6215   0.4689   0.7573   0.6611   0.4177   0.5343  
0.25       0.6215   0.4689   0.7573   0.6611   0.4177   0.5343  
0.30       0.6215   0.4689   0.7573   0.6611   0.4177   0.5343  
0.35       0.6215   0.4689   0.7573   0.6611   0.4177   0.5343  
0.40       0.6215   0.4689   0.7573   0.6611   0.4177   0.5343  
0.45       0.6215   0.4689   0.7573   0.6611   0.4177   0.5343  
0.50       0.6215   0.4689   0.7573   0.6611   0.4177   0.5343  
0.55       0.6215   0.4689   0.7573   0.6611   0.4177   0.5343  
0.60       0.6215   0.4689   0.7573   0.6611   0.4177   0.5343  
0.65       0.6215   0.4689   0.7573   0.6611   0.4177   0.5343  
0.70       0.6215   0.4689   0.7573   0.6611   0.4177   0.5343  
0.75       0.6215   0.4689   0.7573   0.6611   0.4177   0.5343  
0.80       0.6215   0.4689   0.7573   0.6611   0.4177   0.5343  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6215, F1=0.4689, Normal Recall=0.7573, Normal Precision=0.6611, Attack Recall=0.4177, Attack Precision=0.5343

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
0.15       0.5880   0.5034   0.7582   0.5656   0.4177   0.6334   <--
0.20       0.5880   0.5034   0.7582   0.5656   0.4177   0.6334  
0.25       0.5880   0.5034   0.7582   0.5656   0.4177   0.6334  
0.30       0.5880   0.5034   0.7582   0.5656   0.4177   0.6334  
0.35       0.5880   0.5034   0.7582   0.5656   0.4177   0.6334  
0.40       0.5880   0.5034   0.7582   0.5656   0.4177   0.6334  
0.45       0.5880   0.5034   0.7582   0.5656   0.4177   0.6334  
0.50       0.5880   0.5034   0.7582   0.5656   0.4177   0.6334  
0.55       0.5880   0.5034   0.7582   0.5656   0.4177   0.6334  
0.60       0.5880   0.5034   0.7582   0.5656   0.4177   0.6334  
0.65       0.5880   0.5034   0.7582   0.5656   0.4177   0.6334  
0.70       0.5880   0.5034   0.7582   0.5656   0.4177   0.6334  
0.75       0.5880   0.5034   0.7582   0.5656   0.4177   0.6334  
0.80       0.5880   0.5034   0.7582   0.5656   0.4177   0.6334  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5880, F1=0.5034, Normal Recall=0.7582, Normal Precision=0.5656, Attack Recall=0.4177, Attack Precision=0.6334

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
0.15       0.9392   0.7249   0.9546   0.9773   0.8008   0.6621   <--
0.20       0.9392   0.7249   0.9546   0.9773   0.8008   0.6621  
0.25       0.9392   0.7249   0.9546   0.9773   0.8008   0.6621  
0.30       0.9392   0.7249   0.9546   0.9773   0.8008   0.6621  
0.35       0.9392   0.7249   0.9546   0.9773   0.8008   0.6621  
0.40       0.9392   0.7249   0.9546   0.9773   0.8008   0.6621  
0.45       0.9392   0.7249   0.9546   0.9773   0.8008   0.6621  
0.50       0.9392   0.7249   0.9546   0.9773   0.8008   0.6621  
0.55       0.9392   0.7249   0.9546   0.9773   0.8008   0.6621  
0.60       0.9392   0.7249   0.9546   0.9773   0.8008   0.6621  
0.65       0.9392   0.7249   0.9546   0.9773   0.8008   0.6621  
0.70       0.9392   0.7249   0.9546   0.9773   0.8008   0.6621  
0.75       0.9392   0.7249   0.9546   0.9773   0.8008   0.6621  
0.80       0.9392   0.7249   0.9546   0.9773   0.8008   0.6621  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9392, F1=0.7249, Normal Recall=0.9546, Normal Precision=0.9773, Attack Recall=0.8008, Attack Precision=0.6621

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
0.15       0.9234   0.8065   0.9548   0.9497   0.7979   0.8152   <--
0.20       0.9234   0.8065   0.9548   0.9497   0.7979   0.8152  
0.25       0.9234   0.8065   0.9548   0.9497   0.7979   0.8152  
0.30       0.9234   0.8065   0.9548   0.9497   0.7979   0.8152  
0.35       0.9234   0.8065   0.9548   0.9497   0.7979   0.8152  
0.40       0.9234   0.8065   0.9548   0.9497   0.7979   0.8152  
0.45       0.9234   0.8065   0.9548   0.9497   0.7979   0.8152  
0.50       0.9234   0.8065   0.9548   0.9497   0.7979   0.8152  
0.55       0.9234   0.8065   0.9548   0.9497   0.7979   0.8152  
0.60       0.9234   0.8065   0.9548   0.9497   0.7979   0.8152  
0.65       0.9234   0.8065   0.9548   0.9497   0.7979   0.8152  
0.70       0.9234   0.8065   0.9548   0.9497   0.7979   0.8152  
0.75       0.9234   0.8065   0.9548   0.9497   0.7979   0.8152  
0.80       0.9234   0.8065   0.9548   0.9497   0.7979   0.8152  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9234, F1=0.8065, Normal Recall=0.9548, Normal Precision=0.9497, Attack Recall=0.7979, Attack Precision=0.8152

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
0.15       0.9075   0.8381   0.9545   0.9168   0.7979   0.8826   <--
0.20       0.9075   0.8381   0.9545   0.9168   0.7979   0.8826  
0.25       0.9075   0.8381   0.9545   0.9168   0.7979   0.8826  
0.30       0.9075   0.8381   0.9545   0.9168   0.7979   0.8826  
0.35       0.9075   0.8381   0.9545   0.9168   0.7979   0.8826  
0.40       0.9075   0.8381   0.9545   0.9168   0.7979   0.8826  
0.45       0.9075   0.8381   0.9545   0.9168   0.7979   0.8826  
0.50       0.9075   0.8381   0.9545   0.9168   0.7979   0.8826  
0.55       0.9075   0.8381   0.9545   0.9168   0.7979   0.8826  
0.60       0.9075   0.8381   0.9545   0.9168   0.7979   0.8826  
0.65       0.9075   0.8381   0.9545   0.9168   0.7979   0.8826  
0.70       0.9075   0.8381   0.9545   0.9168   0.7979   0.8826  
0.75       0.9075   0.8381   0.9545   0.9168   0.7979   0.8826  
0.80       0.9075   0.8381   0.9545   0.9168   0.7979   0.8826  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9075, F1=0.8381, Normal Recall=0.9545, Normal Precision=0.9168, Attack Recall=0.7979, Attack Precision=0.8826

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
0.15       0.8921   0.8554   0.9549   0.8763   0.7979   0.9218   <--
0.20       0.8921   0.8554   0.9549   0.8763   0.7979   0.9218  
0.25       0.8921   0.8554   0.9549   0.8763   0.7979   0.9218  
0.30       0.8921   0.8554   0.9549   0.8763   0.7979   0.9218  
0.35       0.8921   0.8554   0.9549   0.8763   0.7979   0.9218  
0.40       0.8921   0.8554   0.9549   0.8763   0.7979   0.9218  
0.45       0.8921   0.8554   0.9549   0.8763   0.7979   0.9218  
0.50       0.8921   0.8554   0.9549   0.8763   0.7979   0.9218  
0.55       0.8921   0.8554   0.9549   0.8763   0.7979   0.9218  
0.60       0.8921   0.8554   0.9549   0.8763   0.7979   0.9218  
0.65       0.8921   0.8554   0.9549   0.8763   0.7979   0.9218  
0.70       0.8921   0.8554   0.9549   0.8763   0.7979   0.9218  
0.75       0.8921   0.8554   0.9549   0.8763   0.7979   0.9218  
0.80       0.8921   0.8554   0.9549   0.8763   0.7979   0.9218  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8921, F1=0.8554, Normal Recall=0.9549, Normal Precision=0.8763, Attack Recall=0.7979, Attack Precision=0.9218

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
0.15       0.8763   0.8658   0.9547   0.8253   0.7979   0.9463   <--
0.20       0.8763   0.8658   0.9547   0.8253   0.7979   0.9463  
0.25       0.8763   0.8658   0.9547   0.8253   0.7979   0.9463  
0.30       0.8763   0.8658   0.9547   0.8253   0.7979   0.9463  
0.35       0.8763   0.8658   0.9547   0.8253   0.7979   0.9463  
0.40       0.8763   0.8658   0.9547   0.8253   0.7979   0.9463  
0.45       0.8763   0.8658   0.9547   0.8253   0.7979   0.9463  
0.50       0.8763   0.8658   0.9547   0.8253   0.7979   0.9463  
0.55       0.8763   0.8658   0.9547   0.8253   0.7979   0.9463  
0.60       0.8763   0.8658   0.9547   0.8253   0.7979   0.9463  
0.65       0.8763   0.8658   0.9547   0.8253   0.7979   0.9463  
0.70       0.8763   0.8658   0.9547   0.8253   0.7979   0.9463  
0.75       0.8763   0.8658   0.9547   0.8253   0.7979   0.9463  
0.80       0.8763   0.8658   0.9547   0.8253   0.7979   0.9463  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8763, F1=0.8658, Normal Recall=0.9547, Normal Precision=0.8253, Attack Recall=0.7979, Attack Precision=0.9463

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
0.15       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551   <--
0.20       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.25       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.30       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.35       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.40       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.45       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.50       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.55       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.60       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.65       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.70       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.75       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.80       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9380, F1=0.7212, Normal Recall=0.9531, Normal Precision=0.9775, Attack Recall=0.8022, Attack Precision=0.6551

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
0.15       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102   <--
0.20       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.25       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.30       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.35       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.40       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.45       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.50       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.55       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.60       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.65       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.70       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.75       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.80       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9225, F1=0.8051, Normal Recall=0.9531, Normal Precision=0.9502, Attack Recall=0.8000, Attack Precision=0.8102

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
0.15       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792   <--
0.20       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.25       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.30       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.35       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.40       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.45       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.50       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.55       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.60       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.65       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.70       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.75       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.80       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9070, F1=0.8378, Normal Recall=0.9529, Normal Precision=0.9175, Attack Recall=0.8000, Attack Precision=0.8792

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
0.15       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192   <--
0.20       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.25       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.30       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.35       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.40       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.45       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.50       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.55       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.60       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.65       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.70       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.75       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.80       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8919, F1=0.8555, Normal Recall=0.9531, Normal Precision=0.8773, Attack Recall=0.8000, Attack Precision=0.9192

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
0.15       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444   <--
0.20       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.25       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.30       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.35       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.40       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.45       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.50       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.55       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.60       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.65       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.70       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.75       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.80       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8765, F1=0.8663, Normal Recall=0.9529, Normal Precision=0.8266, Attack Recall=0.8000, Attack Precision=0.9444

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
0.15       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551   <--
0.20       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.25       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.30       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.35       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.40       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.45       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.50       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.55       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.60       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.65       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.70       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.75       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
0.80       0.9380   0.7212   0.9531   0.9775   0.8022   0.6551  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9380, F1=0.7212, Normal Recall=0.9531, Normal Precision=0.9775, Attack Recall=0.8022, Attack Precision=0.6551

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
0.15       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102   <--
0.20       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.25       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.30       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.35       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.40       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.45       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.50       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.55       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.60       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.65       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.70       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.75       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
0.80       0.9225   0.8051   0.9531   0.9502   0.8000   0.8102  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9225, F1=0.8051, Normal Recall=0.9531, Normal Precision=0.9502, Attack Recall=0.8000, Attack Precision=0.8102

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
0.15       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792   <--
0.20       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.25       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.30       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.35       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.40       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.45       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.50       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.55       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.60       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.65       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.70       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.75       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
0.80       0.9070   0.8378   0.9529   0.9175   0.8000   0.8792  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9070, F1=0.8378, Normal Recall=0.9529, Normal Precision=0.9175, Attack Recall=0.8000, Attack Precision=0.8792

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
0.15       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192   <--
0.20       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.25       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.30       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.35       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.40       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.45       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.50       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.55       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.60       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.65       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.70       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.75       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
0.80       0.8919   0.8555   0.9531   0.8773   0.8000   0.9192  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8919, F1=0.8555, Normal Recall=0.9531, Normal Precision=0.8773, Attack Recall=0.8000, Attack Precision=0.9192

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
0.15       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444   <--
0.20       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.25       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.30       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.35       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.40       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.45       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.50       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.55       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.60       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.65       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.70       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.75       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
0.80       0.8765   0.8663   0.9529   0.8266   0.8000   0.9444  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8765, F1=0.8663, Normal Recall=0.9529, Normal Precision=0.8266, Attack Recall=0.8000, Attack Precision=0.9444

```

