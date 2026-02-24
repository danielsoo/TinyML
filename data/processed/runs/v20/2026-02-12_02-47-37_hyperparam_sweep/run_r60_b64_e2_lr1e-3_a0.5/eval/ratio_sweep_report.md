# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e2_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-13 02:07:11 |

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
| Original (TFLite) | 0.7741 | 0.7441 | 0.7154 | 0.6871 | 0.6572 | 0.6302 | 0.6014 | 0.5728 | 0.5444 | 0.5147 | 0.4865 |
| QAT+Prune only | 0.6797 | 0.7120 | 0.7435 | 0.7757 | 0.8077 | 0.8396 | 0.8716 | 0.9032 | 0.9343 | 0.9672 | 0.9990 |
| QAT+PTQ | 0.6801 | 0.7123 | 0.7438 | 0.7760 | 0.8079 | 0.8397 | 0.8719 | 0.9035 | 0.9344 | 0.9673 | 0.9991 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6801 | 0.7123 | 0.7438 | 0.7760 | 0.8079 | 0.8397 | 0.8719 | 0.9035 | 0.9344 | 0.9673 | 0.9991 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2761 | 0.4061 | 0.4826 | 0.5317 | 0.5681 | 0.5943 | 0.6146 | 0.6308 | 0.6434 | 0.6546 |
| QAT+Prune only | 0.0000 | 0.4096 | 0.6090 | 0.7277 | 0.8061 | 0.8616 | 0.9033 | 0.9353 | 0.9605 | 0.9821 | 0.9995 |
| QAT+PTQ | 0.0000 | 0.4099 | 0.6093 | 0.7279 | 0.8062 | 0.8617 | 0.9034 | 0.9354 | 0.9606 | 0.9822 | 0.9995 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4099 | 0.6093 | 0.7279 | 0.8062 | 0.8617 | 0.9034 | 0.9354 | 0.9606 | 0.9822 | 0.9995 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7741 | 0.7726 | 0.7726 | 0.7730 | 0.7710 | 0.7738 | 0.7737 | 0.7743 | 0.7760 | 0.7688 | 0.0000 |
| QAT+Prune only | 0.6797 | 0.6801 | 0.6796 | 0.6799 | 0.6802 | 0.6802 | 0.6805 | 0.6796 | 0.6753 | 0.6806 | 0.0000 |
| QAT+PTQ | 0.6801 | 0.6805 | 0.6800 | 0.6803 | 0.6804 | 0.6803 | 0.6810 | 0.6804 | 0.6758 | 0.6816 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6801 | 0.6805 | 0.6800 | 0.6803 | 0.6804 | 0.6803 | 0.6810 | 0.6804 | 0.6758 | 0.6816 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7741 | 0.0000 | 0.0000 | 0.0000 | 0.7741 | 1.0000 |
| 90 | 10 | 299,940 | 0.7441 | 0.1925 | 0.4879 | 0.2761 | 0.7726 | 0.9314 |
| 80 | 20 | 291,350 | 0.7154 | 0.3484 | 0.4865 | 0.4061 | 0.7726 | 0.8575 |
| 70 | 30 | 194,230 | 0.6871 | 0.4788 | 0.4865 | 0.4826 | 0.7730 | 0.7784 |
| 60 | 40 | 145,675 | 0.6572 | 0.5861 | 0.4865 | 0.5317 | 0.7710 | 0.6925 |
| 50 | 50 | 116,540 | 0.6302 | 0.6826 | 0.4865 | 0.5681 | 0.7738 | 0.6011 |
| 40 | 60 | 97,115 | 0.6014 | 0.7633 | 0.4865 | 0.5943 | 0.7737 | 0.5011 |
| 30 | 70 | 83,240 | 0.5728 | 0.8341 | 0.4865 | 0.6146 | 0.7743 | 0.3925 |
| 20 | 80 | 72,835 | 0.5444 | 0.8968 | 0.4865 | 0.6308 | 0.7760 | 0.2742 |
| 10 | 90 | 64,740 | 0.5147 | 0.9498 | 0.4865 | 0.6434 | 0.7688 | 0.1426 |
| 0 | 100 | 58,270 | 0.4865 | 1.0000 | 0.4865 | 0.6546 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6797 | 0.0000 | 0.0000 | 0.0000 | 0.6797 | 1.0000 |
| 90 | 10 | 299,940 | 0.7120 | 0.2576 | 0.9990 | 0.4096 | 0.6801 | 0.9998 |
| 80 | 20 | 291,350 | 0.7435 | 0.4380 | 0.9990 | 0.6090 | 0.6796 | 0.9996 |
| 70 | 30 | 194,230 | 0.7757 | 0.5722 | 0.9990 | 0.7277 | 0.6799 | 0.9994 |
| 60 | 40 | 145,675 | 0.8077 | 0.6756 | 0.9990 | 0.8061 | 0.6802 | 0.9990 |
| 50 | 50 | 116,540 | 0.8396 | 0.7575 | 0.9990 | 0.8616 | 0.6802 | 0.9985 |
| 40 | 60 | 97,115 | 0.8716 | 0.8242 | 0.9990 | 0.9033 | 0.6805 | 0.9978 |
| 30 | 70 | 83,240 | 0.9032 | 0.8792 | 0.9990 | 0.9353 | 0.6796 | 0.9966 |
| 20 | 80 | 72,835 | 0.9343 | 0.9248 | 0.9990 | 0.9605 | 0.6753 | 0.9941 |
| 10 | 90 | 64,740 | 0.9672 | 0.9657 | 0.9990 | 0.9821 | 0.6806 | 0.9870 |
| 0 | 100 | 58,270 | 0.9990 | 1.0000 | 0.9990 | 0.9995 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6801 | 0.0000 | 0.0000 | 0.0000 | 0.6801 | 1.0000 |
| 90 | 10 | 299,940 | 0.7123 | 0.2578 | 0.9991 | 0.4099 | 0.6805 | 0.9998 |
| 80 | 20 | 291,350 | 0.7438 | 0.4383 | 0.9991 | 0.6093 | 0.6800 | 0.9997 |
| 70 | 30 | 194,230 | 0.7760 | 0.5725 | 0.9991 | 0.7279 | 0.6803 | 0.9994 |
| 60 | 40 | 145,675 | 0.8079 | 0.6758 | 0.9991 | 0.8062 | 0.6804 | 0.9991 |
| 50 | 50 | 116,540 | 0.8397 | 0.7576 | 0.9991 | 0.8617 | 0.6803 | 0.9986 |
| 40 | 60 | 97,115 | 0.8719 | 0.8245 | 0.9991 | 0.9034 | 0.6810 | 0.9980 |
| 30 | 70 | 83,240 | 0.9035 | 0.8794 | 0.9991 | 0.9354 | 0.6804 | 0.9968 |
| 20 | 80 | 72,835 | 0.9344 | 0.9250 | 0.9991 | 0.9606 | 0.6758 | 0.9945 |
| 10 | 90 | 64,740 | 0.9673 | 0.9658 | 0.9991 | 0.9822 | 0.6816 | 0.9879 |
| 0 | 100 | 58,270 | 0.9991 | 1.0000 | 0.9991 | 0.9995 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6801 | 0.0000 | 0.0000 | 0.0000 | 0.6801 | 1.0000 |
| 90 | 10 | 299,940 | 0.7123 | 0.2578 | 0.9991 | 0.4099 | 0.6805 | 0.9998 |
| 80 | 20 | 291,350 | 0.7438 | 0.4383 | 0.9991 | 0.6093 | 0.6800 | 0.9997 |
| 70 | 30 | 194,230 | 0.7760 | 0.5725 | 0.9991 | 0.7279 | 0.6803 | 0.9994 |
| 60 | 40 | 145,675 | 0.8079 | 0.6758 | 0.9991 | 0.8062 | 0.6804 | 0.9991 |
| 50 | 50 | 116,540 | 0.8397 | 0.7576 | 0.9991 | 0.8617 | 0.6803 | 0.9986 |
| 40 | 60 | 97,115 | 0.8719 | 0.8245 | 0.9991 | 0.9034 | 0.6810 | 0.9980 |
| 30 | 70 | 83,240 | 0.9035 | 0.8794 | 0.9991 | 0.9354 | 0.6804 | 0.9968 |
| 20 | 80 | 72,835 | 0.9344 | 0.9250 | 0.9991 | 0.9606 | 0.6758 | 0.9945 |
| 10 | 90 | 64,740 | 0.9673 | 0.9658 | 0.9991 | 0.9822 | 0.6816 | 0.9879 |
| 0 | 100 | 58,270 | 0.9991 | 1.0000 | 0.9991 | 0.9995 | 0.0000 | 0.0000 |


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
0.15       0.7437   0.2740   0.7726   0.9309   0.4836   0.1911   <--
0.20       0.7437   0.2740   0.7726   0.9309   0.4836   0.1911  
0.25       0.7437   0.2740   0.7726   0.9309   0.4836   0.1911  
0.30       0.7437   0.2740   0.7726   0.9309   0.4836   0.1911  
0.35       0.7437   0.2740   0.7726   0.9309   0.4836   0.1911  
0.40       0.7437   0.2740   0.7726   0.9309   0.4836   0.1911  
0.45       0.7437   0.2740   0.7726   0.9309   0.4836   0.1911  
0.50       0.7437   0.2740   0.7726   0.9309   0.4836   0.1911  
0.55       0.7437   0.2740   0.7726   0.9309   0.4836   0.1911  
0.60       0.7437   0.2740   0.7726   0.9309   0.4836   0.1911  
0.65       0.7437   0.2740   0.7726   0.9309   0.4836   0.1911  
0.70       0.7437   0.2740   0.7726   0.9309   0.4836   0.1911  
0.75       0.7437   0.2740   0.7726   0.9309   0.4836   0.1911  
0.80       0.7437   0.2740   0.7726   0.9309   0.4836   0.1911  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7437, F1=0.2740, Normal Recall=0.7726, Normal Precision=0.9309, Attack Recall=0.4836, Attack Precision=0.1911

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
0.15       0.7156   0.4062   0.7728   0.8576   0.4865   0.3487   <--
0.20       0.7156   0.4062   0.7728   0.8576   0.4865   0.3487  
0.25       0.7156   0.4062   0.7728   0.8576   0.4865   0.3487  
0.30       0.7156   0.4062   0.7728   0.8576   0.4865   0.3487  
0.35       0.7156   0.4062   0.7728   0.8576   0.4865   0.3487  
0.40       0.7156   0.4062   0.7728   0.8576   0.4865   0.3487  
0.45       0.7156   0.4062   0.7728   0.8576   0.4865   0.3487  
0.50       0.7156   0.4062   0.7728   0.8576   0.4865   0.3487  
0.55       0.7156   0.4062   0.7728   0.8576   0.4865   0.3487  
0.60       0.7156   0.4062   0.7728   0.8576   0.4865   0.3487  
0.65       0.7156   0.4062   0.7728   0.8576   0.4865   0.3487  
0.70       0.7156   0.4062   0.7728   0.8576   0.4865   0.3487  
0.75       0.7156   0.4062   0.7728   0.8576   0.4865   0.3487  
0.80       0.7156   0.4062   0.7728   0.8576   0.4865   0.3487  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7156, F1=0.4062, Normal Recall=0.7728, Normal Precision=0.8576, Attack Recall=0.4865, Attack Precision=0.3487

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
0.15       0.6874   0.4828   0.7734   0.7785   0.4865   0.4792   <--
0.20       0.6874   0.4828   0.7734   0.7785   0.4865   0.4792  
0.25       0.6874   0.4828   0.7734   0.7785   0.4865   0.4792  
0.30       0.6874   0.4828   0.7734   0.7785   0.4865   0.4792  
0.35       0.6874   0.4828   0.7734   0.7785   0.4865   0.4792  
0.40       0.6874   0.4828   0.7734   0.7785   0.4865   0.4792  
0.45       0.6874   0.4828   0.7734   0.7785   0.4865   0.4792  
0.50       0.6874   0.4828   0.7734   0.7785   0.4865   0.4792  
0.55       0.6874   0.4828   0.7734   0.7785   0.4865   0.4792  
0.60       0.6874   0.4828   0.7734   0.7785   0.4865   0.4792  
0.65       0.6874   0.4828   0.7734   0.7785   0.4865   0.4792  
0.70       0.6874   0.4828   0.7734   0.7785   0.4865   0.4792  
0.75       0.6874   0.4828   0.7734   0.7785   0.4865   0.4792  
0.80       0.6874   0.4828   0.7734   0.7785   0.4865   0.4792  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6874, F1=0.4828, Normal Recall=0.7734, Normal Precision=0.7785, Attack Recall=0.4865, Attack Precision=0.4792

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
0.15       0.6590   0.5330   0.7740   0.6933   0.4865   0.5893   <--
0.20       0.6590   0.5330   0.7740   0.6933   0.4865   0.5893  
0.25       0.6590   0.5330   0.7740   0.6933   0.4865   0.5893  
0.30       0.6590   0.5330   0.7740   0.6933   0.4865   0.5893  
0.35       0.6590   0.5330   0.7740   0.6933   0.4865   0.5893  
0.40       0.6590   0.5330   0.7740   0.6933   0.4865   0.5893  
0.45       0.6590   0.5330   0.7740   0.6933   0.4865   0.5893  
0.50       0.6590   0.5330   0.7740   0.6933   0.4865   0.5893  
0.55       0.6590   0.5330   0.7740   0.6933   0.4865   0.5893  
0.60       0.6590   0.5330   0.7740   0.6933   0.4865   0.5893  
0.65       0.6590   0.5330   0.7740   0.6933   0.4865   0.5893  
0.70       0.6590   0.5330   0.7740   0.6933   0.4865   0.5893  
0.75       0.6590   0.5330   0.7740   0.6933   0.4865   0.5893  
0.80       0.6590   0.5330   0.7740   0.6933   0.4865   0.5893  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6590, F1=0.5330, Normal Recall=0.7740, Normal Precision=0.6933, Attack Recall=0.4865, Attack Precision=0.5893

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
0.15       0.6304   0.5682   0.7742   0.6012   0.4865   0.6830   <--
0.20       0.6304   0.5682   0.7742   0.6012   0.4865   0.6830  
0.25       0.6304   0.5682   0.7742   0.6012   0.4865   0.6830  
0.30       0.6304   0.5682   0.7742   0.6012   0.4865   0.6830  
0.35       0.6304   0.5682   0.7742   0.6012   0.4865   0.6830  
0.40       0.6304   0.5682   0.7742   0.6012   0.4865   0.6830  
0.45       0.6304   0.5682   0.7742   0.6012   0.4865   0.6830  
0.50       0.6304   0.5682   0.7742   0.6012   0.4865   0.6830  
0.55       0.6304   0.5682   0.7742   0.6012   0.4865   0.6830  
0.60       0.6304   0.5682   0.7742   0.6012   0.4865   0.6830  
0.65       0.6304   0.5682   0.7742   0.6012   0.4865   0.6830  
0.70       0.6304   0.5682   0.7742   0.6012   0.4865   0.6830  
0.75       0.6304   0.5682   0.7742   0.6012   0.4865   0.6830  
0.80       0.6304   0.5682   0.7742   0.6012   0.4865   0.6830  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6304, F1=0.5682, Normal Recall=0.7742, Normal Precision=0.6012, Attack Recall=0.4865, Attack Precision=0.6830

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
0.15       0.7120   0.4096   0.6801   0.9998   0.9989   0.2576   <--
0.20       0.7120   0.4096   0.6801   0.9998   0.9989   0.2576  
0.25       0.7120   0.4096   0.6801   0.9998   0.9989   0.2576  
0.30       0.7120   0.4096   0.6801   0.9998   0.9989   0.2576  
0.35       0.7120   0.4096   0.6801   0.9998   0.9989   0.2576  
0.40       0.7120   0.4096   0.6801   0.9998   0.9989   0.2576  
0.45       0.7120   0.4096   0.6801   0.9998   0.9989   0.2576  
0.50       0.7120   0.4096   0.6801   0.9998   0.9989   0.2576  
0.55       0.7120   0.4096   0.6801   0.9998   0.9989   0.2576  
0.60       0.7120   0.4096   0.6801   0.9998   0.9989   0.2576  
0.65       0.7120   0.4096   0.6801   0.9998   0.9989   0.2576  
0.70       0.7120   0.4096   0.6801   0.9998   0.9989   0.2576  
0.75       0.7120   0.4096   0.6801   0.9998   0.9989   0.2576  
0.80       0.7120   0.4096   0.6801   0.9998   0.9989   0.2576  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7120, F1=0.4096, Normal Recall=0.6801, Normal Precision=0.9998, Attack Recall=0.9989, Attack Precision=0.2576

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
0.15       0.7443   0.6098   0.6806   0.9996   0.9990   0.4388   <--
0.20       0.7443   0.6098   0.6806   0.9996   0.9990   0.4388  
0.25       0.7443   0.6098   0.6806   0.9996   0.9990   0.4388  
0.30       0.7443   0.6098   0.6806   0.9996   0.9990   0.4388  
0.35       0.7443   0.6098   0.6806   0.9996   0.9990   0.4388  
0.40       0.7443   0.6098   0.6806   0.9996   0.9990   0.4388  
0.45       0.7443   0.6098   0.6806   0.9996   0.9990   0.4388  
0.50       0.7443   0.6098   0.6806   0.9996   0.9990   0.4388  
0.55       0.7443   0.6098   0.6806   0.9996   0.9990   0.4388  
0.60       0.7443   0.6098   0.6806   0.9996   0.9990   0.4388  
0.65       0.7443   0.6098   0.6806   0.9996   0.9990   0.4388  
0.70       0.7443   0.6098   0.6806   0.9996   0.9990   0.4388  
0.75       0.7443   0.6098   0.6806   0.9996   0.9990   0.4388  
0.80       0.7443   0.6098   0.6806   0.9996   0.9990   0.4388  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7443, F1=0.6098, Normal Recall=0.6806, Normal Precision=0.9996, Attack Recall=0.9990, Attack Precision=0.4388

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
0.15       0.7755   0.7275   0.6798   0.9994   0.9990   0.5721   <--
0.20       0.7755   0.7275   0.6798   0.9994   0.9990   0.5721  
0.25       0.7755   0.7275   0.6798   0.9994   0.9990   0.5721  
0.30       0.7755   0.7275   0.6798   0.9994   0.9990   0.5721  
0.35       0.7755   0.7275   0.6798   0.9994   0.9990   0.5721  
0.40       0.7755   0.7275   0.6798   0.9994   0.9990   0.5721  
0.45       0.7755   0.7275   0.6798   0.9994   0.9990   0.5721  
0.50       0.7755   0.7275   0.6798   0.9994   0.9990   0.5721  
0.55       0.7755   0.7275   0.6798   0.9994   0.9990   0.5721  
0.60       0.7755   0.7275   0.6798   0.9994   0.9990   0.5721  
0.65       0.7755   0.7275   0.6798   0.9994   0.9990   0.5721  
0.70       0.7755   0.7275   0.6798   0.9994   0.9990   0.5721  
0.75       0.7755   0.7275   0.6798   0.9994   0.9990   0.5721  
0.80       0.7755   0.7275   0.6798   0.9994   0.9990   0.5721  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7755, F1=0.7275, Normal Recall=0.6798, Normal Precision=0.9994, Attack Recall=0.9990, Attack Precision=0.5721

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
0.15       0.8080   0.8063   0.6806   0.9990   0.9990   0.6759   <--
0.20       0.8080   0.8063   0.6806   0.9990   0.9990   0.6759  
0.25       0.8080   0.8063   0.6806   0.9990   0.9990   0.6759  
0.30       0.8080   0.8063   0.6806   0.9990   0.9990   0.6759  
0.35       0.8080   0.8063   0.6806   0.9990   0.9990   0.6759  
0.40       0.8080   0.8063   0.6806   0.9990   0.9990   0.6759  
0.45       0.8080   0.8063   0.6806   0.9990   0.9990   0.6759  
0.50       0.8080   0.8063   0.6806   0.9990   0.9990   0.6759  
0.55       0.8080   0.8063   0.6806   0.9990   0.9990   0.6759  
0.60       0.8080   0.8063   0.6806   0.9990   0.9990   0.6759  
0.65       0.8080   0.8063   0.6806   0.9990   0.9990   0.6759  
0.70       0.8080   0.8063   0.6806   0.9990   0.9990   0.6759  
0.75       0.8080   0.8063   0.6806   0.9990   0.9990   0.6759  
0.80       0.8080   0.8063   0.6806   0.9990   0.9990   0.6759  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8080, F1=0.8063, Normal Recall=0.6806, Normal Precision=0.9990, Attack Recall=0.9990, Attack Precision=0.6759

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
0.15       0.8388   0.8611   0.6786   0.9985   0.9990   0.7566   <--
0.20       0.8388   0.8611   0.6786   0.9985   0.9990   0.7566  
0.25       0.8388   0.8611   0.6786   0.9985   0.9990   0.7566  
0.30       0.8388   0.8611   0.6786   0.9985   0.9990   0.7566  
0.35       0.8388   0.8611   0.6786   0.9985   0.9990   0.7566  
0.40       0.8388   0.8611   0.6786   0.9985   0.9990   0.7566  
0.45       0.8388   0.8611   0.6786   0.9985   0.9990   0.7566  
0.50       0.8388   0.8611   0.6786   0.9985   0.9990   0.7566  
0.55       0.8388   0.8611   0.6786   0.9985   0.9990   0.7566  
0.60       0.8388   0.8611   0.6786   0.9985   0.9990   0.7566  
0.65       0.8388   0.8611   0.6786   0.9985   0.9990   0.7566  
0.70       0.8388   0.8611   0.6786   0.9985   0.9990   0.7566  
0.75       0.8388   0.8611   0.6786   0.9985   0.9990   0.7566  
0.80       0.8388   0.8611   0.6786   0.9985   0.9990   0.7566  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8388, F1=0.8611, Normal Recall=0.6786, Normal Precision=0.9985, Attack Recall=0.9990, Attack Precision=0.7566

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
0.15       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578   <--
0.20       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.25       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.30       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.35       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.40       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.45       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.50       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.55       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.60       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.65       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.70       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.75       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.80       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7123, F1=0.4099, Normal Recall=0.6805, Normal Precision=0.9998, Attack Recall=0.9990, Attack Precision=0.2578

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
0.15       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391   <--
0.20       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.25       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.30       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.35       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.40       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.45       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.50       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.55       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.60       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.65       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.70       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.75       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.80       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7446, F1=0.6101, Normal Recall=0.6809, Normal Precision=0.9997, Attack Recall=0.9991, Attack Precision=0.4391

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
0.15       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725   <--
0.20       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.25       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.30       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.35       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.40       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.45       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.50       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.55       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.60       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.65       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.70       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.75       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.80       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7759, F1=0.7279, Normal Recall=0.6802, Normal Precision=0.9994, Attack Recall=0.9991, Attack Precision=0.5725

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
0.15       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761   <--
0.20       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.25       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.30       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.35       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.40       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.45       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.50       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.55       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.60       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.65       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.70       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.75       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.80       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8082, F1=0.8065, Normal Recall=0.6810, Normal Precision=0.9991, Attack Recall=0.9991, Attack Precision=0.6761

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
0.15       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568   <--
0.20       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.25       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.30       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.35       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.40       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.45       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.50       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.55       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.60       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.65       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.70       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.75       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.80       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8390, F1=0.8612, Normal Recall=0.6790, Normal Precision=0.9986, Attack Recall=0.9991, Attack Precision=0.7568

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
0.15       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578   <--
0.20       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.25       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.30       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.35       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.40       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.45       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.50       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.55       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.60       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.65       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.70       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.75       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
0.80       0.7123   0.4099   0.6805   0.9998   0.9990   0.2578  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7123, F1=0.4099, Normal Recall=0.6805, Normal Precision=0.9998, Attack Recall=0.9990, Attack Precision=0.2578

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
0.15       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391   <--
0.20       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.25       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.30       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.35       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.40       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.45       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.50       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.55       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.60       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.65       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.70       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.75       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
0.80       0.7446   0.6101   0.6809   0.9997   0.9991   0.4391  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7446, F1=0.6101, Normal Recall=0.6809, Normal Precision=0.9997, Attack Recall=0.9991, Attack Precision=0.4391

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
0.15       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725   <--
0.20       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.25       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.30       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.35       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.40       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.45       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.50       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.55       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.60       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.65       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.70       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.75       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
0.80       0.7759   0.7279   0.6802   0.9994   0.9991   0.5725  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7759, F1=0.7279, Normal Recall=0.6802, Normal Precision=0.9994, Attack Recall=0.9991, Attack Precision=0.5725

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
0.15       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761   <--
0.20       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.25       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.30       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.35       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.40       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.45       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.50       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.55       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.60       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.65       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.70       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.75       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
0.80       0.8082   0.8065   0.6810   0.9991   0.9991   0.6761  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8082, F1=0.8065, Normal Recall=0.6810, Normal Precision=0.9991, Attack Recall=0.9991, Attack Precision=0.6761

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
0.15       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568   <--
0.20       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.25       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.30       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.35       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.40       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.45       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.50       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.55       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.60       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.65       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.70       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.75       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
0.80       0.8390   0.8612   0.6790   0.9986   0.9991   0.7568  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8390, F1=0.8612, Normal Recall=0.6790, Normal Precision=0.9986, Attack Recall=0.9991, Attack Precision=0.7568

```

