# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e2_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-19 22:42:29 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 2 |
| **Batch size** | 64 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6839 | 0.7117 | 0.7401 | 0.7711 | 0.7988 | 0.8281 | 0.8580 | 0.8863 | 0.9159 | 0.9452 | 0.9741 |
| QAT+Prune only | 0.8793 | 0.8878 | 0.8954 | 0.9038 | 0.9109 | 0.9177 | 0.9267 | 0.9339 | 0.9422 | 0.9490 | 0.9577 |
| QAT+PTQ | 0.8786 | 0.8871 | 0.8950 | 0.9035 | 0.9109 | 0.9178 | 0.9269 | 0.9342 | 0.9428 | 0.9497 | 0.9586 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8786 | 0.8871 | 0.8950 | 0.9035 | 0.9109 | 0.9178 | 0.9269 | 0.9342 | 0.9428 | 0.9497 | 0.9586 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.4031 | 0.5999 | 0.7186 | 0.7947 | 0.8500 | 0.8917 | 0.9231 | 0.9488 | 0.9697 | 0.9869 |
| QAT+Prune only | 0.0000 | 0.6309 | 0.7855 | 0.8566 | 0.8959 | 0.9209 | 0.9400 | 0.9530 | 0.9636 | 0.9713 | 0.9784 |
| QAT+PTQ | 0.0000 | 0.6296 | 0.7850 | 0.8564 | 0.8959 | 0.9210 | 0.9403 | 0.9532 | 0.9640 | 0.9717 | 0.9789 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.6296 | 0.7850 | 0.8564 | 0.8959 | 0.9210 | 0.9403 | 0.9532 | 0.9640 | 0.9717 | 0.9789 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6839 | 0.6826 | 0.6817 | 0.6841 | 0.6819 | 0.6822 | 0.6838 | 0.6816 | 0.6831 | 0.6858 | 0.0000 |
| QAT+Prune only | 0.8793 | 0.8799 | 0.8798 | 0.8807 | 0.8797 | 0.8777 | 0.8802 | 0.8782 | 0.8798 | 0.8706 | 0.0000 |
| QAT+PTQ | 0.8786 | 0.8791 | 0.8790 | 0.8799 | 0.8790 | 0.8770 | 0.8794 | 0.8771 | 0.8793 | 0.8692 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8786 | 0.8791 | 0.8790 | 0.8799 | 0.8790 | 0.8770 | 0.8794 | 0.8771 | 0.8793 | 0.8692 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6839 | 0.0000 | 0.0000 | 0.0000 | 0.6839 | 1.0000 |
| 90 | 10 | 299,940 | 0.7117 | 0.2542 | 0.9735 | 0.4031 | 0.6826 | 0.9957 |
| 80 | 20 | 291,350 | 0.7401 | 0.4334 | 0.9741 | 0.5999 | 0.6817 | 0.9906 |
| 70 | 30 | 194,230 | 0.7711 | 0.5693 | 0.9741 | 0.7186 | 0.6841 | 0.9840 |
| 60 | 40 | 145,675 | 0.7988 | 0.6712 | 0.9741 | 0.7947 | 0.6819 | 0.9753 |
| 50 | 50 | 116,540 | 0.8281 | 0.7540 | 0.9741 | 0.8500 | 0.6822 | 0.9634 |
| 40 | 60 | 97,115 | 0.8580 | 0.8221 | 0.9741 | 0.8917 | 0.6838 | 0.9462 |
| 30 | 70 | 83,240 | 0.8863 | 0.8771 | 0.9741 | 0.9231 | 0.6816 | 0.9185 |
| 20 | 80 | 72,835 | 0.9159 | 0.9248 | 0.9741 | 0.9488 | 0.6831 | 0.8682 |
| 10 | 90 | 64,740 | 0.9452 | 0.9654 | 0.9741 | 0.9697 | 0.6858 | 0.7461 |
| 0 | 100 | 58,270 | 0.9741 | 1.0000 | 0.9741 | 0.9869 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8793 | 0.0000 | 0.0000 | 0.0000 | 0.8793 | 1.0000 |
| 90 | 10 | 299,940 | 0.8878 | 0.4701 | 0.9589 | 0.6309 | 0.8799 | 0.9948 |
| 80 | 20 | 291,350 | 0.8954 | 0.6658 | 0.9577 | 0.7855 | 0.8798 | 0.9881 |
| 70 | 30 | 194,230 | 0.9038 | 0.7749 | 0.9577 | 0.8566 | 0.8807 | 0.9798 |
| 60 | 40 | 145,675 | 0.9109 | 0.8415 | 0.9577 | 0.8959 | 0.8797 | 0.9690 |
| 50 | 50 | 116,540 | 0.9177 | 0.8867 | 0.9577 | 0.9209 | 0.8777 | 0.9541 |
| 40 | 60 | 97,115 | 0.9267 | 0.9230 | 0.9577 | 0.9400 | 0.8802 | 0.9328 |
| 30 | 70 | 83,240 | 0.9339 | 0.9483 | 0.9577 | 0.9530 | 0.8782 | 0.8990 |
| 20 | 80 | 72,835 | 0.9422 | 0.9696 | 0.9577 | 0.9636 | 0.8798 | 0.8389 |
| 10 | 90 | 64,740 | 0.9490 | 0.9852 | 0.9577 | 0.9713 | 0.8706 | 0.6959 |
| 0 | 100 | 58,270 | 0.9577 | 1.0000 | 0.9577 | 0.9784 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8786 | 0.0000 | 0.0000 | 0.0000 | 0.8786 | 1.0000 |
| 90 | 10 | 299,940 | 0.8871 | 0.4685 | 0.9596 | 0.6296 | 0.8791 | 0.9949 |
| 80 | 20 | 291,350 | 0.8950 | 0.6646 | 0.9586 | 0.7850 | 0.8790 | 0.9884 |
| 70 | 30 | 194,230 | 0.9035 | 0.7738 | 0.9586 | 0.8564 | 0.8799 | 0.9802 |
| 60 | 40 | 145,675 | 0.9109 | 0.8408 | 0.9586 | 0.8959 | 0.8790 | 0.9696 |
| 50 | 50 | 116,540 | 0.9178 | 0.8862 | 0.9586 | 0.9210 | 0.8770 | 0.9549 |
| 40 | 60 | 97,115 | 0.9269 | 0.9226 | 0.9586 | 0.9403 | 0.8794 | 0.9341 |
| 30 | 70 | 83,240 | 0.9342 | 0.9479 | 0.9586 | 0.9532 | 0.8771 | 0.9008 |
| 20 | 80 | 72,835 | 0.9428 | 0.9695 | 0.9586 | 0.9640 | 0.8793 | 0.8416 |
| 10 | 90 | 64,740 | 0.9497 | 0.9851 | 0.9586 | 0.9717 | 0.8692 | 0.7000 |
| 0 | 100 | 58,270 | 0.9586 | 1.0000 | 0.9586 | 0.9789 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8786 | 0.0000 | 0.0000 | 0.0000 | 0.8786 | 1.0000 |
| 90 | 10 | 299,940 | 0.8871 | 0.4685 | 0.9596 | 0.6296 | 0.8791 | 0.9949 |
| 80 | 20 | 291,350 | 0.8950 | 0.6646 | 0.9586 | 0.7850 | 0.8790 | 0.9884 |
| 70 | 30 | 194,230 | 0.9035 | 0.7738 | 0.9586 | 0.8564 | 0.8799 | 0.9802 |
| 60 | 40 | 145,675 | 0.9109 | 0.8408 | 0.9586 | 0.8959 | 0.8790 | 0.9696 |
| 50 | 50 | 116,540 | 0.9178 | 0.8862 | 0.9586 | 0.9210 | 0.8770 | 0.9549 |
| 40 | 60 | 97,115 | 0.9269 | 0.9226 | 0.9586 | 0.9403 | 0.8794 | 0.9341 |
| 30 | 70 | 83,240 | 0.9342 | 0.9479 | 0.9586 | 0.9532 | 0.8771 | 0.9008 |
| 20 | 80 | 72,835 | 0.9428 | 0.9695 | 0.9586 | 0.9640 | 0.8793 | 0.8416 |
| 10 | 90 | 64,740 | 0.9497 | 0.9851 | 0.9586 | 0.9717 | 0.8692 | 0.7000 |
| 0 | 100 | 58,270 | 0.9586 | 1.0000 | 0.9586 | 0.9789 | 0.0000 | 0.0000 |


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
0.15       0.7117   0.4034   0.6826   0.9958   0.9744   0.2543   <--
0.20       0.7117   0.4034   0.6826   0.9958   0.9744   0.2543  
0.25       0.7117   0.4034   0.6826   0.9958   0.9744   0.2543  
0.30       0.7117   0.4034   0.6826   0.9958   0.9744   0.2543  
0.35       0.7117   0.4034   0.6826   0.9958   0.9744   0.2543  
0.40       0.7117   0.4034   0.6826   0.9958   0.9744   0.2543  
0.45       0.7117   0.4034   0.6826   0.9958   0.9744   0.2543  
0.50       0.7117   0.4034   0.6826   0.9958   0.9744   0.2543  
0.55       0.7117   0.4034   0.6826   0.9958   0.9744   0.2543  
0.60       0.7117   0.4034   0.6826   0.9958   0.9744   0.2543  
0.65       0.7117   0.4034   0.6826   0.9958   0.9744   0.2543  
0.70       0.7117   0.4034   0.6826   0.9958   0.9744   0.2543  
0.75       0.7117   0.4034   0.6826   0.9958   0.9744   0.2543  
0.80       0.7117   0.4034   0.6826   0.9958   0.9744   0.2543  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7117, F1=0.4034, Normal Recall=0.6826, Normal Precision=0.9958, Attack Recall=0.9744, Attack Precision=0.2543

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
0.15       0.7411   0.6008   0.6829   0.9906   0.9741   0.4344   <--
0.20       0.7411   0.6008   0.6829   0.9906   0.9741   0.4344  
0.25       0.7411   0.6008   0.6829   0.9906   0.9741   0.4344  
0.30       0.7411   0.6008   0.6829   0.9906   0.9741   0.4344  
0.35       0.7411   0.6008   0.6829   0.9906   0.9741   0.4344  
0.40       0.7411   0.6008   0.6829   0.9906   0.9741   0.4344  
0.45       0.7411   0.6008   0.6829   0.9906   0.9741   0.4344  
0.50       0.7411   0.6008   0.6829   0.9906   0.9741   0.4344  
0.55       0.7411   0.6008   0.6829   0.9906   0.9741   0.4344  
0.60       0.7411   0.6008   0.6829   0.9906   0.9741   0.4344  
0.65       0.7411   0.6008   0.6829   0.9906   0.9741   0.4344  
0.70       0.7411   0.6008   0.6829   0.9906   0.9741   0.4344  
0.75       0.7411   0.6008   0.6829   0.9906   0.9741   0.4344  
0.80       0.7411   0.6008   0.6829   0.9906   0.9741   0.4344  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7411, F1=0.6008, Normal Recall=0.6829, Normal Precision=0.9906, Attack Recall=0.9741, Attack Precision=0.4344

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
0.15       0.7708   0.7183   0.6837   0.9840   0.9741   0.5690   <--
0.20       0.7708   0.7183   0.6837   0.9840   0.9741   0.5690  
0.25       0.7708   0.7183   0.6837   0.9840   0.9741   0.5690  
0.30       0.7708   0.7183   0.6837   0.9840   0.9741   0.5690  
0.35       0.7708   0.7183   0.6837   0.9840   0.9741   0.5690  
0.40       0.7708   0.7183   0.6837   0.9840   0.9741   0.5690  
0.45       0.7708   0.7183   0.6837   0.9840   0.9741   0.5690  
0.50       0.7708   0.7183   0.6837   0.9840   0.9741   0.5690  
0.55       0.7708   0.7183   0.6837   0.9840   0.9741   0.5690  
0.60       0.7708   0.7183   0.6837   0.9840   0.9741   0.5690  
0.65       0.7708   0.7183   0.6837   0.9840   0.9741   0.5690  
0.70       0.7708   0.7183   0.6837   0.9840   0.9741   0.5690  
0.75       0.7708   0.7183   0.6837   0.9840   0.9741   0.5690  
0.80       0.7708   0.7183   0.6837   0.9840   0.9741   0.5690  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7708, F1=0.7183, Normal Recall=0.6837, Normal Precision=0.9840, Attack Recall=0.9741, Attack Precision=0.5690

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
0.15       0.8001   0.7958   0.6841   0.9754   0.9741   0.6727   <--
0.20       0.8001   0.7958   0.6841   0.9754   0.9741   0.6727  
0.25       0.8001   0.7958   0.6841   0.9754   0.9741   0.6727  
0.30       0.8001   0.7958   0.6841   0.9754   0.9741   0.6727  
0.35       0.8001   0.7958   0.6841   0.9754   0.9741   0.6727  
0.40       0.8001   0.7958   0.6841   0.9754   0.9741   0.6727  
0.45       0.8001   0.7958   0.6841   0.9754   0.9741   0.6727  
0.50       0.8001   0.7958   0.6841   0.9754   0.9741   0.6727  
0.55       0.8001   0.7958   0.6841   0.9754   0.9741   0.6727  
0.60       0.8001   0.7958   0.6841   0.9754   0.9741   0.6727  
0.65       0.8001   0.7958   0.6841   0.9754   0.9741   0.6727  
0.70       0.8001   0.7958   0.6841   0.9754   0.9741   0.6727  
0.75       0.8001   0.7958   0.6841   0.9754   0.9741   0.6727  
0.80       0.8001   0.7958   0.6841   0.9754   0.9741   0.6727  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8001, F1=0.7958, Normal Recall=0.6841, Normal Precision=0.9754, Attack Recall=0.9741, Attack Precision=0.6727

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
0.15       0.8293   0.8509   0.6846   0.9635   0.9741   0.7554   <--
0.20       0.8293   0.8509   0.6846   0.9635   0.9741   0.7554  
0.25       0.8293   0.8509   0.6846   0.9635   0.9741   0.7554  
0.30       0.8293   0.8509   0.6846   0.9635   0.9741   0.7554  
0.35       0.8293   0.8509   0.6846   0.9635   0.9741   0.7554  
0.40       0.8293   0.8509   0.6846   0.9635   0.9741   0.7554  
0.45       0.8293   0.8509   0.6846   0.9635   0.9741   0.7554  
0.50       0.8293   0.8509   0.6846   0.9635   0.9741   0.7554  
0.55       0.8293   0.8509   0.6846   0.9635   0.9741   0.7554  
0.60       0.8293   0.8509   0.6846   0.9635   0.9741   0.7554  
0.65       0.8293   0.8509   0.6846   0.9635   0.9741   0.7554  
0.70       0.8293   0.8509   0.6846   0.9635   0.9741   0.7554  
0.75       0.8293   0.8509   0.6846   0.9635   0.9741   0.7554  
0.80       0.8293   0.8509   0.6846   0.9635   0.9741   0.7554  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8293, F1=0.8509, Normal Recall=0.6846, Normal Precision=0.9635, Attack Recall=0.9741, Attack Precision=0.7554

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
0.15       0.8877   0.6305   0.8799   0.9947   0.9581   0.4699   <--
0.20       0.8877   0.6305   0.8799   0.9947   0.9581   0.4699  
0.25       0.8877   0.6305   0.8799   0.9947   0.9581   0.4699  
0.30       0.8877   0.6305   0.8799   0.9947   0.9581   0.4699  
0.35       0.8877   0.6305   0.8799   0.9947   0.9581   0.4699  
0.40       0.8877   0.6305   0.8799   0.9947   0.9581   0.4699  
0.45       0.8877   0.6305   0.8799   0.9947   0.9581   0.4699  
0.50       0.8877   0.6305   0.8799   0.9947   0.9581   0.4699  
0.55       0.8877   0.6305   0.8799   0.9947   0.9581   0.4699  
0.60       0.8877   0.6305   0.8799   0.9947   0.9581   0.4699  
0.65       0.8877   0.6305   0.8799   0.9947   0.9581   0.4699  
0.70       0.8877   0.6305   0.8799   0.9947   0.9581   0.4699  
0.75       0.8877   0.6305   0.8799   0.9947   0.9581   0.4699  
0.80       0.8877   0.6305   0.8799   0.9947   0.9581   0.4699  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8877, F1=0.6305, Normal Recall=0.8799, Normal Precision=0.9947, Attack Recall=0.9581, Attack Precision=0.4699

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
0.15       0.8956   0.7858   0.8801   0.9881   0.9577   0.6663   <--
0.20       0.8956   0.7858   0.8801   0.9881   0.9577   0.6663  
0.25       0.8956   0.7858   0.8801   0.9881   0.9577   0.6663  
0.30       0.8956   0.7858   0.8801   0.9881   0.9577   0.6663  
0.35       0.8956   0.7858   0.8801   0.9881   0.9577   0.6663  
0.40       0.8956   0.7858   0.8801   0.9881   0.9577   0.6663  
0.45       0.8956   0.7858   0.8801   0.9881   0.9577   0.6663  
0.50       0.8956   0.7858   0.8801   0.9881   0.9577   0.6663  
0.55       0.8956   0.7858   0.8801   0.9881   0.9577   0.6663  
0.60       0.8956   0.7858   0.8801   0.9881   0.9577   0.6663  
0.65       0.8956   0.7858   0.8801   0.9881   0.9577   0.6663  
0.70       0.8956   0.7858   0.8801   0.9881   0.9577   0.6663  
0.75       0.8956   0.7858   0.8801   0.9881   0.9577   0.6663  
0.80       0.8956   0.7858   0.8801   0.9881   0.9577   0.6663  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8956, F1=0.7858, Normal Recall=0.8801, Normal Precision=0.9881, Attack Recall=0.9577, Attack Precision=0.6663

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
0.15       0.9031   0.8557   0.8797   0.9798   0.9577   0.7733   <--
0.20       0.9031   0.8557   0.8797   0.9798   0.9577   0.7733  
0.25       0.9031   0.8557   0.8797   0.9798   0.9577   0.7733  
0.30       0.9031   0.8557   0.8797   0.9798   0.9577   0.7733  
0.35       0.9031   0.8557   0.8797   0.9798   0.9577   0.7733  
0.40       0.9031   0.8557   0.8797   0.9798   0.9577   0.7733  
0.45       0.9031   0.8557   0.8797   0.9798   0.9577   0.7733  
0.50       0.9031   0.8557   0.8797   0.9798   0.9577   0.7733  
0.55       0.9031   0.8557   0.8797   0.9798   0.9577   0.7733  
0.60       0.9031   0.8557   0.8797   0.9798   0.9577   0.7733  
0.65       0.9031   0.8557   0.8797   0.9798   0.9577   0.7733  
0.70       0.9031   0.8557   0.8797   0.9798   0.9577   0.7733  
0.75       0.9031   0.8557   0.8797   0.9798   0.9577   0.7733  
0.80       0.9031   0.8557   0.8797   0.9798   0.9577   0.7733  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9031, F1=0.8557, Normal Recall=0.8797, Normal Precision=0.9798, Attack Recall=0.9577, Attack Precision=0.7733

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
0.15       0.9109   0.8958   0.8796   0.9690   0.9577   0.8414   <--
0.20       0.9109   0.8958   0.8796   0.9690   0.9577   0.8414  
0.25       0.9109   0.8958   0.8796   0.9690   0.9577   0.8414  
0.30       0.9109   0.8958   0.8796   0.9690   0.9577   0.8414  
0.35       0.9109   0.8958   0.8796   0.9690   0.9577   0.8414  
0.40       0.9109   0.8958   0.8796   0.9690   0.9577   0.8414  
0.45       0.9109   0.8958   0.8796   0.9690   0.9577   0.8414  
0.50       0.9109   0.8958   0.8796   0.9690   0.9577   0.8414  
0.55       0.9109   0.8958   0.8796   0.9690   0.9577   0.8414  
0.60       0.9109   0.8958   0.8796   0.9690   0.9577   0.8414  
0.65       0.9109   0.8958   0.8796   0.9690   0.9577   0.8414  
0.70       0.9109   0.8958   0.8796   0.9690   0.9577   0.8414  
0.75       0.9109   0.8958   0.8796   0.9690   0.9577   0.8414  
0.80       0.9109   0.8958   0.8796   0.9690   0.9577   0.8414  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9109, F1=0.8958, Normal Recall=0.8796, Normal Precision=0.9690, Attack Recall=0.9577, Attack Precision=0.8414

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
0.15       0.9185   0.9215   0.8792   0.9541   0.9577   0.8880   <--
0.20       0.9185   0.9215   0.8792   0.9541   0.9577   0.8880  
0.25       0.9185   0.9215   0.8792   0.9541   0.9577   0.8880  
0.30       0.9185   0.9215   0.8792   0.9541   0.9577   0.8880  
0.35       0.9185   0.9215   0.8792   0.9541   0.9577   0.8880  
0.40       0.9185   0.9215   0.8792   0.9541   0.9577   0.8880  
0.45       0.9185   0.9215   0.8792   0.9541   0.9577   0.8880  
0.50       0.9185   0.9215   0.8792   0.9541   0.9577   0.8880  
0.55       0.9185   0.9215   0.8792   0.9541   0.9577   0.8880  
0.60       0.9185   0.9215   0.8792   0.9541   0.9577   0.8880  
0.65       0.9185   0.9215   0.8792   0.9541   0.9577   0.8880  
0.70       0.9185   0.9215   0.8792   0.9541   0.9577   0.8880  
0.75       0.9185   0.9215   0.8792   0.9541   0.9577   0.8880  
0.80       0.9185   0.9215   0.8792   0.9541   0.9577   0.8880  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9185, F1=0.9215, Normal Recall=0.8792, Normal Precision=0.9541, Attack Recall=0.9577, Attack Precision=0.8880

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
0.15       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684   <--
0.20       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.25       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.30       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.35       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.40       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.45       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.50       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.55       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.60       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.65       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.70       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.75       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.80       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8871, F1=0.6294, Normal Recall=0.8791, Normal Precision=0.9948, Attack Recall=0.9590, Attack Precision=0.4684

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
0.15       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650   <--
0.20       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.25       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.30       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.35       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.40       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.45       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.50       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.55       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.60       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.65       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.70       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.75       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.80       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8952, F1=0.7853, Normal Recall=0.8793, Normal Precision=0.9884, Attack Recall=0.9586, Attack Precision=0.6650

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
0.15       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722   <--
0.20       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.25       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.30       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.35       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.40       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.45       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.50       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.55       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.60       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.65       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.70       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.75       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.80       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9028, F1=0.8554, Normal Recall=0.8788, Normal Precision=0.9802, Attack Recall=0.9586, Attack Precision=0.7722

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
0.15       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407   <--
0.20       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.25       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.30       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.35       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.40       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.45       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.50       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.55       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.60       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.65       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.70       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.75       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.80       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9108, F1=0.8958, Normal Recall=0.8789, Normal Precision=0.9696, Attack Recall=0.9586, Attack Precision=0.8407

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
0.15       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875   <--
0.20       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.25       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.30       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.35       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.40       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.45       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.50       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.55       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.60       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.65       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.70       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.75       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.80       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9186, F1=0.9217, Normal Recall=0.8785, Normal Precision=0.9550, Attack Recall=0.9586, Attack Precision=0.8875

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
0.15       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684   <--
0.20       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.25       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.30       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.35       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.40       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.45       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.50       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.55       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.60       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.65       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.70       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.75       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
0.80       0.8871   0.6294   0.8791   0.9948   0.9590   0.4684  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8871, F1=0.6294, Normal Recall=0.8791, Normal Precision=0.9948, Attack Recall=0.9590, Attack Precision=0.4684

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
0.15       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650   <--
0.20       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.25       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.30       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.35       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.40       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.45       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.50       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.55       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.60       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.65       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.70       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.75       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
0.80       0.8952   0.7853   0.8793   0.9884   0.9586   0.6650  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8952, F1=0.7853, Normal Recall=0.8793, Normal Precision=0.9884, Attack Recall=0.9586, Attack Precision=0.6650

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
0.15       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722   <--
0.20       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.25       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.30       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.35       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.40       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.45       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.50       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.55       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.60       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.65       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.70       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.75       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
0.80       0.9028   0.8554   0.8788   0.9802   0.9586   0.7722  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9028, F1=0.8554, Normal Recall=0.8788, Normal Precision=0.9802, Attack Recall=0.9586, Attack Precision=0.7722

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
0.15       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407   <--
0.20       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.25       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.30       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.35       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.40       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.45       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.50       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.55       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.60       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.65       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.70       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.75       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
0.80       0.9108   0.8958   0.8789   0.9696   0.9586   0.8407  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9108, F1=0.8958, Normal Recall=0.8789, Normal Precision=0.9696, Attack Recall=0.9586, Attack Precision=0.8407

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
0.15       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875   <--
0.20       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.25       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.30       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.35       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.40       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.45       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.50       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.55       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.60       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.65       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.70       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.75       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
0.80       0.9186   0.9217   0.8785   0.9550   0.9586   0.8875  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9186, F1=0.9217, Normal Recall=0.8785, Normal Precision=0.9550, Attack Recall=0.9586, Attack Precision=0.8875

```

