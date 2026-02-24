# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e1_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-17 07:45:00 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 80 |
| **Local epochs** | 1 |
| **Batch size** | 128 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9217 | 0.9186 | 0.9148 | 0.9116 | 0.9077 | 0.9037 | 0.9006 | 0.8972 | 0.8931 | 0.8890 | 0.8859 |
| QAT+Prune only | 0.7973 | 0.7954 | 0.7914 | 0.7871 | 0.7831 | 0.7805 | 0.7760 | 0.7725 | 0.7705 | 0.7643 | 0.7619 |
| QAT+PTQ | 0.7975 | 0.7954 | 0.7914 | 0.7870 | 0.7829 | 0.7803 | 0.7756 | 0.7721 | 0.7700 | 0.7636 | 0.7611 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7975 | 0.7954 | 0.7914 | 0.7870 | 0.7829 | 0.7803 | 0.7756 | 0.7721 | 0.7700 | 0.7636 | 0.7611 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6853 | 0.8061 | 0.8574 | 0.8848 | 0.9020 | 0.9145 | 0.9235 | 0.9299 | 0.9349 | 0.9395 |
| QAT+Prune only | 0.0000 | 0.4273 | 0.5937 | 0.6823 | 0.7375 | 0.7764 | 0.8032 | 0.8242 | 0.8416 | 0.8534 | 0.8649 |
| QAT+PTQ | 0.0000 | 0.4270 | 0.5934 | 0.6819 | 0.7371 | 0.7760 | 0.8027 | 0.8238 | 0.8411 | 0.8528 | 0.8643 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4270 | 0.5934 | 0.6819 | 0.7371 | 0.7760 | 0.8027 | 0.8238 | 0.8411 | 0.8528 | 0.8643 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9217 | 0.9223 | 0.9220 | 0.9226 | 0.9223 | 0.9216 | 0.9226 | 0.9237 | 0.9222 | 0.9174 | 0.0000 |
| QAT+Prune only | 0.7973 | 0.7989 | 0.7988 | 0.7979 | 0.7972 | 0.7992 | 0.7971 | 0.7974 | 0.8049 | 0.7862 | 0.0000 |
| QAT+PTQ | 0.7975 | 0.7991 | 0.7990 | 0.7980 | 0.7974 | 0.7996 | 0.7973 | 0.7979 | 0.8056 | 0.7864 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7975 | 0.7991 | 0.7990 | 0.7980 | 0.7974 | 0.7996 | 0.7973 | 0.7979 | 0.8056 | 0.7864 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9217 | 0.0000 | 0.0000 | 0.0000 | 0.9217 | 1.0000 |
| 90 | 10 | 299,940 | 0.9186 | 0.5588 | 0.8858 | 0.6853 | 0.9223 | 0.9864 |
| 80 | 20 | 291,350 | 0.9148 | 0.7395 | 0.8859 | 0.8061 | 0.9220 | 0.9700 |
| 70 | 30 | 194,230 | 0.9116 | 0.8306 | 0.8859 | 0.8574 | 0.9226 | 0.9497 |
| 60 | 40 | 145,675 | 0.9077 | 0.8837 | 0.8859 | 0.8848 | 0.9223 | 0.9238 |
| 50 | 50 | 116,540 | 0.9037 | 0.9187 | 0.8859 | 0.9020 | 0.9216 | 0.8898 |
| 40 | 60 | 97,115 | 0.9006 | 0.9450 | 0.8859 | 0.9145 | 0.9226 | 0.8435 |
| 30 | 70 | 83,240 | 0.8972 | 0.9644 | 0.8859 | 0.9235 | 0.9237 | 0.7762 |
| 20 | 80 | 72,835 | 0.8931 | 0.9785 | 0.8859 | 0.9299 | 0.9222 | 0.6689 |
| 10 | 90 | 64,740 | 0.8890 | 0.9897 | 0.8859 | 0.9349 | 0.9174 | 0.4718 |
| 0 | 100 | 58,270 | 0.8859 | 1.0000 | 0.8859 | 0.9395 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7973 | 0.0000 | 0.0000 | 0.0000 | 0.7973 | 1.0000 |
| 90 | 10 | 299,940 | 0.7954 | 0.2967 | 0.7634 | 0.4273 | 0.7989 | 0.9681 |
| 80 | 20 | 291,350 | 0.7914 | 0.4863 | 0.7619 | 0.5937 | 0.7988 | 0.9306 |
| 70 | 30 | 194,230 | 0.7871 | 0.6177 | 0.7619 | 0.6823 | 0.7979 | 0.8866 |
| 60 | 40 | 145,675 | 0.7831 | 0.7146 | 0.7619 | 0.7375 | 0.7972 | 0.8339 |
| 50 | 50 | 116,540 | 0.7805 | 0.7914 | 0.7619 | 0.7764 | 0.7992 | 0.7705 |
| 40 | 60 | 97,115 | 0.7760 | 0.8493 | 0.7619 | 0.8032 | 0.7971 | 0.6906 |
| 30 | 70 | 83,240 | 0.7725 | 0.8977 | 0.7619 | 0.8242 | 0.7974 | 0.5894 |
| 20 | 80 | 72,835 | 0.7705 | 0.9398 | 0.7619 | 0.8416 | 0.8049 | 0.4580 |
| 10 | 90 | 64,740 | 0.7643 | 0.9698 | 0.7619 | 0.8534 | 0.7862 | 0.2684 |
| 0 | 100 | 58,270 | 0.7619 | 1.0000 | 0.7619 | 0.8649 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7975 | 0.0000 | 0.0000 | 0.0000 | 0.7975 | 1.0000 |
| 90 | 10 | 299,940 | 0.7954 | 0.2966 | 0.7623 | 0.4270 | 0.7991 | 0.9680 |
| 80 | 20 | 291,350 | 0.7914 | 0.4863 | 0.7611 | 0.5934 | 0.7990 | 0.9304 |
| 70 | 30 | 194,230 | 0.7870 | 0.6176 | 0.7611 | 0.6819 | 0.7980 | 0.8863 |
| 60 | 40 | 145,675 | 0.7829 | 0.7147 | 0.7611 | 0.7371 | 0.7974 | 0.8335 |
| 50 | 50 | 116,540 | 0.7803 | 0.7916 | 0.7611 | 0.7760 | 0.7996 | 0.7699 |
| 40 | 60 | 97,115 | 0.7756 | 0.8492 | 0.7611 | 0.8027 | 0.7973 | 0.6899 |
| 30 | 70 | 83,240 | 0.7721 | 0.8978 | 0.7611 | 0.8238 | 0.7979 | 0.5887 |
| 20 | 80 | 72,835 | 0.7700 | 0.9400 | 0.7611 | 0.8411 | 0.8056 | 0.4574 |
| 10 | 90 | 64,740 | 0.7636 | 0.9698 | 0.7611 | 0.8528 | 0.7864 | 0.2678 |
| 0 | 100 | 58,270 | 0.7611 | 1.0000 | 0.7611 | 0.8643 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7975 | 0.0000 | 0.0000 | 0.0000 | 0.7975 | 1.0000 |
| 90 | 10 | 299,940 | 0.7954 | 0.2966 | 0.7623 | 0.4270 | 0.7991 | 0.9680 |
| 80 | 20 | 291,350 | 0.7914 | 0.4863 | 0.7611 | 0.5934 | 0.7990 | 0.9304 |
| 70 | 30 | 194,230 | 0.7870 | 0.6176 | 0.7611 | 0.6819 | 0.7980 | 0.8863 |
| 60 | 40 | 145,675 | 0.7829 | 0.7147 | 0.7611 | 0.7371 | 0.7974 | 0.8335 |
| 50 | 50 | 116,540 | 0.7803 | 0.7916 | 0.7611 | 0.7760 | 0.7996 | 0.7699 |
| 40 | 60 | 97,115 | 0.7756 | 0.8492 | 0.7611 | 0.8027 | 0.7973 | 0.6899 |
| 30 | 70 | 83,240 | 0.7721 | 0.8978 | 0.7611 | 0.8238 | 0.7979 | 0.5887 |
| 20 | 80 | 72,835 | 0.7700 | 0.9400 | 0.7611 | 0.8411 | 0.8056 | 0.4574 |
| 10 | 90 | 64,740 | 0.7636 | 0.9698 | 0.7611 | 0.8528 | 0.7864 | 0.2678 |
| 0 | 100 | 58,270 | 0.7611 | 1.0000 | 0.7611 | 0.8643 | 0.0000 | 0.0000 |


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
0.15       0.9189   0.6866   0.9223   0.9867   0.8883   0.5595   <--
0.20       0.9189   0.6866   0.9223   0.9867   0.8883   0.5595  
0.25       0.9189   0.6866   0.9223   0.9867   0.8883   0.5595  
0.30       0.9189   0.6866   0.9223   0.9867   0.8883   0.5595  
0.35       0.9189   0.6866   0.9223   0.9867   0.8883   0.5595  
0.40       0.9189   0.6866   0.9223   0.9867   0.8883   0.5595  
0.45       0.9189   0.6866   0.9223   0.9867   0.8883   0.5595  
0.50       0.9189   0.6866   0.9223   0.9867   0.8883   0.5595  
0.55       0.9189   0.6866   0.9223   0.9867   0.8883   0.5595  
0.60       0.9189   0.6866   0.9223   0.9867   0.8883   0.5595  
0.65       0.9189   0.6866   0.9223   0.9867   0.8883   0.5595  
0.70       0.9189   0.6866   0.9223   0.9867   0.8883   0.5595  
0.75       0.9189   0.6866   0.9223   0.9867   0.8883   0.5595  
0.80       0.9189   0.6866   0.9223   0.9867   0.8883   0.5595  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9189, F1=0.6866, Normal Recall=0.9223, Normal Precision=0.9867, Attack Recall=0.8883, Attack Precision=0.5595

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
0.15       0.9151   0.8066   0.9223   0.9700   0.8859   0.7404   <--
0.20       0.9151   0.8066   0.9223   0.9700   0.8859   0.7404  
0.25       0.9151   0.8066   0.9223   0.9700   0.8859   0.7404  
0.30       0.9151   0.8066   0.9223   0.9700   0.8859   0.7404  
0.35       0.9151   0.8066   0.9223   0.9700   0.8859   0.7404  
0.40       0.9151   0.8066   0.9223   0.9700   0.8859   0.7404  
0.45       0.9151   0.8066   0.9223   0.9700   0.8859   0.7404  
0.50       0.9151   0.8066   0.9223   0.9700   0.8859   0.7404  
0.55       0.9151   0.8066   0.9223   0.9700   0.8859   0.7404  
0.60       0.9151   0.8066   0.9223   0.9700   0.8859   0.7404  
0.65       0.9151   0.8066   0.9223   0.9700   0.8859   0.7404  
0.70       0.9151   0.8066   0.9223   0.9700   0.8859   0.7404  
0.75       0.9151   0.8066   0.9223   0.9700   0.8859   0.7404  
0.80       0.9151   0.8066   0.9223   0.9700   0.8859   0.7404  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9151, F1=0.8066, Normal Recall=0.9223, Normal Precision=0.9700, Attack Recall=0.8859, Attack Precision=0.7404

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
0.15       0.9114   0.8571   0.9223   0.9496   0.8859   0.8302   <--
0.20       0.9114   0.8571   0.9223   0.9496   0.8859   0.8302  
0.25       0.9114   0.8571   0.9223   0.9496   0.8859   0.8302  
0.30       0.9114   0.8571   0.9223   0.9496   0.8859   0.8302  
0.35       0.9114   0.8571   0.9223   0.9496   0.8859   0.8302  
0.40       0.9114   0.8571   0.9223   0.9496   0.8859   0.8302  
0.45       0.9114   0.8571   0.9223   0.9496   0.8859   0.8302  
0.50       0.9114   0.8571   0.9223   0.9496   0.8859   0.8302  
0.55       0.9114   0.8571   0.9223   0.9496   0.8859   0.8302  
0.60       0.9114   0.8571   0.9223   0.9496   0.8859   0.8302  
0.65       0.9114   0.8571   0.9223   0.9496   0.8859   0.8302  
0.70       0.9114   0.8571   0.9223   0.9496   0.8859   0.8302  
0.75       0.9114   0.8571   0.9223   0.9496   0.8859   0.8302  
0.80       0.9114   0.8571   0.9223   0.9496   0.8859   0.8302  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9114, F1=0.8571, Normal Recall=0.9223, Normal Precision=0.9496, Attack Recall=0.8859, Attack Precision=0.8302

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
0.15       0.9076   0.8847   0.9221   0.9238   0.8859   0.8835   <--
0.20       0.9076   0.8847   0.9221   0.9238   0.8859   0.8835  
0.25       0.9076   0.8847   0.9221   0.9238   0.8859   0.8835  
0.30       0.9076   0.8847   0.9221   0.9238   0.8859   0.8835  
0.35       0.9076   0.8847   0.9221   0.9238   0.8859   0.8835  
0.40       0.9076   0.8847   0.9221   0.9238   0.8859   0.8835  
0.45       0.9076   0.8847   0.9221   0.9238   0.8859   0.8835  
0.50       0.9076   0.8847   0.9221   0.9238   0.8859   0.8835  
0.55       0.9076   0.8847   0.9221   0.9238   0.8859   0.8835  
0.60       0.9076   0.8847   0.9221   0.9238   0.8859   0.8835  
0.65       0.9076   0.8847   0.9221   0.9238   0.8859   0.8835  
0.70       0.9076   0.8847   0.9221   0.9238   0.8859   0.8835  
0.75       0.9076   0.8847   0.9221   0.9238   0.8859   0.8835  
0.80       0.9076   0.8847   0.9221   0.9238   0.8859   0.8835  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9076, F1=0.8847, Normal Recall=0.9221, Normal Precision=0.9238, Attack Recall=0.8859, Attack Precision=0.8835

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
0.15       0.9041   0.9023   0.9222   0.8899   0.8859   0.9193   <--
0.20       0.9041   0.9023   0.9222   0.8899   0.8859   0.9193  
0.25       0.9041   0.9023   0.9222   0.8899   0.8859   0.9193  
0.30       0.9041   0.9023   0.9222   0.8899   0.8859   0.9193  
0.35       0.9041   0.9023   0.9222   0.8899   0.8859   0.9193  
0.40       0.9041   0.9023   0.9222   0.8899   0.8859   0.9193  
0.45       0.9041   0.9023   0.9222   0.8899   0.8859   0.9193  
0.50       0.9041   0.9023   0.9222   0.8899   0.8859   0.9193  
0.55       0.9041   0.9023   0.9222   0.8899   0.8859   0.9193  
0.60       0.9041   0.9023   0.9222   0.8899   0.8859   0.9193  
0.65       0.9041   0.9023   0.9222   0.8899   0.8859   0.9193  
0.70       0.9041   0.9023   0.9222   0.8899   0.8859   0.9193  
0.75       0.9041   0.9023   0.9222   0.8899   0.8859   0.9193  
0.80       0.9041   0.9023   0.9222   0.8899   0.8859   0.9193  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9041, F1=0.9023, Normal Recall=0.9222, Normal Precision=0.8899, Attack Recall=0.8859, Attack Precision=0.9193

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
0.15       0.7953   0.4269   0.7989   0.9680   0.7624   0.2964   <--
0.20       0.7953   0.4269   0.7989   0.9680   0.7624   0.2964  
0.25       0.7953   0.4269   0.7989   0.9680   0.7624   0.2964  
0.30       0.7953   0.4269   0.7989   0.9680   0.7624   0.2964  
0.35       0.7953   0.4269   0.7989   0.9680   0.7624   0.2964  
0.40       0.7953   0.4269   0.7989   0.9680   0.7624   0.2964  
0.45       0.7953   0.4269   0.7989   0.9680   0.7624   0.2964  
0.50       0.7953   0.4269   0.7989   0.9680   0.7624   0.2964  
0.55       0.7953   0.4269   0.7989   0.9680   0.7624   0.2964  
0.60       0.7953   0.4269   0.7989   0.9680   0.7624   0.2964  
0.65       0.7953   0.4269   0.7989   0.9680   0.7624   0.2964  
0.70       0.7953   0.4269   0.7989   0.9680   0.7624   0.2964  
0.75       0.7953   0.4269   0.7989   0.9680   0.7624   0.2964  
0.80       0.7953   0.4269   0.7989   0.9680   0.7624   0.2964  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7953, F1=0.4269, Normal Recall=0.7989, Normal Precision=0.9680, Attack Recall=0.7624, Attack Precision=0.2964

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
0.15       0.7913   0.5936   0.7987   0.9306   0.7619   0.4862   <--
0.20       0.7913   0.5936   0.7987   0.9306   0.7619   0.4862  
0.25       0.7913   0.5936   0.7987   0.9306   0.7619   0.4862  
0.30       0.7913   0.5936   0.7987   0.9306   0.7619   0.4862  
0.35       0.7913   0.5936   0.7987   0.9306   0.7619   0.4862  
0.40       0.7913   0.5936   0.7987   0.9306   0.7619   0.4862  
0.45       0.7913   0.5936   0.7987   0.9306   0.7619   0.4862  
0.50       0.7913   0.5936   0.7987   0.9306   0.7619   0.4862  
0.55       0.7913   0.5936   0.7987   0.9306   0.7619   0.4862  
0.60       0.7913   0.5936   0.7987   0.9306   0.7619   0.4862  
0.65       0.7913   0.5936   0.7987   0.9306   0.7619   0.4862  
0.70       0.7913   0.5936   0.7987   0.9306   0.7619   0.4862  
0.75       0.7913   0.5936   0.7987   0.9306   0.7619   0.4862  
0.80       0.7913   0.5936   0.7987   0.9306   0.7619   0.4862  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7913, F1=0.5936, Normal Recall=0.7987, Normal Precision=0.9306, Attack Recall=0.7619, Attack Precision=0.4862

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
0.15       0.7869   0.6821   0.7976   0.8866   0.7619   0.6174   <--
0.20       0.7869   0.6821   0.7976   0.8866   0.7619   0.6174  
0.25       0.7869   0.6821   0.7976   0.8866   0.7619   0.6174  
0.30       0.7869   0.6821   0.7976   0.8866   0.7619   0.6174  
0.35       0.7869   0.6821   0.7976   0.8866   0.7619   0.6174  
0.40       0.7869   0.6821   0.7976   0.8866   0.7619   0.6174  
0.45       0.7869   0.6821   0.7976   0.8866   0.7619   0.6174  
0.50       0.7869   0.6821   0.7976   0.8866   0.7619   0.6174  
0.55       0.7869   0.6821   0.7976   0.8866   0.7619   0.6174  
0.60       0.7869   0.6821   0.7976   0.8866   0.7619   0.6174  
0.65       0.7869   0.6821   0.7976   0.8866   0.7619   0.6174  
0.70       0.7869   0.6821   0.7976   0.8866   0.7619   0.6174  
0.75       0.7869   0.6821   0.7976   0.8866   0.7619   0.6174  
0.80       0.7869   0.6821   0.7976   0.8866   0.7619   0.6174  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7869, F1=0.6821, Normal Recall=0.7976, Normal Precision=0.8866, Attack Recall=0.7619, Attack Precision=0.6174

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
0.15       0.7827   0.7371   0.7965   0.8338   0.7619   0.7139   <--
0.20       0.7827   0.7371   0.7965   0.8338   0.7619   0.7139  
0.25       0.7827   0.7371   0.7965   0.8338   0.7619   0.7139  
0.30       0.7827   0.7371   0.7965   0.8338   0.7619   0.7139  
0.35       0.7827   0.7371   0.7965   0.8338   0.7619   0.7139  
0.40       0.7827   0.7371   0.7965   0.8338   0.7619   0.7139  
0.45       0.7827   0.7371   0.7965   0.8338   0.7619   0.7139  
0.50       0.7827   0.7371   0.7965   0.8338   0.7619   0.7139  
0.55       0.7827   0.7371   0.7965   0.8338   0.7619   0.7139  
0.60       0.7827   0.7371   0.7965   0.8338   0.7619   0.7139  
0.65       0.7827   0.7371   0.7965   0.8338   0.7619   0.7139  
0.70       0.7827   0.7371   0.7965   0.8338   0.7619   0.7139  
0.75       0.7827   0.7371   0.7965   0.8338   0.7619   0.7139  
0.80       0.7827   0.7371   0.7965   0.8338   0.7619   0.7139  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7827, F1=0.7371, Normal Recall=0.7965, Normal Precision=0.8338, Attack Recall=0.7619, Attack Precision=0.7139

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
0.15       0.7793   0.7754   0.7968   0.7699   0.7619   0.7894   <--
0.20       0.7793   0.7754   0.7968   0.7699   0.7619   0.7894  
0.25       0.7793   0.7754   0.7968   0.7699   0.7619   0.7894  
0.30       0.7793   0.7754   0.7968   0.7699   0.7619   0.7894  
0.35       0.7793   0.7754   0.7968   0.7699   0.7619   0.7894  
0.40       0.7793   0.7754   0.7968   0.7699   0.7619   0.7894  
0.45       0.7793   0.7754   0.7968   0.7699   0.7619   0.7894  
0.50       0.7793   0.7754   0.7968   0.7699   0.7619   0.7894  
0.55       0.7793   0.7754   0.7968   0.7699   0.7619   0.7894  
0.60       0.7793   0.7754   0.7968   0.7699   0.7619   0.7894  
0.65       0.7793   0.7754   0.7968   0.7699   0.7619   0.7894  
0.70       0.7793   0.7754   0.7968   0.7699   0.7619   0.7894  
0.75       0.7793   0.7754   0.7968   0.7699   0.7619   0.7894  
0.80       0.7793   0.7754   0.7968   0.7699   0.7619   0.7894  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7793, F1=0.7754, Normal Recall=0.7968, Normal Precision=0.7699, Attack Recall=0.7619, Attack Precision=0.7894

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
0.15       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963   <--
0.20       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.25       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.30       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.35       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.40       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.45       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.50       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.55       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.60       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.65       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.70       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.75       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.80       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7953, F1=0.4266, Normal Recall=0.7991, Normal Precision=0.9679, Attack Recall=0.7614, Attack Precision=0.2963

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
0.15       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862   <--
0.20       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.25       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.30       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.35       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.40       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.45       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.50       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.55       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.60       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.65       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.70       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.75       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.80       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7914, F1=0.5934, Normal Recall=0.7990, Normal Precision=0.9304, Attack Recall=0.7611, Attack Precision=0.4862

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
0.15       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174   <--
0.20       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.25       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.30       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.35       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.40       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.45       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.50       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.55       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.60       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.65       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.70       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.75       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.80       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7869, F1=0.6818, Normal Recall=0.7979, Normal Precision=0.8863, Attack Recall=0.7611, Attack Precision=0.6174

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
0.15       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140   <--
0.20       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.25       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.30       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.35       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.40       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.45       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.50       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.55       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.60       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.65       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.70       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.75       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.80       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7825, F1=0.7368, Normal Recall=0.7968, Normal Precision=0.8334, Attack Recall=0.7611, Attack Precision=0.7140

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
0.15       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896   <--
0.20       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.25       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.30       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.35       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.40       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.45       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.50       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.55       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.60       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.65       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.70       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.75       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.80       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7791, F1=0.7751, Normal Recall=0.7972, Normal Precision=0.7694, Attack Recall=0.7611, Attack Precision=0.7896

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
0.15       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963   <--
0.20       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.25       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.30       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.35       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.40       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.45       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.50       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.55       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.60       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.65       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.70       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.75       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
0.80       0.7953   0.4266   0.7991   0.9679   0.7614   0.2963  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7953, F1=0.4266, Normal Recall=0.7991, Normal Precision=0.9679, Attack Recall=0.7614, Attack Precision=0.2963

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
0.15       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862   <--
0.20       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.25       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.30       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.35       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.40       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.45       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.50       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.55       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.60       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.65       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.70       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.75       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
0.80       0.7914   0.5934   0.7990   0.9304   0.7611   0.4862  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7914, F1=0.5934, Normal Recall=0.7990, Normal Precision=0.9304, Attack Recall=0.7611, Attack Precision=0.4862

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
0.15       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174   <--
0.20       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.25       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.30       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.35       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.40       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.45       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.50       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.55       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.60       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.65       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.70       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.75       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
0.80       0.7869   0.6818   0.7979   0.8863   0.7611   0.6174  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7869, F1=0.6818, Normal Recall=0.7979, Normal Precision=0.8863, Attack Recall=0.7611, Attack Precision=0.6174

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
0.15       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140   <--
0.20       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.25       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.30       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.35       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.40       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.45       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.50       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.55       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.60       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.65       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.70       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.75       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
0.80       0.7825   0.7368   0.7968   0.8334   0.7611   0.7140  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7825, F1=0.7368, Normal Recall=0.7968, Normal Precision=0.8334, Attack Recall=0.7611, Attack Precision=0.7140

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
0.15       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896   <--
0.20       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.25       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.30       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.35       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.40       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.45       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.50       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.55       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.60       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.65       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.70       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.75       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
0.80       0.7791   0.7751   0.7972   0.7694   0.7611   0.7896  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7791, F1=0.7751, Normal Recall=0.7972, Normal Precision=0.7694, Attack Recall=0.7611, Attack Precision=0.7896

```

