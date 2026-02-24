# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e1_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-16 00:23:09 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7819 | 0.7892 | 0.7972 | 0.8065 | 0.8142 | 0.8229 | 0.8314 | 0.8410 | 0.8484 | 0.8568 | 0.8653 |
| QAT+Prune only | 0.7308 | 0.7581 | 0.7837 | 0.8100 | 0.8358 | 0.8602 | 0.8889 | 0.9138 | 0.9405 | 0.9649 | 0.9918 |
| QAT+PTQ | 0.7293 | 0.7568 | 0.7826 | 0.8091 | 0.8347 | 0.8594 | 0.8881 | 0.9132 | 0.9400 | 0.9647 | 0.9918 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7293 | 0.7568 | 0.7826 | 0.8091 | 0.8347 | 0.8594 | 0.8881 | 0.9132 | 0.9400 | 0.9647 | 0.9918 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.4510 | 0.6305 | 0.7285 | 0.7884 | 0.8301 | 0.8603 | 0.8840 | 0.9013 | 0.9158 | 0.9278 |
| QAT+Prune only | 0.0000 | 0.4506 | 0.6472 | 0.7580 | 0.8285 | 0.8764 | 0.9146 | 0.9416 | 0.9638 | 0.9807 | 0.9959 |
| QAT+PTQ | 0.0000 | 0.4493 | 0.6460 | 0.7571 | 0.8276 | 0.8759 | 0.9140 | 0.9412 | 0.9636 | 0.9806 | 0.9959 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4493 | 0.6460 | 0.7571 | 0.8276 | 0.8759 | 0.9140 | 0.9412 | 0.9636 | 0.9806 | 0.9959 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7819 | 0.7806 | 0.7801 | 0.7813 | 0.7800 | 0.7804 | 0.7805 | 0.7844 | 0.7807 | 0.7805 | 0.0000 |
| QAT+Prune only | 0.7308 | 0.7321 | 0.7317 | 0.7321 | 0.7318 | 0.7285 | 0.7346 | 0.7317 | 0.7349 | 0.7229 | 0.0000 |
| QAT+PTQ | 0.7293 | 0.7307 | 0.7303 | 0.7307 | 0.7300 | 0.7271 | 0.7325 | 0.7299 | 0.7329 | 0.7207 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7293 | 0.7307 | 0.7303 | 0.7307 | 0.7300 | 0.7271 | 0.7325 | 0.7299 | 0.7329 | 0.7207 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7819 | 0.0000 | 0.0000 | 0.0000 | 0.7819 | 1.0000 |
| 90 | 10 | 299,940 | 0.7892 | 0.3049 | 0.8661 | 0.4510 | 0.7806 | 0.9813 |
| 80 | 20 | 291,350 | 0.7972 | 0.4959 | 0.8653 | 0.6305 | 0.7801 | 0.9586 |
| 70 | 30 | 194,230 | 0.8065 | 0.6290 | 0.8653 | 0.7285 | 0.7813 | 0.9312 |
| 60 | 40 | 145,675 | 0.8142 | 0.7240 | 0.8653 | 0.7884 | 0.7800 | 0.8968 |
| 50 | 50 | 116,540 | 0.8229 | 0.7976 | 0.8653 | 0.8301 | 0.7804 | 0.8528 |
| 40 | 60 | 97,115 | 0.8314 | 0.8554 | 0.8653 | 0.8603 | 0.7805 | 0.7944 |
| 30 | 70 | 83,240 | 0.8410 | 0.9035 | 0.8653 | 0.8840 | 0.7844 | 0.7139 |
| 20 | 80 | 72,835 | 0.8484 | 0.9404 | 0.8653 | 0.9013 | 0.7807 | 0.5917 |
| 10 | 90 | 64,740 | 0.8568 | 0.9726 | 0.8653 | 0.9158 | 0.7805 | 0.3917 |
| 0 | 100 | 58,270 | 0.8653 | 1.0000 | 0.8653 | 0.9278 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7308 | 0.0000 | 0.0000 | 0.0000 | 0.7308 | 1.0000 |
| 90 | 10 | 299,940 | 0.7581 | 0.2915 | 0.9920 | 0.4506 | 0.7321 | 0.9988 |
| 80 | 20 | 291,350 | 0.7837 | 0.4803 | 0.9918 | 0.6472 | 0.7317 | 0.9972 |
| 70 | 30 | 194,230 | 0.8100 | 0.6134 | 0.9918 | 0.7580 | 0.7321 | 0.9952 |
| 60 | 40 | 145,675 | 0.8358 | 0.7114 | 0.9918 | 0.8285 | 0.7318 | 0.9926 |
| 50 | 50 | 116,540 | 0.8602 | 0.7851 | 0.9918 | 0.8764 | 0.7285 | 0.9889 |
| 40 | 60 | 97,115 | 0.8889 | 0.8486 | 0.9918 | 0.9146 | 0.7346 | 0.9836 |
| 30 | 70 | 83,240 | 0.9138 | 0.8961 | 0.9918 | 0.9416 | 0.7317 | 0.9746 |
| 20 | 80 | 72,835 | 0.9405 | 0.9374 | 0.9918 | 0.9638 | 0.7349 | 0.9574 |
| 10 | 90 | 64,740 | 0.9649 | 0.9699 | 0.9918 | 0.9807 | 0.7229 | 0.9077 |
| 0 | 100 | 58,270 | 0.9918 | 1.0000 | 0.9918 | 0.9959 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7293 | 0.0000 | 0.0000 | 0.0000 | 0.7293 | 1.0000 |
| 90 | 10 | 299,940 | 0.7568 | 0.2904 | 0.9920 | 0.4493 | 0.7307 | 0.9988 |
| 80 | 20 | 291,350 | 0.7826 | 0.4790 | 0.9918 | 0.6460 | 0.7303 | 0.9972 |
| 70 | 30 | 194,230 | 0.8091 | 0.6122 | 0.9918 | 0.7571 | 0.7307 | 0.9952 |
| 60 | 40 | 145,675 | 0.8347 | 0.7101 | 0.9918 | 0.8276 | 0.7300 | 0.9925 |
| 50 | 50 | 116,540 | 0.8594 | 0.7842 | 0.9918 | 0.8759 | 0.7271 | 0.9888 |
| 40 | 60 | 97,115 | 0.8881 | 0.8476 | 0.9918 | 0.9140 | 0.7325 | 0.9834 |
| 30 | 70 | 83,240 | 0.9132 | 0.8955 | 0.9918 | 0.9412 | 0.7299 | 0.9744 |
| 20 | 80 | 72,835 | 0.9400 | 0.9369 | 0.9918 | 0.9636 | 0.7329 | 0.9571 |
| 10 | 90 | 64,740 | 0.9647 | 0.9697 | 0.9918 | 0.9806 | 0.7207 | 0.9069 |
| 0 | 100 | 58,270 | 0.9918 | 1.0000 | 0.9918 | 0.9959 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7293 | 0.0000 | 0.0000 | 0.0000 | 0.7293 | 1.0000 |
| 90 | 10 | 299,940 | 0.7568 | 0.2904 | 0.9920 | 0.4493 | 0.7307 | 0.9988 |
| 80 | 20 | 291,350 | 0.7826 | 0.4790 | 0.9918 | 0.6460 | 0.7303 | 0.9972 |
| 70 | 30 | 194,230 | 0.8091 | 0.6122 | 0.9918 | 0.7571 | 0.7307 | 0.9952 |
| 60 | 40 | 145,675 | 0.8347 | 0.7101 | 0.9918 | 0.8276 | 0.7300 | 0.9925 |
| 50 | 50 | 116,540 | 0.8594 | 0.7842 | 0.9918 | 0.8759 | 0.7271 | 0.9888 |
| 40 | 60 | 97,115 | 0.8881 | 0.8476 | 0.9918 | 0.9140 | 0.7325 | 0.9834 |
| 30 | 70 | 83,240 | 0.9132 | 0.8955 | 0.9918 | 0.9412 | 0.7299 | 0.9744 |
| 20 | 80 | 72,835 | 0.9400 | 0.9369 | 0.9918 | 0.9636 | 0.7329 | 0.9571 |
| 10 | 90 | 64,740 | 0.9647 | 0.9697 | 0.9918 | 0.9806 | 0.7207 | 0.9069 |
| 0 | 100 | 58,270 | 0.9918 | 1.0000 | 0.9918 | 0.9959 | 0.0000 | 0.0000 |


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
0.15       0.7889   0.4501   0.7806   0.9810   0.8637   0.3043   <--
0.20       0.7889   0.4501   0.7806   0.9810   0.8637   0.3043  
0.25       0.7889   0.4501   0.7806   0.9810   0.8637   0.3043  
0.30       0.7889   0.4501   0.7806   0.9810   0.8637   0.3043  
0.35       0.7889   0.4501   0.7806   0.9810   0.8637   0.3043  
0.40       0.7889   0.4501   0.7806   0.9810   0.8637   0.3043  
0.45       0.7889   0.4501   0.7806   0.9810   0.8637   0.3043  
0.50       0.7889   0.4501   0.7806   0.9810   0.8637   0.3043  
0.55       0.7889   0.4501   0.7806   0.9810   0.8637   0.3043  
0.60       0.7889   0.4501   0.7806   0.9810   0.8637   0.3043  
0.65       0.7889   0.4501   0.7806   0.9810   0.8637   0.3043  
0.70       0.7889   0.4501   0.7806   0.9810   0.8637   0.3043  
0.75       0.7889   0.4501   0.7806   0.9810   0.8637   0.3043  
0.80       0.7889   0.4501   0.7806   0.9810   0.8637   0.3043  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7889, F1=0.4501, Normal Recall=0.7806, Normal Precision=0.9810, Attack Recall=0.8637, Attack Precision=0.3043

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
0.15       0.7976   0.6310   0.7806   0.9586   0.8653   0.4965   <--
0.20       0.7976   0.6310   0.7806   0.9586   0.8653   0.4965  
0.25       0.7976   0.6310   0.7806   0.9586   0.8653   0.4965  
0.30       0.7976   0.6310   0.7806   0.9586   0.8653   0.4965  
0.35       0.7976   0.6310   0.7806   0.9586   0.8653   0.4965  
0.40       0.7976   0.6310   0.7806   0.9586   0.8653   0.4965  
0.45       0.7976   0.6310   0.7806   0.9586   0.8653   0.4965  
0.50       0.7976   0.6310   0.7806   0.9586   0.8653   0.4965  
0.55       0.7976   0.6310   0.7806   0.9586   0.8653   0.4965  
0.60       0.7976   0.6310   0.7806   0.9586   0.8653   0.4965  
0.65       0.7976   0.6310   0.7806   0.9586   0.8653   0.4965  
0.70       0.7976   0.6310   0.7806   0.9586   0.8653   0.4965  
0.75       0.7976   0.6310   0.7806   0.9586   0.8653   0.4965  
0.80       0.7976   0.6310   0.7806   0.9586   0.8653   0.4965  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7976, F1=0.6310, Normal Recall=0.7806, Normal Precision=0.9586, Attack Recall=0.8653, Attack Precision=0.4965

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
0.15       0.8065   0.7285   0.7813   0.9312   0.8653   0.6291   <--
0.20       0.8065   0.7285   0.7813   0.9312   0.8653   0.6291  
0.25       0.8065   0.7285   0.7813   0.9312   0.8653   0.6291  
0.30       0.8065   0.7285   0.7813   0.9312   0.8653   0.6291  
0.35       0.8065   0.7285   0.7813   0.9312   0.8653   0.6291  
0.40       0.8065   0.7285   0.7813   0.9312   0.8653   0.6291  
0.45       0.8065   0.7285   0.7813   0.9312   0.8653   0.6291  
0.50       0.8065   0.7285   0.7813   0.9312   0.8653   0.6291  
0.55       0.8065   0.7285   0.7813   0.9312   0.8653   0.6291  
0.60       0.8065   0.7285   0.7813   0.9312   0.8653   0.6291  
0.65       0.8065   0.7285   0.7813   0.9312   0.8653   0.6291  
0.70       0.8065   0.7285   0.7813   0.9312   0.8653   0.6291  
0.75       0.8065   0.7285   0.7813   0.9312   0.8653   0.6291  
0.80       0.8065   0.7285   0.7813   0.9312   0.8653   0.6291  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8065, F1=0.7285, Normal Recall=0.7813, Normal Precision=0.9312, Attack Recall=0.8653, Attack Precision=0.6291

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
0.15       0.8159   0.7899   0.7830   0.8971   0.8653   0.7266   <--
0.20       0.8159   0.7899   0.7830   0.8971   0.8653   0.7266  
0.25       0.8159   0.7899   0.7830   0.8971   0.8653   0.7266  
0.30       0.8159   0.7899   0.7830   0.8971   0.8653   0.7266  
0.35       0.8159   0.7899   0.7830   0.8971   0.8653   0.7266  
0.40       0.8159   0.7899   0.7830   0.8971   0.8653   0.7266  
0.45       0.8159   0.7899   0.7830   0.8971   0.8653   0.7266  
0.50       0.8159   0.7899   0.7830   0.8971   0.8653   0.7266  
0.55       0.8159   0.7899   0.7830   0.8971   0.8653   0.7266  
0.60       0.8159   0.7899   0.7830   0.8971   0.8653   0.7266  
0.65       0.8159   0.7899   0.7830   0.8971   0.8653   0.7266  
0.70       0.8159   0.7899   0.7830   0.8971   0.8653   0.7266  
0.75       0.8159   0.7899   0.7830   0.8971   0.8653   0.7266  
0.80       0.8159   0.7899   0.7830   0.8971   0.8653   0.7266  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8159, F1=0.7899, Normal Recall=0.7830, Normal Precision=0.8971, Attack Recall=0.8653, Attack Precision=0.7266

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
0.15       0.8245   0.8314   0.7836   0.8533   0.8653   0.8000   <--
0.20       0.8245   0.8314   0.7836   0.8533   0.8653   0.8000  
0.25       0.8245   0.8314   0.7836   0.8533   0.8653   0.8000  
0.30       0.8245   0.8314   0.7836   0.8533   0.8653   0.8000  
0.35       0.8245   0.8314   0.7836   0.8533   0.8653   0.8000  
0.40       0.8245   0.8314   0.7836   0.8533   0.8653   0.8000  
0.45       0.8245   0.8314   0.7836   0.8533   0.8653   0.8000  
0.50       0.8245   0.8314   0.7836   0.8533   0.8653   0.8000  
0.55       0.8245   0.8314   0.7836   0.8533   0.8653   0.8000  
0.60       0.8245   0.8314   0.7836   0.8533   0.8653   0.8000  
0.65       0.8245   0.8314   0.7836   0.8533   0.8653   0.8000  
0.70       0.8245   0.8314   0.7836   0.8533   0.8653   0.8000  
0.75       0.8245   0.8314   0.7836   0.8533   0.8653   0.8000  
0.80       0.8245   0.8314   0.7836   0.8533   0.8653   0.8000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8245, F1=0.8314, Normal Recall=0.7836, Normal Precision=0.8533, Attack Recall=0.8653, Attack Precision=0.8000

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
0.15       0.7582   0.4508   0.7321   0.9989   0.9925   0.2916   <--
0.20       0.7582   0.4508   0.7321   0.9989   0.9925   0.2916  
0.25       0.7582   0.4508   0.7321   0.9989   0.9925   0.2916  
0.30       0.7582   0.4508   0.7321   0.9989   0.9925   0.2916  
0.35       0.7582   0.4508   0.7321   0.9989   0.9925   0.2916  
0.40       0.7582   0.4508   0.7321   0.9989   0.9925   0.2916  
0.45       0.7582   0.4508   0.7321   0.9989   0.9925   0.2916  
0.50       0.7582   0.4508   0.7321   0.9989   0.9925   0.2916  
0.55       0.7582   0.4508   0.7321   0.9989   0.9925   0.2916  
0.60       0.7582   0.4508   0.7321   0.9989   0.9925   0.2916  
0.65       0.7582   0.4508   0.7321   0.9989   0.9925   0.2916  
0.70       0.7582   0.4508   0.7321   0.9989   0.9925   0.2916  
0.75       0.7582   0.4508   0.7321   0.9989   0.9925   0.2916  
0.80       0.7582   0.4508   0.7321   0.9989   0.9925   0.2916  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7582, F1=0.4508, Normal Recall=0.7321, Normal Precision=0.9989, Attack Recall=0.9925, Attack Precision=0.2916

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
0.15       0.7843   0.6478   0.7324   0.9972   0.9918   0.4810   <--
0.20       0.7843   0.6478   0.7324   0.9972   0.9918   0.4810  
0.25       0.7843   0.6478   0.7324   0.9972   0.9918   0.4810  
0.30       0.7843   0.6478   0.7324   0.9972   0.9918   0.4810  
0.35       0.7843   0.6478   0.7324   0.9972   0.9918   0.4810  
0.40       0.7843   0.6478   0.7324   0.9972   0.9918   0.4810  
0.45       0.7843   0.6478   0.7324   0.9972   0.9918   0.4810  
0.50       0.7843   0.6478   0.7324   0.9972   0.9918   0.4810  
0.55       0.7843   0.6478   0.7324   0.9972   0.9918   0.4810  
0.60       0.7843   0.6478   0.7324   0.9972   0.9918   0.4810  
0.65       0.7843   0.6478   0.7324   0.9972   0.9918   0.4810  
0.70       0.7843   0.6478   0.7324   0.9972   0.9918   0.4810  
0.75       0.7843   0.6478   0.7324   0.9972   0.9918   0.4810  
0.80       0.7843   0.6478   0.7324   0.9972   0.9918   0.4810  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7843, F1=0.6478, Normal Recall=0.7324, Normal Precision=0.9972, Attack Recall=0.9918, Attack Precision=0.4810

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
0.15       0.8096   0.7576   0.7315   0.9952   0.9918   0.6129   <--
0.20       0.8096   0.7576   0.7315   0.9952   0.9918   0.6129  
0.25       0.8096   0.7576   0.7315   0.9952   0.9918   0.6129  
0.30       0.8096   0.7576   0.7315   0.9952   0.9918   0.6129  
0.35       0.8096   0.7576   0.7315   0.9952   0.9918   0.6129  
0.40       0.8096   0.7576   0.7315   0.9952   0.9918   0.6129  
0.45       0.8096   0.7576   0.7315   0.9952   0.9918   0.6129  
0.50       0.8096   0.7576   0.7315   0.9952   0.9918   0.6129  
0.55       0.8096   0.7576   0.7315   0.9952   0.9918   0.6129  
0.60       0.8096   0.7576   0.7315   0.9952   0.9918   0.6129  
0.65       0.8096   0.7576   0.7315   0.9952   0.9918   0.6129  
0.70       0.8096   0.7576   0.7315   0.9952   0.9918   0.6129  
0.75       0.8096   0.7576   0.7315   0.9952   0.9918   0.6129  
0.80       0.8096   0.7576   0.7315   0.9952   0.9918   0.6129  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8096, F1=0.7576, Normal Recall=0.7315, Normal Precision=0.9952, Attack Recall=0.9918, Attack Precision=0.6129

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
0.15       0.8352   0.8280   0.7307   0.9926   0.9918   0.7106   <--
0.20       0.8352   0.8280   0.7307   0.9926   0.9918   0.7106  
0.25       0.8352   0.8280   0.7307   0.9926   0.9918   0.7106  
0.30       0.8352   0.8280   0.7307   0.9926   0.9918   0.7106  
0.35       0.8352   0.8280   0.7307   0.9926   0.9918   0.7106  
0.40       0.8352   0.8280   0.7307   0.9926   0.9918   0.7106  
0.45       0.8352   0.8280   0.7307   0.9926   0.9918   0.7106  
0.50       0.8352   0.8280   0.7307   0.9926   0.9918   0.7106  
0.55       0.8352   0.8280   0.7307   0.9926   0.9918   0.7106  
0.60       0.8352   0.8280   0.7307   0.9926   0.9918   0.7106  
0.65       0.8352   0.8280   0.7307   0.9926   0.9918   0.7106  
0.70       0.8352   0.8280   0.7307   0.9926   0.9918   0.7106  
0.75       0.8352   0.8280   0.7307   0.9926   0.9918   0.7106  
0.80       0.8352   0.8280   0.7307   0.9926   0.9918   0.7106  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8352, F1=0.8280, Normal Recall=0.7307, Normal Precision=0.9926, Attack Recall=0.9918, Attack Precision=0.7106

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
0.15       0.8605   0.8767   0.7291   0.9889   0.9918   0.7855   <--
0.20       0.8605   0.8767   0.7291   0.9889   0.9918   0.7855  
0.25       0.8605   0.8767   0.7291   0.9889   0.9918   0.7855  
0.30       0.8605   0.8767   0.7291   0.9889   0.9918   0.7855  
0.35       0.8605   0.8767   0.7291   0.9889   0.9918   0.7855  
0.40       0.8605   0.8767   0.7291   0.9889   0.9918   0.7855  
0.45       0.8605   0.8767   0.7291   0.9889   0.9918   0.7855  
0.50       0.8605   0.8767   0.7291   0.9889   0.9918   0.7855  
0.55       0.8605   0.8767   0.7291   0.9889   0.9918   0.7855  
0.60       0.8605   0.8767   0.7291   0.9889   0.9918   0.7855  
0.65       0.8605   0.8767   0.7291   0.9889   0.9918   0.7855  
0.70       0.8605   0.8767   0.7291   0.9889   0.9918   0.7855  
0.75       0.8605   0.8767   0.7291   0.9889   0.9918   0.7855  
0.80       0.8605   0.8767   0.7291   0.9889   0.9918   0.7855  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8605, F1=0.8767, Normal Recall=0.7291, Normal Precision=0.9889, Attack Recall=0.9918, Attack Precision=0.7855

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
0.15       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905   <--
0.20       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.25       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.30       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.35       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.40       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.45       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.50       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.55       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.60       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.65       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.70       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.75       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.80       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7569, F1=0.4495, Normal Recall=0.7307, Normal Precision=0.9989, Attack Recall=0.9924, Attack Precision=0.2905

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
0.15       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796   <--
0.20       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.25       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.30       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.35       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.40       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.45       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.50       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.55       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.60       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.65       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.70       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.75       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.80       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7832, F1=0.6466, Normal Recall=0.7310, Normal Precision=0.9972, Attack Recall=0.9918, Attack Precision=0.4796

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
0.15       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116   <--
0.20       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.25       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.30       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.35       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.40       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.45       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.50       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.55       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.60       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.65       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.70       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.75       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.80       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8085, F1=0.7566, Normal Recall=0.7300, Normal Precision=0.9952, Attack Recall=0.9918, Attack Precision=0.6116

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
0.15       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094   <--
0.20       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.25       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.30       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.35       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.40       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.45       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.50       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.55       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.60       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.65       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.70       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.75       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.80       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8342, F1=0.8272, Normal Recall=0.7292, Normal Precision=0.9925, Attack Recall=0.9918, Attack Precision=0.7094

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
0.15       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845   <--
0.20       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.25       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.30       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.35       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.40       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.45       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.50       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.55       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.60       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.65       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.70       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.75       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.80       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8597, F1=0.8760, Normal Recall=0.7275, Normal Precision=0.9888, Attack Recall=0.9918, Attack Precision=0.7845

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
0.15       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905   <--
0.20       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.25       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.30       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.35       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.40       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.45       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.50       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.55       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.60       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.65       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.70       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.75       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
0.80       0.7569   0.4495   0.7307   0.9989   0.9924   0.2905  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7569, F1=0.4495, Normal Recall=0.7307, Normal Precision=0.9989, Attack Recall=0.9924, Attack Precision=0.2905

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
0.15       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796   <--
0.20       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.25       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.30       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.35       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.40       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.45       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.50       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.55       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.60       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.65       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.70       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.75       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
0.80       0.7832   0.6466   0.7310   0.9972   0.9918   0.4796  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7832, F1=0.6466, Normal Recall=0.7310, Normal Precision=0.9972, Attack Recall=0.9918, Attack Precision=0.4796

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
0.15       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116   <--
0.20       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.25       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.30       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.35       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.40       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.45       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.50       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.55       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.60       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.65       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.70       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.75       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
0.80       0.8085   0.7566   0.7300   0.9952   0.9918   0.6116  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8085, F1=0.7566, Normal Recall=0.7300, Normal Precision=0.9952, Attack Recall=0.9918, Attack Precision=0.6116

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
0.15       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094   <--
0.20       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.25       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.30       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.35       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.40       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.45       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.50       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.55       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.60       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.65       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.70       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.75       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
0.80       0.8342   0.8272   0.7292   0.9925   0.9918   0.7094  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8342, F1=0.8272, Normal Recall=0.7292, Normal Precision=0.9925, Attack Recall=0.9918, Attack Precision=0.7094

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
0.15       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845   <--
0.20       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.25       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.30       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.35       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.40       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.45       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.50       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.55       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.60       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.65       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.70       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.75       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
0.80       0.8597   0.8760   0.7275   0.9888   0.9918   0.7845  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8597, F1=0.8760, Normal Recall=0.7275, Normal Precision=0.9888, Attack Recall=0.9918, Attack Precision=0.7845

```

