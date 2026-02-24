# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e2_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-23 00:08:47 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2398 | 0.3151 | 0.3907 | 0.4673 | 0.5432 | 0.6185 | 0.6951 | 0.7704 | 0.8463 | 0.9226 | 0.9986 |
| QAT+Prune only | 0.8150 | 0.8336 | 0.8511 | 0.8700 | 0.8875 | 0.9037 | 0.9229 | 0.9408 | 0.9591 | 0.9765 | 0.9948 |
| QAT+PTQ | 0.8129 | 0.8319 | 0.8496 | 0.8687 | 0.8863 | 0.9027 | 0.9223 | 0.9401 | 0.9588 | 0.9762 | 0.9948 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8129 | 0.8319 | 0.8496 | 0.8687 | 0.8863 | 0.9027 | 0.9223 | 0.9401 | 0.9588 | 0.9762 | 0.9948 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2257 | 0.3960 | 0.5294 | 0.6362 | 0.7236 | 0.7972 | 0.8590 | 0.9123 | 0.9587 | 0.9993 |
| QAT+Prune only | 0.0000 | 0.5446 | 0.7277 | 0.8212 | 0.8762 | 0.9117 | 0.9394 | 0.9592 | 0.9749 | 0.9870 | 0.9974 |
| QAT+PTQ | 0.0000 | 0.5420 | 0.7258 | 0.8197 | 0.8750 | 0.9109 | 0.9389 | 0.9588 | 0.9748 | 0.9869 | 0.9974 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5420 | 0.7258 | 0.8197 | 0.8750 | 0.9109 | 0.9389 | 0.9588 | 0.9748 | 0.9869 | 0.9974 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2398 | 0.2392 | 0.2387 | 0.2396 | 0.2395 | 0.2384 | 0.2398 | 0.2381 | 0.2372 | 0.2390 | 0.0000 |
| QAT+Prune only | 0.8150 | 0.8157 | 0.8152 | 0.8165 | 0.8160 | 0.8126 | 0.8152 | 0.8149 | 0.8163 | 0.8117 | 0.0000 |
| QAT+PTQ | 0.8129 | 0.8138 | 0.8134 | 0.8147 | 0.8139 | 0.8106 | 0.8135 | 0.8126 | 0.8146 | 0.8088 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8129 | 0.8138 | 0.8134 | 0.8147 | 0.8139 | 0.8106 | 0.8135 | 0.8126 | 0.8146 | 0.8088 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2398 | 0.0000 | 0.0000 | 0.0000 | 0.2398 | 1.0000 |
| 90 | 10 | 299,940 | 0.3151 | 0.1272 | 0.9984 | 0.2257 | 0.2392 | 0.9992 |
| 80 | 20 | 291,350 | 0.3907 | 0.2470 | 0.9986 | 0.3960 | 0.2387 | 0.9985 |
| 70 | 30 | 194,230 | 0.4673 | 0.3601 | 0.9986 | 0.5294 | 0.2396 | 0.9975 |
| 60 | 40 | 145,675 | 0.5432 | 0.4668 | 0.9986 | 0.6362 | 0.2395 | 0.9961 |
| 50 | 50 | 116,540 | 0.6185 | 0.5673 | 0.9986 | 0.7236 | 0.2384 | 0.9942 |
| 40 | 60 | 97,115 | 0.6951 | 0.6633 | 0.9986 | 0.7972 | 0.2398 | 0.9914 |
| 30 | 70 | 83,240 | 0.7704 | 0.7536 | 0.9986 | 0.8590 | 0.2381 | 0.9866 |
| 20 | 80 | 72,835 | 0.8463 | 0.8397 | 0.9986 | 0.9123 | 0.2372 | 0.9771 |
| 10 | 90 | 64,740 | 0.9226 | 0.9219 | 0.9986 | 0.9587 | 0.2390 | 0.9502 |
| 0 | 100 | 58,270 | 0.9986 | 1.0000 | 0.9986 | 0.9993 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8150 | 0.0000 | 0.0000 | 0.0000 | 0.8150 | 1.0000 |
| 90 | 10 | 299,940 | 0.8336 | 0.3749 | 0.9948 | 0.5446 | 0.8157 | 0.9993 |
| 80 | 20 | 291,350 | 0.8511 | 0.5737 | 0.9948 | 0.7277 | 0.8152 | 0.9984 |
| 70 | 30 | 194,230 | 0.8700 | 0.6991 | 0.9948 | 0.8212 | 0.8165 | 0.9973 |
| 60 | 40 | 145,675 | 0.8875 | 0.7828 | 0.9948 | 0.8762 | 0.8160 | 0.9957 |
| 50 | 50 | 116,540 | 0.9037 | 0.8415 | 0.9948 | 0.9117 | 0.8126 | 0.9936 |
| 40 | 60 | 97,115 | 0.9229 | 0.8898 | 0.9948 | 0.9394 | 0.8152 | 0.9905 |
| 30 | 70 | 83,240 | 0.9408 | 0.9261 | 0.9948 | 0.9592 | 0.8149 | 0.9852 |
| 20 | 80 | 72,835 | 0.9591 | 0.9559 | 0.9948 | 0.9749 | 0.8163 | 0.9750 |
| 10 | 90 | 64,740 | 0.9765 | 0.9794 | 0.9948 | 0.9870 | 0.8117 | 0.9451 |
| 0 | 100 | 58,270 | 0.9948 | 1.0000 | 0.9948 | 0.9974 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8129 | 0.0000 | 0.0000 | 0.0000 | 0.8129 | 1.0000 |
| 90 | 10 | 299,940 | 0.8319 | 0.3725 | 0.9949 | 0.5420 | 0.8138 | 0.9993 |
| 80 | 20 | 291,350 | 0.8496 | 0.5713 | 0.9948 | 0.7258 | 0.8134 | 0.9984 |
| 70 | 30 | 194,230 | 0.8687 | 0.6970 | 0.9948 | 0.8197 | 0.8147 | 0.9973 |
| 60 | 40 | 145,675 | 0.8863 | 0.7809 | 0.9948 | 0.8750 | 0.8139 | 0.9958 |
| 50 | 50 | 116,540 | 0.9027 | 0.8401 | 0.9948 | 0.9109 | 0.8106 | 0.9937 |
| 40 | 60 | 97,115 | 0.9223 | 0.8889 | 0.9948 | 0.9389 | 0.8135 | 0.9906 |
| 30 | 70 | 83,240 | 0.9401 | 0.9253 | 0.9948 | 0.9588 | 0.8126 | 0.9854 |
| 20 | 80 | 72,835 | 0.9588 | 0.9555 | 0.9948 | 0.9748 | 0.8146 | 0.9753 |
| 10 | 90 | 64,740 | 0.9762 | 0.9791 | 0.9948 | 0.9869 | 0.8088 | 0.9456 |
| 0 | 100 | 58,270 | 0.9948 | 1.0000 | 0.9948 | 0.9974 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8129 | 0.0000 | 0.0000 | 0.0000 | 0.8129 | 1.0000 |
| 90 | 10 | 299,940 | 0.8319 | 0.3725 | 0.9949 | 0.5420 | 0.8138 | 0.9993 |
| 80 | 20 | 291,350 | 0.8496 | 0.5713 | 0.9948 | 0.7258 | 0.8134 | 0.9984 |
| 70 | 30 | 194,230 | 0.8687 | 0.6970 | 0.9948 | 0.8197 | 0.8147 | 0.9973 |
| 60 | 40 | 145,675 | 0.8863 | 0.7809 | 0.9948 | 0.8750 | 0.8139 | 0.9958 |
| 50 | 50 | 116,540 | 0.9027 | 0.8401 | 0.9948 | 0.9109 | 0.8106 | 0.9937 |
| 40 | 60 | 97,115 | 0.9223 | 0.8889 | 0.9948 | 0.9389 | 0.8135 | 0.9906 |
| 30 | 70 | 83,240 | 0.9401 | 0.9253 | 0.9948 | 0.9588 | 0.8126 | 0.9854 |
| 20 | 80 | 72,835 | 0.9588 | 0.9555 | 0.9948 | 0.9748 | 0.8146 | 0.9753 |
| 10 | 90 | 64,740 | 0.9762 | 0.9791 | 0.9948 | 0.9869 | 0.8088 | 0.9456 |
| 0 | 100 | 58,270 | 0.9948 | 1.0000 | 0.9948 | 0.9974 | 0.0000 | 0.0000 |


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
0.15       0.3151   0.2258   0.2392   0.9994   0.9988   0.1273   <--
0.20       0.3151   0.2258   0.2392   0.9994   0.9988   0.1273  
0.25       0.3151   0.2258   0.2392   0.9994   0.9988   0.1273  
0.30       0.3151   0.2258   0.2392   0.9994   0.9988   0.1273  
0.35       0.3151   0.2258   0.2392   0.9994   0.9988   0.1273  
0.40       0.3151   0.2258   0.2392   0.9994   0.9988   0.1273  
0.45       0.3151   0.2258   0.2392   0.9994   0.9988   0.1273  
0.50       0.3151   0.2258   0.2392   0.9994   0.9988   0.1273  
0.55       0.3151   0.2258   0.2392   0.9994   0.9988   0.1273  
0.60       0.3151   0.2258   0.2392   0.9994   0.9988   0.1273  
0.65       0.3151   0.2258   0.2392   0.9994   0.9988   0.1273  
0.70       0.3151   0.2258   0.2392   0.9994   0.9988   0.1273  
0.75       0.3151   0.2258   0.2392   0.9994   0.9988   0.1273  
0.80       0.3151   0.2258   0.2392   0.9994   0.9988   0.1273  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3151, F1=0.2258, Normal Recall=0.2392, Normal Precision=0.9994, Attack Recall=0.9988, Attack Precision=0.1273

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
0.15       0.3913   0.3962   0.2395   0.9986   0.9986   0.2471   <--
0.20       0.3913   0.3962   0.2395   0.9986   0.9986   0.2471  
0.25       0.3913   0.3962   0.2395   0.9986   0.9986   0.2471  
0.30       0.3913   0.3962   0.2395   0.9986   0.9986   0.2471  
0.35       0.3913   0.3962   0.2395   0.9986   0.9986   0.2471  
0.40       0.3913   0.3962   0.2395   0.9986   0.9986   0.2471  
0.45       0.3913   0.3962   0.2395   0.9986   0.9986   0.2471  
0.50       0.3913   0.3962   0.2395   0.9986   0.9986   0.2471  
0.55       0.3913   0.3962   0.2395   0.9986   0.9986   0.2471  
0.60       0.3913   0.3962   0.2395   0.9986   0.9986   0.2471  
0.65       0.3913   0.3962   0.2395   0.9986   0.9986   0.2471  
0.70       0.3913   0.3962   0.2395   0.9986   0.9986   0.2471  
0.75       0.3913   0.3962   0.2395   0.9986   0.9986   0.2471  
0.80       0.3913   0.3962   0.2395   0.9986   0.9986   0.2471  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3913, F1=0.3962, Normal Recall=0.2395, Normal Precision=0.9986, Attack Recall=0.9986, Attack Precision=0.2471

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
0.15       0.4680   0.5297   0.2405   0.9975   0.9986   0.3604   <--
0.20       0.4680   0.5297   0.2405   0.9975   0.9986   0.3604  
0.25       0.4680   0.5297   0.2405   0.9975   0.9986   0.3604  
0.30       0.4680   0.5297   0.2405   0.9975   0.9986   0.3604  
0.35       0.4680   0.5297   0.2405   0.9975   0.9986   0.3604  
0.40       0.4680   0.5297   0.2405   0.9975   0.9986   0.3604  
0.45       0.4680   0.5297   0.2405   0.9975   0.9986   0.3604  
0.50       0.4680   0.5297   0.2405   0.9975   0.9986   0.3604  
0.55       0.4680   0.5297   0.2405   0.9975   0.9986   0.3604  
0.60       0.4680   0.5297   0.2405   0.9975   0.9986   0.3604  
0.65       0.4680   0.5297   0.2405   0.9975   0.9986   0.3604  
0.70       0.4680   0.5297   0.2405   0.9975   0.9986   0.3604  
0.75       0.4680   0.5297   0.2405   0.9975   0.9986   0.3604  
0.80       0.4680   0.5297   0.2405   0.9975   0.9986   0.3604  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4680, F1=0.5297, Normal Recall=0.2405, Normal Precision=0.9975, Attack Recall=0.9986, Attack Precision=0.3604

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
0.15       0.5429   0.6361   0.2391   0.9961   0.9986   0.4667   <--
0.20       0.5429   0.6361   0.2391   0.9961   0.9986   0.4667  
0.25       0.5429   0.6361   0.2391   0.9961   0.9986   0.4667  
0.30       0.5429   0.6361   0.2391   0.9961   0.9986   0.4667  
0.35       0.5429   0.6361   0.2391   0.9961   0.9986   0.4667  
0.40       0.5429   0.6361   0.2391   0.9961   0.9986   0.4667  
0.45       0.5429   0.6361   0.2391   0.9961   0.9986   0.4667  
0.50       0.5429   0.6361   0.2391   0.9961   0.9986   0.4667  
0.55       0.5429   0.6361   0.2391   0.9961   0.9986   0.4667  
0.60       0.5429   0.6361   0.2391   0.9961   0.9986   0.4667  
0.65       0.5429   0.6361   0.2391   0.9961   0.9986   0.4667  
0.70       0.5429   0.6361   0.2391   0.9961   0.9986   0.4667  
0.75       0.5429   0.6361   0.2391   0.9961   0.9986   0.4667  
0.80       0.5429   0.6361   0.2391   0.9961   0.9986   0.4667  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5429, F1=0.6361, Normal Recall=0.2391, Normal Precision=0.9961, Attack Recall=0.9986, Attack Precision=0.4667

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
0.15       0.6193   0.7240   0.2400   0.9942   0.9986   0.5678   <--
0.20       0.6193   0.7240   0.2400   0.9942   0.9986   0.5678  
0.25       0.6193   0.7240   0.2400   0.9942   0.9986   0.5678  
0.30       0.6193   0.7240   0.2400   0.9942   0.9986   0.5678  
0.35       0.6193   0.7240   0.2400   0.9942   0.9986   0.5678  
0.40       0.6193   0.7240   0.2400   0.9942   0.9986   0.5678  
0.45       0.6193   0.7240   0.2400   0.9942   0.9986   0.5678  
0.50       0.6193   0.7240   0.2400   0.9942   0.9986   0.5678  
0.55       0.6193   0.7240   0.2400   0.9942   0.9986   0.5678  
0.60       0.6193   0.7240   0.2400   0.9942   0.9986   0.5678  
0.65       0.6193   0.7240   0.2400   0.9942   0.9986   0.5678  
0.70       0.6193   0.7240   0.2400   0.9942   0.9986   0.5678  
0.75       0.6193   0.7240   0.2400   0.9942   0.9986   0.5678  
0.80       0.6193   0.7240   0.2400   0.9942   0.9986   0.5678  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6193, F1=0.7240, Normal Recall=0.2400, Normal Precision=0.9942, Attack Recall=0.9986, Attack Precision=0.5678

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
0.15       0.8337   0.5447   0.8157   0.9993   0.9950   0.3750   <--
0.20       0.8337   0.5447   0.8157   0.9993   0.9950   0.3750  
0.25       0.8337   0.5447   0.8157   0.9993   0.9950   0.3750  
0.30       0.8337   0.5447   0.8157   0.9993   0.9950   0.3750  
0.35       0.8337   0.5447   0.8157   0.9993   0.9950   0.3750  
0.40       0.8337   0.5447   0.8157   0.9993   0.9950   0.3750  
0.45       0.8337   0.5447   0.8157   0.9993   0.9950   0.3750  
0.50       0.8337   0.5447   0.8157   0.9993   0.9950   0.3750  
0.55       0.8337   0.5447   0.8157   0.9993   0.9950   0.3750  
0.60       0.8337   0.5447   0.8157   0.9993   0.9950   0.3750  
0.65       0.8337   0.5447   0.8157   0.9993   0.9950   0.3750  
0.70       0.8337   0.5447   0.8157   0.9993   0.9950   0.3750  
0.75       0.8337   0.5447   0.8157   0.9993   0.9950   0.3750  
0.80       0.8337   0.5447   0.8157   0.9993   0.9950   0.3750  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8337, F1=0.5447, Normal Recall=0.8157, Normal Precision=0.9993, Attack Recall=0.9950, Attack Precision=0.3750

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
0.15       0.8519   0.7287   0.8161   0.9984   0.9948   0.5749   <--
0.20       0.8519   0.7287   0.8161   0.9984   0.9948   0.5749  
0.25       0.8519   0.7287   0.8161   0.9984   0.9948   0.5749  
0.30       0.8519   0.7287   0.8161   0.9984   0.9948   0.5749  
0.35       0.8519   0.7287   0.8161   0.9984   0.9948   0.5749  
0.40       0.8519   0.7287   0.8161   0.9984   0.9948   0.5749  
0.45       0.8519   0.7287   0.8161   0.9984   0.9948   0.5749  
0.50       0.8519   0.7287   0.8161   0.9984   0.9948   0.5749  
0.55       0.8519   0.7287   0.8161   0.9984   0.9948   0.5749  
0.60       0.8519   0.7287   0.8161   0.9984   0.9948   0.5749  
0.65       0.8519   0.7287   0.8161   0.9984   0.9948   0.5749  
0.70       0.8519   0.7287   0.8161   0.9984   0.9948   0.5749  
0.75       0.8519   0.7287   0.8161   0.9984   0.9948   0.5749  
0.80       0.8519   0.7287   0.8161   0.9984   0.9948   0.5749  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8519, F1=0.7287, Normal Recall=0.8161, Normal Precision=0.9984, Attack Recall=0.9948, Attack Precision=0.5749

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
0.15       0.8694   0.8205   0.8157   0.9973   0.9948   0.6982   <--
0.20       0.8694   0.8205   0.8157   0.9973   0.9948   0.6982  
0.25       0.8694   0.8205   0.8157   0.9973   0.9948   0.6982  
0.30       0.8694   0.8205   0.8157   0.9973   0.9948   0.6982  
0.35       0.8694   0.8205   0.8157   0.9973   0.9948   0.6982  
0.40       0.8694   0.8205   0.8157   0.9973   0.9948   0.6982  
0.45       0.8694   0.8205   0.8157   0.9973   0.9948   0.6982  
0.50       0.8694   0.8205   0.8157   0.9973   0.9948   0.6982  
0.55       0.8694   0.8205   0.8157   0.9973   0.9948   0.6982  
0.60       0.8694   0.8205   0.8157   0.9973   0.9948   0.6982  
0.65       0.8694   0.8205   0.8157   0.9973   0.9948   0.6982  
0.70       0.8694   0.8205   0.8157   0.9973   0.9948   0.6982  
0.75       0.8694   0.8205   0.8157   0.9973   0.9948   0.6982  
0.80       0.8694   0.8205   0.8157   0.9973   0.9948   0.6982  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8694, F1=0.8205, Normal Recall=0.8157, Normal Precision=0.9973, Attack Recall=0.9948, Attack Precision=0.6982

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
0.15       0.8869   0.8756   0.8150   0.9957   0.9948   0.7819   <--
0.20       0.8869   0.8756   0.8150   0.9957   0.9948   0.7819  
0.25       0.8869   0.8756   0.8150   0.9957   0.9948   0.7819  
0.30       0.8869   0.8756   0.8150   0.9957   0.9948   0.7819  
0.35       0.8869   0.8756   0.8150   0.9957   0.9948   0.7819  
0.40       0.8869   0.8756   0.8150   0.9957   0.9948   0.7819  
0.45       0.8869   0.8756   0.8150   0.9957   0.9948   0.7819  
0.50       0.8869   0.8756   0.8150   0.9957   0.9948   0.7819  
0.55       0.8869   0.8756   0.8150   0.9957   0.9948   0.7819  
0.60       0.8869   0.8756   0.8150   0.9957   0.9948   0.7819  
0.65       0.8869   0.8756   0.8150   0.9957   0.9948   0.7819  
0.70       0.8869   0.8756   0.8150   0.9957   0.9948   0.7819  
0.75       0.8869   0.8756   0.8150   0.9957   0.9948   0.7819  
0.80       0.8869   0.8756   0.8150   0.9957   0.9948   0.7819  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8869, F1=0.8756, Normal Recall=0.8150, Normal Precision=0.9957, Attack Recall=0.9948, Attack Precision=0.7819

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
0.15       0.9042   0.9121   0.8136   0.9936   0.9948   0.8422   <--
0.20       0.9042   0.9121   0.8136   0.9936   0.9948   0.8422  
0.25       0.9042   0.9121   0.8136   0.9936   0.9948   0.8422  
0.30       0.9042   0.9121   0.8136   0.9936   0.9948   0.8422  
0.35       0.9042   0.9121   0.8136   0.9936   0.9948   0.8422  
0.40       0.9042   0.9121   0.8136   0.9936   0.9948   0.8422  
0.45       0.9042   0.9121   0.8136   0.9936   0.9948   0.8422  
0.50       0.9042   0.9121   0.8136   0.9936   0.9948   0.8422  
0.55       0.9042   0.9121   0.8136   0.9936   0.9948   0.8422  
0.60       0.9042   0.9121   0.8136   0.9936   0.9948   0.8422  
0.65       0.9042   0.9121   0.8136   0.9936   0.9948   0.8422  
0.70       0.9042   0.9121   0.8136   0.9936   0.9948   0.8422  
0.75       0.9042   0.9121   0.8136   0.9936   0.9948   0.8422  
0.80       0.9042   0.9121   0.8136   0.9936   0.9948   0.8422  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9042, F1=0.9121, Normal Recall=0.8136, Normal Precision=0.9936, Attack Recall=0.9948, Attack Precision=0.8422

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
0.15       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725   <--
0.20       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.25       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.30       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.35       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.40       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.45       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.50       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.55       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.60       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.65       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.70       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.75       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.80       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8319, F1=0.5421, Normal Recall=0.8138, Normal Precision=0.9993, Attack Recall=0.9950, Attack Precision=0.3725

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
0.15       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724   <--
0.20       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.25       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.30       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.35       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.40       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.45       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.50       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.55       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.60       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.65       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.70       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.75       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.80       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8503, F1=0.7267, Normal Recall=0.8142, Normal Precision=0.9984, Attack Recall=0.9948, Attack Precision=0.5724

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
0.15       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958   <--
0.20       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.25       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.30       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.35       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.40       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.45       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.50       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.55       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.60       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.65       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.70       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.75       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.80       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8680, F1=0.8189, Normal Recall=0.8136, Normal Precision=0.9973, Attack Recall=0.9948, Attack Precision=0.6958

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
0.15       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801   <--
0.20       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.25       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.30       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.35       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.40       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.45       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.50       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.55       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.60       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.65       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.70       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.75       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.80       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8857, F1=0.8745, Normal Recall=0.8130, Normal Precision=0.9958, Attack Recall=0.9948, Attack Precision=0.7801

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
0.15       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406   <--
0.20       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.25       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.30       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.35       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.40       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.45       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.50       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.55       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.60       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.65       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.70       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.75       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.80       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9031, F1=0.9113, Normal Recall=0.8114, Normal Precision=0.9937, Attack Recall=0.9948, Attack Precision=0.8406

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
0.15       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725   <--
0.20       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.25       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.30       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.35       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.40       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.45       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.50       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.55       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.60       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.65       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.70       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.75       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
0.80       0.8319   0.5421   0.8138   0.9993   0.9950   0.3725  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8319, F1=0.5421, Normal Recall=0.8138, Normal Precision=0.9993, Attack Recall=0.9950, Attack Precision=0.3725

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
0.15       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724   <--
0.20       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.25       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.30       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.35       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.40       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.45       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.50       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.55       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.60       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.65       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.70       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.75       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
0.80       0.8503   0.7267   0.8142   0.9984   0.9948   0.5724  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8503, F1=0.7267, Normal Recall=0.8142, Normal Precision=0.9984, Attack Recall=0.9948, Attack Precision=0.5724

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
0.15       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958   <--
0.20       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.25       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.30       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.35       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.40       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.45       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.50       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.55       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.60       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.65       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.70       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.75       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
0.80       0.8680   0.8189   0.8136   0.9973   0.9948   0.6958  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8680, F1=0.8189, Normal Recall=0.8136, Normal Precision=0.9973, Attack Recall=0.9948, Attack Precision=0.6958

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
0.15       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801   <--
0.20       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.25       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.30       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.35       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.40       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.45       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.50       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.55       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.60       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.65       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.70       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.75       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
0.80       0.8857   0.8745   0.8130   0.9958   0.9948   0.7801  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8857, F1=0.8745, Normal Recall=0.8130, Normal Precision=0.9958, Attack Recall=0.9948, Attack Precision=0.7801

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
0.15       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406   <--
0.20       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.25       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.30       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.35       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.40       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.45       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.50       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.55       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.60       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.65       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.70       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.75       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
0.80       0.9031   0.9113   0.8114   0.9937   0.9948   0.8406  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9031, F1=0.9113, Normal Recall=0.8114, Normal Precision=0.9937, Attack Recall=0.9948, Attack Precision=0.8406

```

