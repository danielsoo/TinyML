# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-12 09:25:19 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5258 | 0.5729 | 0.6195 | 0.6668 | 0.7132 | 0.7583 | 0.8062 | 0.8529 | 0.8982 | 0.9460 | 0.9922 |
| QAT+Prune only | 0.4237 | 0.4813 | 0.5387 | 0.5956 | 0.6541 | 0.7113 | 0.7678 | 0.8264 | 0.8838 | 0.9403 | 0.9985 |
| QAT+PTQ | 0.4224 | 0.4799 | 0.5375 | 0.5948 | 0.6533 | 0.7105 | 0.7672 | 0.8262 | 0.8832 | 0.9402 | 0.9986 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.4224 | 0.4799 | 0.5375 | 0.5948 | 0.6533 | 0.7105 | 0.7672 | 0.8262 | 0.8832 | 0.9402 | 0.9986 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3173 | 0.5105 | 0.6411 | 0.7346 | 0.8041 | 0.8600 | 0.9043 | 0.9398 | 0.9707 | 0.9961 |
| QAT+Prune only | 0.0000 | 0.2780 | 0.4641 | 0.5970 | 0.6978 | 0.7757 | 0.8377 | 0.8895 | 0.9322 | 0.9678 | 0.9993 |
| QAT+PTQ | 0.0000 | 0.2774 | 0.4634 | 0.5965 | 0.6974 | 0.7752 | 0.8373 | 0.8894 | 0.9319 | 0.9678 | 0.9993 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2774 | 0.4634 | 0.5965 | 0.6974 | 0.7752 | 0.8373 | 0.8894 | 0.9319 | 0.9678 | 0.9993 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5258 | 0.5262 | 0.5263 | 0.5273 | 0.5272 | 0.5245 | 0.5273 | 0.5280 | 0.5225 | 0.5304 | 0.0000 |
| QAT+Prune only | 0.4237 | 0.4238 | 0.4238 | 0.4230 | 0.4245 | 0.4240 | 0.4217 | 0.4246 | 0.4249 | 0.4158 | 0.0000 |
| QAT+PTQ | 0.4224 | 0.4222 | 0.4222 | 0.4217 | 0.4231 | 0.4224 | 0.4200 | 0.4239 | 0.4219 | 0.4152 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.4224 | 0.4222 | 0.4222 | 0.4217 | 0.4231 | 0.4224 | 0.4200 | 0.4239 | 0.4219 | 0.4152 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5258 | 0.0000 | 0.0000 | 0.0000 | 0.5258 | 1.0000 |
| 90 | 10 | 299,940 | 0.5729 | 0.1888 | 0.9925 | 0.3173 | 0.5262 | 0.9984 |
| 80 | 20 | 291,350 | 0.6195 | 0.3437 | 0.9922 | 0.5105 | 0.5263 | 0.9963 |
| 70 | 30 | 194,230 | 0.6668 | 0.4736 | 0.9922 | 0.6411 | 0.5273 | 0.9937 |
| 60 | 40 | 145,675 | 0.7132 | 0.5832 | 0.9922 | 0.7346 | 0.5272 | 0.9902 |
| 50 | 50 | 116,540 | 0.7583 | 0.6760 | 0.9922 | 0.8041 | 0.5245 | 0.9853 |
| 40 | 60 | 97,115 | 0.8062 | 0.7589 | 0.9922 | 0.8600 | 0.5273 | 0.9782 |
| 30 | 70 | 83,240 | 0.8529 | 0.8307 | 0.9922 | 0.9043 | 0.5280 | 0.9666 |
| 20 | 80 | 72,835 | 0.8982 | 0.8926 | 0.9922 | 0.9398 | 0.5225 | 0.9435 |
| 10 | 90 | 64,740 | 0.9460 | 0.9500 | 0.9922 | 0.9707 | 0.5304 | 0.8828 |
| 0 | 100 | 58,270 | 0.9922 | 1.0000 | 0.9922 | 0.9961 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4237 | 0.0000 | 0.0000 | 0.0000 | 0.4237 | 1.0000 |
| 90 | 10 | 299,940 | 0.4813 | 0.1615 | 0.9986 | 0.2780 | 0.4238 | 0.9996 |
| 80 | 20 | 291,350 | 0.5387 | 0.3023 | 0.9985 | 0.4641 | 0.4238 | 0.9991 |
| 70 | 30 | 194,230 | 0.5956 | 0.4258 | 0.9985 | 0.5970 | 0.4230 | 0.9985 |
| 60 | 40 | 145,675 | 0.6541 | 0.5363 | 0.9985 | 0.6978 | 0.4245 | 0.9977 |
| 50 | 50 | 116,540 | 0.7113 | 0.6342 | 0.9985 | 0.7757 | 0.4240 | 0.9966 |
| 40 | 60 | 97,115 | 0.7678 | 0.7215 | 0.9985 | 0.8377 | 0.4217 | 0.9948 |
| 30 | 70 | 83,240 | 0.8264 | 0.8019 | 0.9985 | 0.8895 | 0.4246 | 0.9920 |
| 20 | 80 | 72,835 | 0.8838 | 0.8741 | 0.9985 | 0.9322 | 0.4249 | 0.9865 |
| 10 | 90 | 64,740 | 0.9403 | 0.9390 | 0.9985 | 0.9678 | 0.4158 | 0.9694 |
| 0 | 100 | 58,270 | 0.9985 | 1.0000 | 0.9985 | 0.9993 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4224 | 0.0000 | 0.0000 | 0.0000 | 0.4224 | 1.0000 |
| 90 | 10 | 299,940 | 0.4799 | 0.1611 | 0.9986 | 0.2774 | 0.4222 | 0.9996 |
| 80 | 20 | 291,350 | 0.5375 | 0.3017 | 0.9986 | 0.4634 | 0.4222 | 0.9992 |
| 70 | 30 | 194,230 | 0.5948 | 0.4253 | 0.9986 | 0.5965 | 0.4217 | 0.9986 |
| 60 | 40 | 145,675 | 0.6533 | 0.5358 | 0.9986 | 0.6974 | 0.4231 | 0.9978 |
| 50 | 50 | 116,540 | 0.7105 | 0.6335 | 0.9986 | 0.7752 | 0.4224 | 0.9966 |
| 40 | 60 | 97,115 | 0.7672 | 0.7209 | 0.9986 | 0.8373 | 0.4200 | 0.9949 |
| 30 | 70 | 83,240 | 0.8262 | 0.8018 | 0.9986 | 0.8894 | 0.4239 | 0.9922 |
| 20 | 80 | 72,835 | 0.8832 | 0.8736 | 0.9986 | 0.9319 | 0.4219 | 0.9867 |
| 10 | 90 | 64,740 | 0.9402 | 0.9389 | 0.9986 | 0.9678 | 0.4152 | 0.9700 |
| 0 | 100 | 58,270 | 0.9986 | 1.0000 | 0.9986 | 0.9993 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.4224 | 0.0000 | 0.0000 | 0.0000 | 0.4224 | 1.0000 |
| 90 | 10 | 299,940 | 0.4799 | 0.1611 | 0.9986 | 0.2774 | 0.4222 | 0.9996 |
| 80 | 20 | 291,350 | 0.5375 | 0.3017 | 0.9986 | 0.4634 | 0.4222 | 0.9992 |
| 70 | 30 | 194,230 | 0.5948 | 0.4253 | 0.9986 | 0.5965 | 0.4217 | 0.9986 |
| 60 | 40 | 145,675 | 0.6533 | 0.5358 | 0.9986 | 0.6974 | 0.4231 | 0.9978 |
| 50 | 50 | 116,540 | 0.7105 | 0.6335 | 0.9986 | 0.7752 | 0.4224 | 0.9966 |
| 40 | 60 | 97,115 | 0.7672 | 0.7209 | 0.9986 | 0.8373 | 0.4200 | 0.9949 |
| 30 | 70 | 83,240 | 0.8262 | 0.8018 | 0.9986 | 0.8894 | 0.4239 | 0.9922 |
| 20 | 80 | 72,835 | 0.8832 | 0.8736 | 0.9986 | 0.9319 | 0.4219 | 0.9867 |
| 10 | 90 | 64,740 | 0.9402 | 0.9389 | 0.9986 | 0.9678 | 0.4152 | 0.9700 |
| 0 | 100 | 58,270 | 0.9986 | 1.0000 | 0.9986 | 0.9993 | 0.0000 | 0.0000 |


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
0.15       0.5728   0.3171   0.5262   0.9983   0.9918   0.1887   <--
0.20       0.5728   0.3171   0.5262   0.9983   0.9918   0.1887  
0.25       0.5728   0.3171   0.5262   0.9983   0.9918   0.1887  
0.30       0.5728   0.3171   0.5262   0.9983   0.9918   0.1887  
0.35       0.5728   0.3171   0.5262   0.9983   0.9918   0.1887  
0.40       0.5728   0.3171   0.5262   0.9983   0.9918   0.1887  
0.45       0.5728   0.3171   0.5262   0.9983   0.9918   0.1887  
0.50       0.5728   0.3171   0.5262   0.9983   0.9918   0.1887  
0.55       0.5728   0.3171   0.5262   0.9983   0.9918   0.1887  
0.60       0.5728   0.3171   0.5262   0.9983   0.9918   0.1887  
0.65       0.5728   0.3171   0.5262   0.9983   0.9918   0.1887  
0.70       0.5728   0.3171   0.5262   0.9983   0.9918   0.1887  
0.75       0.5728   0.3171   0.5262   0.9983   0.9918   0.1887  
0.80       0.5728   0.3171   0.5262   0.9983   0.9918   0.1887  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5728, F1=0.3171, Normal Recall=0.5262, Normal Precision=0.9983, Attack Recall=0.9918, Attack Precision=0.1887

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
0.15       0.6195   0.5105   0.5264   0.9963   0.9922   0.3437   <--
0.20       0.6195   0.5105   0.5264   0.9963   0.9922   0.3437  
0.25       0.6195   0.5105   0.5264   0.9963   0.9922   0.3437  
0.30       0.6195   0.5105   0.5264   0.9963   0.9922   0.3437  
0.35       0.6195   0.5105   0.5264   0.9963   0.9922   0.3437  
0.40       0.6195   0.5105   0.5264   0.9963   0.9922   0.3437  
0.45       0.6195   0.5105   0.5264   0.9963   0.9922   0.3437  
0.50       0.6195   0.5105   0.5264   0.9963   0.9922   0.3437  
0.55       0.6195   0.5105   0.5264   0.9963   0.9922   0.3437  
0.60       0.6195   0.5105   0.5264   0.9963   0.9922   0.3437  
0.65       0.6195   0.5105   0.5264   0.9963   0.9922   0.3437  
0.70       0.6195   0.5105   0.5264   0.9963   0.9922   0.3437  
0.75       0.6195   0.5105   0.5264   0.9963   0.9922   0.3437  
0.80       0.6195   0.5105   0.5264   0.9963   0.9922   0.3437  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6195, F1=0.5105, Normal Recall=0.5264, Normal Precision=0.9963, Attack Recall=0.9922, Attack Precision=0.3437

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
0.15       0.6661   0.6407   0.5264   0.9937   0.9922   0.4731   <--
0.20       0.6661   0.6407   0.5264   0.9937   0.9922   0.4731  
0.25       0.6661   0.6407   0.5264   0.9937   0.9922   0.4731  
0.30       0.6661   0.6407   0.5264   0.9937   0.9922   0.4731  
0.35       0.6661   0.6407   0.5264   0.9937   0.9922   0.4731  
0.40       0.6661   0.6407   0.5264   0.9937   0.9922   0.4731  
0.45       0.6661   0.6407   0.5264   0.9937   0.9922   0.4731  
0.50       0.6661   0.6407   0.5264   0.9937   0.9922   0.4731  
0.55       0.6661   0.6407   0.5264   0.9937   0.9922   0.4731  
0.60       0.6661   0.6407   0.5264   0.9937   0.9922   0.4731  
0.65       0.6661   0.6407   0.5264   0.9937   0.9922   0.4731  
0.70       0.6661   0.6407   0.5264   0.9937   0.9922   0.4731  
0.75       0.6661   0.6407   0.5264   0.9937   0.9922   0.4731  
0.80       0.6661   0.6407   0.5264   0.9937   0.9922   0.4731  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6661, F1=0.6407, Normal Recall=0.5264, Normal Precision=0.9937, Attack Recall=0.9922, Attack Precision=0.4731

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
0.15       0.7124   0.7340   0.5259   0.9902   0.9922   0.5825   <--
0.20       0.7124   0.7340   0.5259   0.9902   0.9922   0.5825  
0.25       0.7124   0.7340   0.5259   0.9902   0.9922   0.5825  
0.30       0.7124   0.7340   0.5259   0.9902   0.9922   0.5825  
0.35       0.7124   0.7340   0.5259   0.9902   0.9922   0.5825  
0.40       0.7124   0.7340   0.5259   0.9902   0.9922   0.5825  
0.45       0.7124   0.7340   0.5259   0.9902   0.9922   0.5825  
0.50       0.7124   0.7340   0.5259   0.9902   0.9922   0.5825  
0.55       0.7124   0.7340   0.5259   0.9902   0.9922   0.5825  
0.60       0.7124   0.7340   0.5259   0.9902   0.9922   0.5825  
0.65       0.7124   0.7340   0.5259   0.9902   0.9922   0.5825  
0.70       0.7124   0.7340   0.5259   0.9902   0.9922   0.5825  
0.75       0.7124   0.7340   0.5259   0.9902   0.9922   0.5825  
0.80       0.7124   0.7340   0.5259   0.9902   0.9922   0.5825  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7124, F1=0.7340, Normal Recall=0.5259, Normal Precision=0.9902, Attack Recall=0.9922, Attack Precision=0.5825

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
0.15       0.7584   0.8042   0.5246   0.9853   0.9922   0.6761   <--
0.20       0.7584   0.8042   0.5246   0.9853   0.9922   0.6761  
0.25       0.7584   0.8042   0.5246   0.9853   0.9922   0.6761  
0.30       0.7584   0.8042   0.5246   0.9853   0.9922   0.6761  
0.35       0.7584   0.8042   0.5246   0.9853   0.9922   0.6761  
0.40       0.7584   0.8042   0.5246   0.9853   0.9922   0.6761  
0.45       0.7584   0.8042   0.5246   0.9853   0.9922   0.6761  
0.50       0.7584   0.8042   0.5246   0.9853   0.9922   0.6761  
0.55       0.7584   0.8042   0.5246   0.9853   0.9922   0.6761  
0.60       0.7584   0.8042   0.5246   0.9853   0.9922   0.6761  
0.65       0.7584   0.8042   0.5246   0.9853   0.9922   0.6761  
0.70       0.7584   0.8042   0.5246   0.9853   0.9922   0.6761  
0.75       0.7584   0.8042   0.5246   0.9853   0.9922   0.6761  
0.80       0.7584   0.8042   0.5246   0.9853   0.9922   0.6761  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7584, F1=0.8042, Normal Recall=0.5246, Normal Precision=0.9853, Attack Recall=0.9922, Attack Precision=0.6761

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
0.15       0.4813   0.2780   0.4238   0.9997   0.9987   0.1615   <--
0.20       0.4813   0.2780   0.4238   0.9997   0.9987   0.1615  
0.25       0.4813   0.2780   0.4238   0.9997   0.9987   0.1615  
0.30       0.4813   0.2780   0.4238   0.9997   0.9987   0.1615  
0.35       0.4813   0.2780   0.4238   0.9997   0.9987   0.1615  
0.40       0.4813   0.2780   0.4238   0.9997   0.9987   0.1615  
0.45       0.4813   0.2780   0.4238   0.9997   0.9987   0.1615  
0.50       0.4813   0.2780   0.4238   0.9997   0.9987   0.1615  
0.55       0.4813   0.2780   0.4238   0.9997   0.9987   0.1615  
0.60       0.4813   0.2780   0.4238   0.9997   0.9987   0.1615  
0.65       0.4813   0.2780   0.4238   0.9997   0.9987   0.1615  
0.70       0.4813   0.2780   0.4238   0.9997   0.9987   0.1615  
0.75       0.4813   0.2780   0.4238   0.9997   0.9987   0.1615  
0.80       0.4813   0.2780   0.4238   0.9997   0.9987   0.1615  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4813, F1=0.2780, Normal Recall=0.4238, Normal Precision=0.9997, Attack Recall=0.9987, Attack Precision=0.1615

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
0.15       0.5387   0.4640   0.4237   0.9991   0.9985   0.3023   <--
0.20       0.5387   0.4640   0.4237   0.9991   0.9985   0.3023  
0.25       0.5387   0.4640   0.4237   0.9991   0.9985   0.3023  
0.30       0.5387   0.4640   0.4237   0.9991   0.9985   0.3023  
0.35       0.5387   0.4640   0.4237   0.9991   0.9985   0.3023  
0.40       0.5387   0.4640   0.4237   0.9991   0.9985   0.3023  
0.45       0.5387   0.4640   0.4237   0.9991   0.9985   0.3023  
0.50       0.5387   0.4640   0.4237   0.9991   0.9985   0.3023  
0.55       0.5387   0.4640   0.4237   0.9991   0.9985   0.3023  
0.60       0.5387   0.4640   0.4237   0.9991   0.9985   0.3023  
0.65       0.5387   0.4640   0.4237   0.9991   0.9985   0.3023  
0.70       0.5387   0.4640   0.4237   0.9991   0.9985   0.3023  
0.75       0.5387   0.4640   0.4237   0.9991   0.9985   0.3023  
0.80       0.5387   0.4640   0.4237   0.9991   0.9985   0.3023  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5387, F1=0.4640, Normal Recall=0.4237, Normal Precision=0.9991, Attack Recall=0.9985, Attack Precision=0.3023

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
0.15       0.5961   0.5973   0.4236   0.9985   0.9985   0.4261   <--
0.20       0.5961   0.5973   0.4236   0.9985   0.9985   0.4261  
0.25       0.5961   0.5973   0.4236   0.9985   0.9985   0.4261  
0.30       0.5961   0.5973   0.4236   0.9985   0.9985   0.4261  
0.35       0.5961   0.5973   0.4236   0.9985   0.9985   0.4261  
0.40       0.5961   0.5973   0.4236   0.9985   0.9985   0.4261  
0.45       0.5961   0.5973   0.4236   0.9985   0.9985   0.4261  
0.50       0.5961   0.5973   0.4236   0.9985   0.9985   0.4261  
0.55       0.5961   0.5973   0.4236   0.9985   0.9985   0.4261  
0.60       0.5961   0.5973   0.4236   0.9985   0.9985   0.4261  
0.65       0.5961   0.5973   0.4236   0.9985   0.9985   0.4261  
0.70       0.5961   0.5973   0.4236   0.9985   0.9985   0.4261  
0.75       0.5961   0.5973   0.4236   0.9985   0.9985   0.4261  
0.80       0.5961   0.5973   0.4236   0.9985   0.9985   0.4261  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5961, F1=0.5973, Normal Recall=0.4236, Normal Precision=0.9985, Attack Recall=0.9985, Attack Precision=0.4261

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
0.15       0.6533   0.6974   0.4232   0.9977   0.9985   0.5358   <--
0.20       0.6533   0.6974   0.4232   0.9977   0.9985   0.5358  
0.25       0.6533   0.6974   0.4232   0.9977   0.9985   0.5358  
0.30       0.6533   0.6974   0.4232   0.9977   0.9985   0.5358  
0.35       0.6533   0.6974   0.4232   0.9977   0.9985   0.5358  
0.40       0.6533   0.6974   0.4232   0.9977   0.9985   0.5358  
0.45       0.6533   0.6974   0.4232   0.9977   0.9985   0.5358  
0.50       0.6533   0.6974   0.4232   0.9977   0.9985   0.5358  
0.55       0.6533   0.6974   0.4232   0.9977   0.9985   0.5358  
0.60       0.6533   0.6974   0.4232   0.9977   0.9985   0.5358  
0.65       0.6533   0.6974   0.4232   0.9977   0.9985   0.5358  
0.70       0.6533   0.6974   0.4232   0.9977   0.9985   0.5358  
0.75       0.6533   0.6974   0.4232   0.9977   0.9985   0.5358  
0.80       0.6533   0.6974   0.4232   0.9977   0.9985   0.5358  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6533, F1=0.6974, Normal Recall=0.4232, Normal Precision=0.9977, Attack Recall=0.9985, Attack Precision=0.5358

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
0.15       0.7108   0.7754   0.4231   0.9966   0.9985   0.6338   <--
0.20       0.7108   0.7754   0.4231   0.9966   0.9985   0.6338  
0.25       0.7108   0.7754   0.4231   0.9966   0.9985   0.6338  
0.30       0.7108   0.7754   0.4231   0.9966   0.9985   0.6338  
0.35       0.7108   0.7754   0.4231   0.9966   0.9985   0.6338  
0.40       0.7108   0.7754   0.4231   0.9966   0.9985   0.6338  
0.45       0.7108   0.7754   0.4231   0.9966   0.9985   0.6338  
0.50       0.7108   0.7754   0.4231   0.9966   0.9985   0.6338  
0.55       0.7108   0.7754   0.4231   0.9966   0.9985   0.6338  
0.60       0.7108   0.7754   0.4231   0.9966   0.9985   0.6338  
0.65       0.7108   0.7754   0.4231   0.9966   0.9985   0.6338  
0.70       0.7108   0.7754   0.4231   0.9966   0.9985   0.6338  
0.75       0.7108   0.7754   0.4231   0.9966   0.9985   0.6338  
0.80       0.7108   0.7754   0.4231   0.9966   0.9985   0.6338  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7108, F1=0.7754, Normal Recall=0.4231, Normal Precision=0.9966, Attack Recall=0.9985, Attack Precision=0.6338

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
0.15       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611   <--
0.20       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.25       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.30       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.35       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.40       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.45       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.50       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.55       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.60       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.65       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.70       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.75       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.80       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4799, F1=0.2775, Normal Recall=0.4222, Normal Precision=0.9996, Attack Recall=0.9987, Attack Precision=0.1611

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
0.15       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017   <--
0.20       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.25       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.30       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.35       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.40       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.45       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.50       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.55       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.60       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.65       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.70       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.75       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.80       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5374, F1=0.4634, Normal Recall=0.4221, Normal Precision=0.9992, Attack Recall=0.9986, Attack Precision=0.3017

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
0.15       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255   <--
0.20       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.25       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.30       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.35       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.40       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.45       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.50       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.55       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.60       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.65       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.70       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.75       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.80       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5951, F1=0.5967, Normal Recall=0.4222, Normal Precision=0.9986, Attack Recall=0.9986, Attack Precision=0.4255

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
0.15       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352   <--
0.20       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.25       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.30       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.35       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.40       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.45       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.50       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.55       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.60       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.65       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.70       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.75       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.80       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6525, F1=0.6969, Normal Recall=0.4218, Normal Precision=0.9978, Attack Recall=0.9986, Attack Precision=0.5352

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
0.15       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333   <--
0.20       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.25       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.30       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.35       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.40       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.45       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.50       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.55       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.60       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.65       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.70       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.75       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.80       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7102, F1=0.7751, Normal Recall=0.4218, Normal Precision=0.9966, Attack Recall=0.9986, Attack Precision=0.6333

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
0.15       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611   <--
0.20       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.25       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.30       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.35       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.40       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.45       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.50       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.55       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.60       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.65       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.70       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.75       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
0.80       0.4799   0.2775   0.4222   0.9996   0.9987   0.1611  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4799, F1=0.2775, Normal Recall=0.4222, Normal Precision=0.9996, Attack Recall=0.9987, Attack Precision=0.1611

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
0.15       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017   <--
0.20       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.25       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.30       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.35       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.40       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.45       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.50       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.55       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.60       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.65       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.70       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.75       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
0.80       0.5374   0.4634   0.4221   0.9992   0.9986   0.3017  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5374, F1=0.4634, Normal Recall=0.4221, Normal Precision=0.9992, Attack Recall=0.9986, Attack Precision=0.3017

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
0.15       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255   <--
0.20       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.25       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.30       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.35       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.40       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.45       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.50       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.55       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.60       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.65       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.70       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.75       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
0.80       0.5951   0.5967   0.4222   0.9986   0.9986   0.4255  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5951, F1=0.5967, Normal Recall=0.4222, Normal Precision=0.9986, Attack Recall=0.9986, Attack Precision=0.4255

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
0.15       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352   <--
0.20       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.25       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.30       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.35       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.40       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.45       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.50       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.55       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.60       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.65       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.70       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.75       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
0.80       0.6525   0.6969   0.4218   0.9978   0.9986   0.5352  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6525, F1=0.6969, Normal Recall=0.4218, Normal Precision=0.9978, Attack Recall=0.9986, Attack Precision=0.5352

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
0.15       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333   <--
0.20       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.25       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.30       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.35       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.40       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.45       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.50       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.55       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.60       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.65       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.70       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.75       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
0.80       0.7102   0.7751   0.4218   0.9966   0.9986   0.6333  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7102, F1=0.7751, Normal Recall=0.4218, Normal Precision=0.9966, Attack Recall=0.9986, Attack Precision=0.6333

```

