# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e1_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-22 04:08:37 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 1 |
| **Batch size** | 256 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8522 | 0.8660 | 0.8800 | 0.8951 | 0.9094 | 0.9227 | 0.9380 | 0.9516 | 0.9666 | 0.9804 | 0.9949 |
| QAT+Prune only | 0.7797 | 0.7964 | 0.8132 | 0.8304 | 0.8471 | 0.8635 | 0.8810 | 0.8964 | 0.9159 | 0.9321 | 0.9490 |
| QAT+PTQ | 0.7799 | 0.7965 | 0.8133 | 0.8306 | 0.8473 | 0.8638 | 0.8811 | 0.8967 | 0.9161 | 0.9324 | 0.9493 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7799 | 0.7965 | 0.8133 | 0.8306 | 0.8473 | 0.8638 | 0.8811 | 0.8967 | 0.9161 | 0.9324 | 0.9493 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.5976 | 0.7683 | 0.8506 | 0.8978 | 0.9279 | 0.9507 | 0.9664 | 0.9794 | 0.9892 | 0.9974 |
| QAT+Prune only | 0.0000 | 0.4824 | 0.6702 | 0.7705 | 0.8323 | 0.8743 | 0.9054 | 0.9277 | 0.9475 | 0.9618 | 0.9738 |
| QAT+PTQ | 0.0000 | 0.4826 | 0.6704 | 0.7708 | 0.8326 | 0.8745 | 0.9055 | 0.9279 | 0.9476 | 0.9619 | 0.9740 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4826 | 0.6704 | 0.7708 | 0.8326 | 0.8745 | 0.9055 | 0.9279 | 0.9476 | 0.9619 | 0.9740 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8522 | 0.8517 | 0.8513 | 0.8524 | 0.8525 | 0.8505 | 0.8528 | 0.8507 | 0.8534 | 0.8505 | 0.0000 |
| QAT+Prune only | 0.7797 | 0.7794 | 0.7793 | 0.7796 | 0.7791 | 0.7780 | 0.7791 | 0.7738 | 0.7836 | 0.7797 | 0.0000 |
| QAT+PTQ | 0.7799 | 0.7795 | 0.7793 | 0.7798 | 0.7792 | 0.7783 | 0.7789 | 0.7740 | 0.7830 | 0.7804 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7799 | 0.7795 | 0.7793 | 0.7798 | 0.7792 | 0.7783 | 0.7789 | 0.7740 | 0.7830 | 0.7804 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8522 | 0.0000 | 0.0000 | 0.0000 | 0.8522 | 1.0000 |
| 90 | 10 | 299,940 | 0.8660 | 0.4271 | 0.9948 | 0.5976 | 0.8517 | 0.9993 |
| 80 | 20 | 291,350 | 0.8800 | 0.6259 | 0.9949 | 0.7683 | 0.8513 | 0.9985 |
| 70 | 30 | 194,230 | 0.8951 | 0.7428 | 0.9949 | 0.8506 | 0.8524 | 0.9974 |
| 60 | 40 | 145,675 | 0.9094 | 0.8181 | 0.9949 | 0.8978 | 0.8525 | 0.9960 |
| 50 | 50 | 116,540 | 0.9227 | 0.8693 | 0.9949 | 0.9279 | 0.8505 | 0.9940 |
| 40 | 60 | 97,115 | 0.9380 | 0.9102 | 0.9949 | 0.9507 | 0.8528 | 0.9910 |
| 30 | 70 | 83,240 | 0.9516 | 0.9396 | 0.9949 | 0.9664 | 0.8507 | 0.9861 |
| 20 | 80 | 72,835 | 0.9666 | 0.9645 | 0.9949 | 0.9794 | 0.8534 | 0.9764 |
| 10 | 90 | 64,740 | 0.9804 | 0.9836 | 0.9949 | 0.9892 | 0.8505 | 0.9483 |
| 0 | 100 | 58,270 | 0.9949 | 1.0000 | 0.9949 | 0.9974 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7797 | 0.0000 | 0.0000 | 0.0000 | 0.7797 | 1.0000 |
| 90 | 10 | 299,940 | 0.7964 | 0.3234 | 0.9489 | 0.4824 | 0.7794 | 0.9928 |
| 80 | 20 | 291,350 | 0.8132 | 0.5180 | 0.9490 | 0.6702 | 0.7793 | 0.9839 |
| 70 | 30 | 194,230 | 0.8304 | 0.6486 | 0.9490 | 0.7705 | 0.7796 | 0.9727 |
| 60 | 40 | 145,675 | 0.8471 | 0.7412 | 0.9490 | 0.8323 | 0.7791 | 0.9582 |
| 50 | 50 | 116,540 | 0.8635 | 0.8104 | 0.9490 | 0.8743 | 0.7780 | 0.9385 |
| 40 | 60 | 97,115 | 0.8810 | 0.8657 | 0.9490 | 0.9054 | 0.7791 | 0.9106 |
| 30 | 70 | 83,240 | 0.8964 | 0.9073 | 0.9490 | 0.9277 | 0.7738 | 0.8667 |
| 20 | 80 | 72,835 | 0.9159 | 0.9461 | 0.9490 | 0.9475 | 0.7836 | 0.7935 |
| 10 | 90 | 64,740 | 0.9321 | 0.9749 | 0.9490 | 0.9618 | 0.7797 | 0.6295 |
| 0 | 100 | 58,270 | 0.9490 | 1.0000 | 0.9490 | 0.9738 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7799 | 0.0000 | 0.0000 | 0.0000 | 0.7799 | 1.0000 |
| 90 | 10 | 299,940 | 0.7965 | 0.3236 | 0.9493 | 0.4826 | 0.7795 | 0.9928 |
| 80 | 20 | 291,350 | 0.8133 | 0.5182 | 0.9493 | 0.6704 | 0.7793 | 0.9840 |
| 70 | 30 | 194,230 | 0.8306 | 0.6488 | 0.9493 | 0.7708 | 0.7798 | 0.9729 |
| 60 | 40 | 145,675 | 0.8473 | 0.7414 | 0.9493 | 0.8326 | 0.7792 | 0.9584 |
| 50 | 50 | 116,540 | 0.8638 | 0.8107 | 0.9493 | 0.8745 | 0.7783 | 0.9388 |
| 40 | 60 | 97,115 | 0.8811 | 0.8656 | 0.9493 | 0.9055 | 0.7789 | 0.9111 |
| 30 | 70 | 83,240 | 0.8967 | 0.9074 | 0.9493 | 0.9279 | 0.7740 | 0.8674 |
| 20 | 80 | 72,835 | 0.9161 | 0.9459 | 0.9493 | 0.9476 | 0.7830 | 0.7943 |
| 10 | 90 | 64,740 | 0.9324 | 0.9749 | 0.9493 | 0.9619 | 0.7804 | 0.6310 |
| 0 | 100 | 58,270 | 0.9493 | 1.0000 | 0.9493 | 0.9740 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7799 | 0.0000 | 0.0000 | 0.0000 | 0.7799 | 1.0000 |
| 90 | 10 | 299,940 | 0.7965 | 0.3236 | 0.9493 | 0.4826 | 0.7795 | 0.9928 |
| 80 | 20 | 291,350 | 0.8133 | 0.5182 | 0.9493 | 0.6704 | 0.7793 | 0.9840 |
| 70 | 30 | 194,230 | 0.8306 | 0.6488 | 0.9493 | 0.7708 | 0.7798 | 0.9729 |
| 60 | 40 | 145,675 | 0.8473 | 0.7414 | 0.9493 | 0.8326 | 0.7792 | 0.9584 |
| 50 | 50 | 116,540 | 0.8638 | 0.8107 | 0.9493 | 0.8745 | 0.7783 | 0.9388 |
| 40 | 60 | 97,115 | 0.8811 | 0.8656 | 0.9493 | 0.9055 | 0.7789 | 0.9111 |
| 30 | 70 | 83,240 | 0.8967 | 0.9074 | 0.9493 | 0.9279 | 0.7740 | 0.8674 |
| 20 | 80 | 72,835 | 0.9161 | 0.9459 | 0.9493 | 0.9476 | 0.7830 | 0.7943 |
| 10 | 90 | 64,740 | 0.9324 | 0.9749 | 0.9493 | 0.9619 | 0.7804 | 0.6310 |
| 0 | 100 | 58,270 | 0.9493 | 1.0000 | 0.9493 | 0.9740 | 0.0000 | 0.0000 |


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
0.15       0.8661   0.5979   0.8517   0.9994   0.9954   0.4273   <--
0.20       0.8661   0.5979   0.8517   0.9994   0.9954   0.4273  
0.25       0.8661   0.5979   0.8517   0.9994   0.9954   0.4273  
0.30       0.8661   0.5979   0.8517   0.9994   0.9954   0.4273  
0.35       0.8661   0.5979   0.8517   0.9994   0.9954   0.4273  
0.40       0.8661   0.5979   0.8517   0.9994   0.9954   0.4273  
0.45       0.8661   0.5979   0.8517   0.9994   0.9954   0.4273  
0.50       0.8661   0.5979   0.8517   0.9994   0.9954   0.4273  
0.55       0.8661   0.5979   0.8517   0.9994   0.9954   0.4273  
0.60       0.8661   0.5979   0.8517   0.9994   0.9954   0.4273  
0.65       0.8661   0.5979   0.8517   0.9994   0.9954   0.4273  
0.70       0.8661   0.5979   0.8517   0.9994   0.9954   0.4273  
0.75       0.8661   0.5979   0.8517   0.9994   0.9954   0.4273  
0.80       0.8661   0.5979   0.8517   0.9994   0.9954   0.4273  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8661, F1=0.5979, Normal Recall=0.8517, Normal Precision=0.9994, Attack Recall=0.9954, Attack Precision=0.4273

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
0.15       0.8806   0.7693   0.8521   0.9985   0.9949   0.6271   <--
0.20       0.8806   0.7693   0.8521   0.9985   0.9949   0.6271  
0.25       0.8806   0.7693   0.8521   0.9985   0.9949   0.6271  
0.30       0.8806   0.7693   0.8521   0.9985   0.9949   0.6271  
0.35       0.8806   0.7693   0.8521   0.9985   0.9949   0.6271  
0.40       0.8806   0.7693   0.8521   0.9985   0.9949   0.6271  
0.45       0.8806   0.7693   0.8521   0.9985   0.9949   0.6271  
0.50       0.8806   0.7693   0.8521   0.9985   0.9949   0.6271  
0.55       0.8806   0.7693   0.8521   0.9985   0.9949   0.6271  
0.60       0.8806   0.7693   0.8521   0.9985   0.9949   0.6271  
0.65       0.8806   0.7693   0.8521   0.9985   0.9949   0.6271  
0.70       0.8806   0.7693   0.8521   0.9985   0.9949   0.6271  
0.75       0.8806   0.7693   0.8521   0.9985   0.9949   0.6271  
0.80       0.8806   0.7693   0.8521   0.9985   0.9949   0.6271  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8806, F1=0.7693, Normal Recall=0.8521, Normal Precision=0.9985, Attack Recall=0.9949, Attack Precision=0.6271

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
0.15       0.8951   0.8505   0.8523   0.9974   0.9949   0.7427   <--
0.20       0.8951   0.8505   0.8523   0.9974   0.9949   0.7427  
0.25       0.8951   0.8505   0.8523   0.9974   0.9949   0.7427  
0.30       0.8951   0.8505   0.8523   0.9974   0.9949   0.7427  
0.35       0.8951   0.8505   0.8523   0.9974   0.9949   0.7427  
0.40       0.8951   0.8505   0.8523   0.9974   0.9949   0.7427  
0.45       0.8951   0.8505   0.8523   0.9974   0.9949   0.7427  
0.50       0.8951   0.8505   0.8523   0.9974   0.9949   0.7427  
0.55       0.8951   0.8505   0.8523   0.9974   0.9949   0.7427  
0.60       0.8951   0.8505   0.8523   0.9974   0.9949   0.7427  
0.65       0.8951   0.8505   0.8523   0.9974   0.9949   0.7427  
0.70       0.8951   0.8505   0.8523   0.9974   0.9949   0.7427  
0.75       0.8951   0.8505   0.8523   0.9974   0.9949   0.7427  
0.80       0.8951   0.8505   0.8523   0.9974   0.9949   0.7427  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8951, F1=0.8505, Normal Recall=0.8523, Normal Precision=0.9974, Attack Recall=0.9949, Attack Precision=0.7427

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
0.15       0.9095   0.8979   0.8526   0.9960   0.9949   0.8182   <--
0.20       0.9095   0.8979   0.8526   0.9960   0.9949   0.8182  
0.25       0.9095   0.8979   0.8526   0.9960   0.9949   0.8182  
0.30       0.9095   0.8979   0.8526   0.9960   0.9949   0.8182  
0.35       0.9095   0.8979   0.8526   0.9960   0.9949   0.8182  
0.40       0.9095   0.8979   0.8526   0.9960   0.9949   0.8182  
0.45       0.9095   0.8979   0.8526   0.9960   0.9949   0.8182  
0.50       0.9095   0.8979   0.8526   0.9960   0.9949   0.8182  
0.55       0.9095   0.8979   0.8526   0.9960   0.9949   0.8182  
0.60       0.9095   0.8979   0.8526   0.9960   0.9949   0.8182  
0.65       0.9095   0.8979   0.8526   0.9960   0.9949   0.8182  
0.70       0.9095   0.8979   0.8526   0.9960   0.9949   0.8182  
0.75       0.9095   0.8979   0.8526   0.9960   0.9949   0.8182  
0.80       0.9095   0.8979   0.8526   0.9960   0.9949   0.8182  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9095, F1=0.8979, Normal Recall=0.8526, Normal Precision=0.9960, Attack Recall=0.9949, Attack Precision=0.8182

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
0.15       0.9235   0.9286   0.8521   0.9940   0.9949   0.8705   <--
0.20       0.9235   0.9286   0.8521   0.9940   0.9949   0.8705  
0.25       0.9235   0.9286   0.8521   0.9940   0.9949   0.8705  
0.30       0.9235   0.9286   0.8521   0.9940   0.9949   0.8705  
0.35       0.9235   0.9286   0.8521   0.9940   0.9949   0.8705  
0.40       0.9235   0.9286   0.8521   0.9940   0.9949   0.8705  
0.45       0.9235   0.9286   0.8521   0.9940   0.9949   0.8705  
0.50       0.9235   0.9286   0.8521   0.9940   0.9949   0.8705  
0.55       0.9235   0.9286   0.8521   0.9940   0.9949   0.8705  
0.60       0.9235   0.9286   0.8521   0.9940   0.9949   0.8705  
0.65       0.9235   0.9286   0.8521   0.9940   0.9949   0.8705  
0.70       0.9235   0.9286   0.8521   0.9940   0.9949   0.8705  
0.75       0.9235   0.9286   0.8521   0.9940   0.9949   0.8705  
0.80       0.9235   0.9286   0.8521   0.9940   0.9949   0.8705  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9235, F1=0.9286, Normal Recall=0.8521, Normal Precision=0.9940, Attack Recall=0.9949, Attack Precision=0.8705

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
0.15       0.7964   0.4824   0.7794   0.9928   0.9488   0.3234   <--
0.20       0.7964   0.4824   0.7794   0.9928   0.9488   0.3234  
0.25       0.7964   0.4824   0.7794   0.9928   0.9488   0.3234  
0.30       0.7964   0.4824   0.7794   0.9928   0.9488   0.3234  
0.35       0.7964   0.4824   0.7794   0.9928   0.9488   0.3234  
0.40       0.7964   0.4824   0.7794   0.9928   0.9488   0.3234  
0.45       0.7964   0.4824   0.7794   0.9928   0.9488   0.3234  
0.50       0.7964   0.4824   0.7794   0.9928   0.9488   0.3234  
0.55       0.7964   0.4824   0.7794   0.9928   0.9488   0.3234  
0.60       0.7964   0.4824   0.7794   0.9928   0.9488   0.3234  
0.65       0.7964   0.4824   0.7794   0.9928   0.9488   0.3234  
0.70       0.7964   0.4824   0.7794   0.9928   0.9488   0.3234  
0.75       0.7964   0.4824   0.7794   0.9928   0.9488   0.3234  
0.80       0.7964   0.4824   0.7794   0.9928   0.9488   0.3234  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7964, F1=0.4824, Normal Recall=0.7794, Normal Precision=0.9928, Attack Recall=0.9488, Attack Precision=0.3234

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
0.15       0.8136   0.6706   0.7797   0.9839   0.9490   0.5185   <--
0.20       0.8136   0.6706   0.7797   0.9839   0.9490   0.5185  
0.25       0.8136   0.6706   0.7797   0.9839   0.9490   0.5185  
0.30       0.8136   0.6706   0.7797   0.9839   0.9490   0.5185  
0.35       0.8136   0.6706   0.7797   0.9839   0.9490   0.5185  
0.40       0.8136   0.6706   0.7797   0.9839   0.9490   0.5185  
0.45       0.8136   0.6706   0.7797   0.9839   0.9490   0.5185  
0.50       0.8136   0.6706   0.7797   0.9839   0.9490   0.5185  
0.55       0.8136   0.6706   0.7797   0.9839   0.9490   0.5185  
0.60       0.8136   0.6706   0.7797   0.9839   0.9490   0.5185  
0.65       0.8136   0.6706   0.7797   0.9839   0.9490   0.5185  
0.70       0.8136   0.6706   0.7797   0.9839   0.9490   0.5185  
0.75       0.8136   0.6706   0.7797   0.9839   0.9490   0.5185  
0.80       0.8136   0.6706   0.7797   0.9839   0.9490   0.5185  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8136, F1=0.6706, Normal Recall=0.7797, Normal Precision=0.9839, Attack Recall=0.9490, Attack Precision=0.5185

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
0.15       0.8304   0.7705   0.7795   0.9727   0.9490   0.6485   <--
0.20       0.8304   0.7705   0.7795   0.9727   0.9490   0.6485  
0.25       0.8304   0.7705   0.7795   0.9727   0.9490   0.6485  
0.30       0.8304   0.7705   0.7795   0.9727   0.9490   0.6485  
0.35       0.8304   0.7705   0.7795   0.9727   0.9490   0.6485  
0.40       0.8304   0.7705   0.7795   0.9727   0.9490   0.6485  
0.45       0.8304   0.7705   0.7795   0.9727   0.9490   0.6485  
0.50       0.8304   0.7705   0.7795   0.9727   0.9490   0.6485  
0.55       0.8304   0.7705   0.7795   0.9727   0.9490   0.6485  
0.60       0.8304   0.7705   0.7795   0.9727   0.9490   0.6485  
0.65       0.8304   0.7705   0.7795   0.9727   0.9490   0.6485  
0.70       0.8304   0.7705   0.7795   0.9727   0.9490   0.6485  
0.75       0.8304   0.7705   0.7795   0.9727   0.9490   0.6485  
0.80       0.8304   0.7705   0.7795   0.9727   0.9490   0.6485  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8304, F1=0.7705, Normal Recall=0.7795, Normal Precision=0.9727, Attack Recall=0.9490, Attack Precision=0.6485

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
0.15       0.8472   0.8324   0.7793   0.9582   0.9490   0.7413   <--
0.20       0.8472   0.8324   0.7793   0.9582   0.9490   0.7413  
0.25       0.8472   0.8324   0.7793   0.9582   0.9490   0.7413  
0.30       0.8472   0.8324   0.7793   0.9582   0.9490   0.7413  
0.35       0.8472   0.8324   0.7793   0.9582   0.9490   0.7413  
0.40       0.8472   0.8324   0.7793   0.9582   0.9490   0.7413  
0.45       0.8472   0.8324   0.7793   0.9582   0.9490   0.7413  
0.50       0.8472   0.8324   0.7793   0.9582   0.9490   0.7413  
0.55       0.8472   0.8324   0.7793   0.9582   0.9490   0.7413  
0.60       0.8472   0.8324   0.7793   0.9582   0.9490   0.7413  
0.65       0.8472   0.8324   0.7793   0.9582   0.9490   0.7413  
0.70       0.8472   0.8324   0.7793   0.9582   0.9490   0.7413  
0.75       0.8472   0.8324   0.7793   0.9582   0.9490   0.7413  
0.80       0.8472   0.8324   0.7793   0.9582   0.9490   0.7413  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8472, F1=0.8324, Normal Recall=0.7793, Normal Precision=0.9582, Attack Recall=0.9490, Attack Precision=0.7413

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
0.15       0.8639   0.8746   0.7788   0.9386   0.9490   0.8110   <--
0.20       0.8639   0.8746   0.7788   0.9386   0.9490   0.8110  
0.25       0.8639   0.8746   0.7788   0.9386   0.9490   0.8110  
0.30       0.8639   0.8746   0.7788   0.9386   0.9490   0.8110  
0.35       0.8639   0.8746   0.7788   0.9386   0.9490   0.8110  
0.40       0.8639   0.8746   0.7788   0.9386   0.9490   0.8110  
0.45       0.8639   0.8746   0.7788   0.9386   0.9490   0.8110  
0.50       0.8639   0.8746   0.7788   0.9386   0.9490   0.8110  
0.55       0.8639   0.8746   0.7788   0.9386   0.9490   0.8110  
0.60       0.8639   0.8746   0.7788   0.9386   0.9490   0.8110  
0.65       0.8639   0.8746   0.7788   0.9386   0.9490   0.8110  
0.70       0.8639   0.8746   0.7788   0.9386   0.9490   0.8110  
0.75       0.8639   0.8746   0.7788   0.9386   0.9490   0.8110  
0.80       0.8639   0.8746   0.7788   0.9386   0.9490   0.8110  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8639, F1=0.8746, Normal Recall=0.7788, Normal Precision=0.9386, Attack Recall=0.9490, Attack Precision=0.8110

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
0.15       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236   <--
0.20       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.25       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.30       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.35       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.40       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.45       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.50       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.55       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.60       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.65       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.70       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.75       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.80       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7965, F1=0.4826, Normal Recall=0.7795, Normal Precision=0.9928, Attack Recall=0.9492, Attack Precision=0.3236

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
0.15       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187   <--
0.20       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.25       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.30       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.35       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.40       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.45       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.50       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.55       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.60       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.65       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.70       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.75       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.80       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8137, F1=0.6709, Normal Recall=0.7798, Normal Precision=0.9840, Attack Recall=0.9493, Attack Precision=0.5187

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
0.15       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487   <--
0.20       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.25       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.30       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.35       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.40       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.45       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.50       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.55       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.60       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.65       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.70       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.75       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.80       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8306, F1=0.7708, Normal Recall=0.7797, Normal Precision=0.9729, Attack Recall=0.9493, Attack Precision=0.6487

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
0.15       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416   <--
0.20       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.25       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.30       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.35       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.40       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.45       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.50       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.55       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.60       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.65       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.70       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.75       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.80       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8474, F1=0.8327, Normal Recall=0.7795, Normal Precision=0.9584, Attack Recall=0.9493, Attack Precision=0.7416

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
0.15       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112   <--
0.20       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.25       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.30       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.35       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.40       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.45       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.50       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.55       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.60       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.65       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.70       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.75       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.80       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8641, F1=0.8748, Normal Recall=0.7790, Normal Precision=0.9389, Attack Recall=0.9493, Attack Precision=0.8112

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
0.15       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236   <--
0.20       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.25       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.30       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.35       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.40       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.45       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.50       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.55       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.60       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.65       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.70       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.75       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
0.80       0.7965   0.4826   0.7795   0.9928   0.9492   0.3236  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7965, F1=0.4826, Normal Recall=0.7795, Normal Precision=0.9928, Attack Recall=0.9492, Attack Precision=0.3236

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
0.15       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187   <--
0.20       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.25       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.30       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.35       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.40       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.45       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.50       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.55       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.60       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.65       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.70       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.75       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
0.80       0.8137   0.6709   0.7798   0.9840   0.9493   0.5187  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8137, F1=0.6709, Normal Recall=0.7798, Normal Precision=0.9840, Attack Recall=0.9493, Attack Precision=0.5187

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
0.15       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487   <--
0.20       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.25       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.30       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.35       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.40       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.45       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.50       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.55       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.60       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.65       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.70       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.75       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
0.80       0.8306   0.7708   0.7797   0.9729   0.9493   0.6487  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8306, F1=0.7708, Normal Recall=0.7797, Normal Precision=0.9729, Attack Recall=0.9493, Attack Precision=0.6487

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
0.15       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416   <--
0.20       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.25       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.30       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.35       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.40       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.45       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.50       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.55       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.60       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.65       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.70       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.75       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
0.80       0.8474   0.8327   0.7795   0.9584   0.9493   0.7416  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8474, F1=0.8327, Normal Recall=0.7795, Normal Precision=0.9584, Attack Recall=0.9493, Attack Precision=0.7416

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
0.15       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112   <--
0.20       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.25       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.30       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.35       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.40       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.45       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.50       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.55       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.60       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.65       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.70       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.75       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
0.80       0.8641   0.8748   0.7790   0.9389   0.9493   0.8112  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8641, F1=0.8748, Normal Recall=0.7790, Normal Precision=0.9389, Attack Recall=0.9493, Attack Precision=0.8112

```

