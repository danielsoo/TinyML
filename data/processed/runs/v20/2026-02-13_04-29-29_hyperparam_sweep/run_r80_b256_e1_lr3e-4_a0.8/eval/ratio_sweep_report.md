# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e1_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-18 09:32:42 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5626 | 0.5770 | 0.5921 | 0.6081 | 0.6226 | 0.6386 | 0.6523 | 0.6703 | 0.6855 | 0.7007 | 0.7161 |
| QAT+Prune only | 0.0064 | 0.1056 | 0.2049 | 0.3044 | 0.4041 | 0.5029 | 0.6024 | 0.7018 | 0.8014 | 0.9007 | 1.0000 |
| QAT+PTQ | 0.0066 | 0.1058 | 0.2051 | 0.3045 | 0.4042 | 0.5030 | 0.6024 | 0.7019 | 0.8015 | 0.9007 | 1.0000 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.0066 | 0.1058 | 0.2051 | 0.3045 | 0.4042 | 0.5030 | 0.6024 | 0.7019 | 0.8015 | 0.9007 | 1.0000 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2528 | 0.4125 | 0.5230 | 0.6028 | 0.6646 | 0.7119 | 0.7525 | 0.7846 | 0.8116 | 0.8345 |
| QAT+Prune only | 0.0000 | 0.1828 | 0.3347 | 0.4631 | 0.5731 | 0.6679 | 0.7511 | 0.8244 | 0.8896 | 0.9477 | 1.0000 |
| QAT+PTQ | 0.0000 | 0.1828 | 0.3348 | 0.4632 | 0.5731 | 0.6680 | 0.7511 | 0.8244 | 0.8896 | 0.9477 | 1.0000 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.1828 | 0.3348 | 0.4632 | 0.5731 | 0.6680 | 0.7511 | 0.8244 | 0.8896 | 0.9477 | 1.0000 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5626 | 0.5616 | 0.5611 | 0.5618 | 0.5602 | 0.5611 | 0.5567 | 0.5635 | 0.5634 | 0.5629 | 0.0000 |
| QAT+Prune only | 0.0064 | 0.0063 | 0.0062 | 0.0063 | 0.0068 | 0.0057 | 0.0059 | 0.0060 | 0.0070 | 0.0071 | 0.0000 |
| QAT+PTQ | 0.0066 | 0.0065 | 0.0064 | 0.0065 | 0.0070 | 0.0059 | 0.0059 | 0.0063 | 0.0074 | 0.0073 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.0066 | 0.0065 | 0.0064 | 0.0065 | 0.0070 | 0.0059 | 0.0059 | 0.0063 | 0.0074 | 0.0073 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5626 | 0.0000 | 0.0000 | 0.0000 | 0.5626 | 1.0000 |
| 90 | 10 | 299,940 | 0.5770 | 0.1535 | 0.7154 | 0.2528 | 0.5616 | 0.9467 |
| 80 | 20 | 291,350 | 0.5921 | 0.2897 | 0.7161 | 0.4125 | 0.5611 | 0.8877 |
| 70 | 30 | 194,230 | 0.6081 | 0.4119 | 0.7161 | 0.5230 | 0.5618 | 0.8220 |
| 60 | 40 | 145,675 | 0.6226 | 0.5205 | 0.7161 | 0.6028 | 0.5602 | 0.7474 |
| 50 | 50 | 116,540 | 0.6386 | 0.6200 | 0.7161 | 0.6646 | 0.5611 | 0.6640 |
| 40 | 60 | 97,115 | 0.6523 | 0.7078 | 0.7161 | 0.7119 | 0.5567 | 0.5665 |
| 30 | 70 | 83,240 | 0.6703 | 0.7929 | 0.7161 | 0.7525 | 0.5635 | 0.4596 |
| 20 | 80 | 72,835 | 0.6855 | 0.8677 | 0.7161 | 0.7846 | 0.5634 | 0.3316 |
| 10 | 90 | 64,740 | 0.7007 | 0.9365 | 0.7161 | 0.8116 | 0.5629 | 0.1805 |
| 0 | 100 | 58,270 | 0.7161 | 1.0000 | 0.7161 | 0.8345 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0064 | 0.0000 | 0.0000 | 0.0000 | 0.0064 | 1.0000 |
| 90 | 10 | 299,940 | 0.1056 | 0.1006 | 1.0000 | 0.1828 | 0.0063 | 1.0000 |
| 80 | 20 | 291,350 | 0.2049 | 0.2010 | 1.0000 | 0.3347 | 0.0062 | 1.0000 |
| 70 | 30 | 194,230 | 0.3044 | 0.3013 | 1.0000 | 0.4631 | 0.0063 | 1.0000 |
| 60 | 40 | 145,675 | 0.4041 | 0.4016 | 1.0000 | 0.5731 | 0.0068 | 1.0000 |
| 50 | 50 | 116,540 | 0.5029 | 0.5014 | 1.0000 | 0.6679 | 0.0057 | 1.0000 |
| 40 | 60 | 97,115 | 0.6024 | 0.6014 | 1.0000 | 0.7511 | 0.0059 | 1.0000 |
| 30 | 70 | 83,240 | 0.7018 | 0.7013 | 1.0000 | 0.8244 | 0.0060 | 1.0000 |
| 20 | 80 | 72,835 | 0.8014 | 0.8011 | 1.0000 | 0.8896 | 0.0070 | 1.0000 |
| 10 | 90 | 64,740 | 0.9007 | 0.9006 | 1.0000 | 0.9477 | 0.0071 | 1.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0066 | 0.0000 | 0.0000 | 0.0000 | 0.0066 | 1.0000 |
| 90 | 10 | 299,940 | 0.1058 | 0.1006 | 1.0000 | 0.1828 | 0.0065 | 1.0000 |
| 80 | 20 | 291,350 | 0.2051 | 0.2010 | 1.0000 | 0.3348 | 0.0064 | 1.0000 |
| 70 | 30 | 194,230 | 0.3045 | 0.3014 | 1.0000 | 0.4632 | 0.0065 | 1.0000 |
| 60 | 40 | 145,675 | 0.4042 | 0.4017 | 1.0000 | 0.5731 | 0.0070 | 1.0000 |
| 50 | 50 | 116,540 | 0.5030 | 0.5015 | 1.0000 | 0.6680 | 0.0059 | 1.0000 |
| 40 | 60 | 97,115 | 0.6024 | 0.6014 | 1.0000 | 0.7511 | 0.0059 | 1.0000 |
| 30 | 70 | 83,240 | 0.7019 | 0.7013 | 1.0000 | 0.8244 | 0.0063 | 1.0000 |
| 20 | 80 | 72,835 | 0.8015 | 0.8012 | 1.0000 | 0.8896 | 0.0074 | 1.0000 |
| 10 | 90 | 64,740 | 0.9007 | 0.9007 | 1.0000 | 0.9477 | 0.0073 | 1.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.0066 | 0.0000 | 0.0000 | 0.0000 | 0.0066 | 1.0000 |
| 90 | 10 | 299,940 | 0.1058 | 0.1006 | 1.0000 | 0.1828 | 0.0065 | 1.0000 |
| 80 | 20 | 291,350 | 0.2051 | 0.2010 | 1.0000 | 0.3348 | 0.0064 | 1.0000 |
| 70 | 30 | 194,230 | 0.3045 | 0.3014 | 1.0000 | 0.4632 | 0.0065 | 1.0000 |
| 60 | 40 | 145,675 | 0.4042 | 0.4017 | 1.0000 | 0.5731 | 0.0070 | 1.0000 |
| 50 | 50 | 116,540 | 0.5030 | 0.5015 | 1.0000 | 0.6680 | 0.0059 | 1.0000 |
| 40 | 60 | 97,115 | 0.6024 | 0.6014 | 1.0000 | 0.7511 | 0.0059 | 1.0000 |
| 30 | 70 | 83,240 | 0.7019 | 0.7013 | 1.0000 | 0.8244 | 0.0063 | 1.0000 |
| 20 | 80 | 72,835 | 0.8015 | 0.8012 | 1.0000 | 0.8896 | 0.0074 | 1.0000 |
| 10 | 90 | 64,740 | 0.9007 | 0.9007 | 1.0000 | 0.9477 | 0.0073 | 1.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |


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
0.15       0.5772   0.2533   0.5616   0.9470   0.7172   0.1538   <--
0.20       0.5772   0.2533   0.5616   0.9470   0.7172   0.1538  
0.25       0.5772   0.2533   0.5616   0.9470   0.7172   0.1538  
0.30       0.5772   0.2533   0.5616   0.9470   0.7172   0.1538  
0.35       0.5772   0.2533   0.5616   0.9470   0.7172   0.1538  
0.40       0.5772   0.2533   0.5616   0.9470   0.7172   0.1538  
0.45       0.5772   0.2533   0.5616   0.9470   0.7172   0.1538  
0.50       0.5772   0.2533   0.5616   0.9470   0.7172   0.1538  
0.55       0.5772   0.2533   0.5616   0.9470   0.7172   0.1538  
0.60       0.5772   0.2533   0.5616   0.9470   0.7172   0.1538  
0.65       0.5772   0.2533   0.5616   0.9470   0.7172   0.1538  
0.70       0.5772   0.2533   0.5616   0.9470   0.7172   0.1538  
0.75       0.5772   0.2533   0.5616   0.9470   0.7172   0.1538  
0.80       0.5772   0.2533   0.5616   0.9470   0.7172   0.1538  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5772, F1=0.2533, Normal Recall=0.5616, Normal Precision=0.9470, Attack Recall=0.7172, Attack Precision=0.1538

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
0.15       0.5929   0.4130   0.5621   0.8879   0.7161   0.2902   <--
0.20       0.5929   0.4130   0.5621   0.8879   0.7161   0.2902  
0.25       0.5929   0.4130   0.5621   0.8879   0.7161   0.2902  
0.30       0.5929   0.4130   0.5621   0.8879   0.7161   0.2902  
0.35       0.5929   0.4130   0.5621   0.8879   0.7161   0.2902  
0.40       0.5929   0.4130   0.5621   0.8879   0.7161   0.2902  
0.45       0.5929   0.4130   0.5621   0.8879   0.7161   0.2902  
0.50       0.5929   0.4130   0.5621   0.8879   0.7161   0.2902  
0.55       0.5929   0.4130   0.5621   0.8879   0.7161   0.2902  
0.60       0.5929   0.4130   0.5621   0.8879   0.7161   0.2902  
0.65       0.5929   0.4130   0.5621   0.8879   0.7161   0.2902  
0.70       0.5929   0.4130   0.5621   0.8879   0.7161   0.2902  
0.75       0.5929   0.4130   0.5621   0.8879   0.7161   0.2902  
0.80       0.5929   0.4130   0.5621   0.8879   0.7161   0.2902  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5929, F1=0.4130, Normal Recall=0.5621, Normal Precision=0.8879, Attack Recall=0.7161, Attack Precision=0.2902

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
0.15       0.6084   0.5232   0.5623   0.8221   0.7161   0.4122   <--
0.20       0.6084   0.5232   0.5623   0.8221   0.7161   0.4122  
0.25       0.6084   0.5232   0.5623   0.8221   0.7161   0.4122  
0.30       0.6084   0.5232   0.5623   0.8221   0.7161   0.4122  
0.35       0.6084   0.5232   0.5623   0.8221   0.7161   0.4122  
0.40       0.6084   0.5232   0.5623   0.8221   0.7161   0.4122  
0.45       0.6084   0.5232   0.5623   0.8221   0.7161   0.4122  
0.50       0.6084   0.5232   0.5623   0.8221   0.7161   0.4122  
0.55       0.6084   0.5232   0.5623   0.8221   0.7161   0.4122  
0.60       0.6084   0.5232   0.5623   0.8221   0.7161   0.4122  
0.65       0.6084   0.5232   0.5623   0.8221   0.7161   0.4122  
0.70       0.6084   0.5232   0.5623   0.8221   0.7161   0.4122  
0.75       0.6084   0.5232   0.5623   0.8221   0.7161   0.4122  
0.80       0.6084   0.5232   0.5623   0.8221   0.7161   0.4122  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6084, F1=0.5232, Normal Recall=0.5623, Normal Precision=0.8221, Attack Recall=0.7161, Attack Precision=0.4122

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
0.15       0.6238   0.6036   0.5623   0.7481   0.7161   0.5217   <--
0.20       0.6238   0.6036   0.5623   0.7481   0.7161   0.5217  
0.25       0.6238   0.6036   0.5623   0.7481   0.7161   0.5217  
0.30       0.6238   0.6036   0.5623   0.7481   0.7161   0.5217  
0.35       0.6238   0.6036   0.5623   0.7481   0.7161   0.5217  
0.40       0.6238   0.6036   0.5623   0.7481   0.7161   0.5217  
0.45       0.6238   0.6036   0.5623   0.7481   0.7161   0.5217  
0.50       0.6238   0.6036   0.5623   0.7481   0.7161   0.5217  
0.55       0.6238   0.6036   0.5623   0.7481   0.7161   0.5217  
0.60       0.6238   0.6036   0.5623   0.7481   0.7161   0.5217  
0.65       0.6238   0.6036   0.5623   0.7481   0.7161   0.5217  
0.70       0.6238   0.6036   0.5623   0.7481   0.7161   0.5217  
0.75       0.6238   0.6036   0.5623   0.7481   0.7161   0.5217  
0.80       0.6238   0.6036   0.5623   0.7481   0.7161   0.5217  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6238, F1=0.6036, Normal Recall=0.5623, Normal Precision=0.7481, Attack Recall=0.7161, Attack Precision=0.5217

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
0.15       0.6393   0.6650   0.5625   0.6646   0.7161   0.6208   <--
0.20       0.6393   0.6650   0.5625   0.6646   0.7161   0.6208  
0.25       0.6393   0.6650   0.5625   0.6646   0.7161   0.6208  
0.30       0.6393   0.6650   0.5625   0.6646   0.7161   0.6208  
0.35       0.6393   0.6650   0.5625   0.6646   0.7161   0.6208  
0.40       0.6393   0.6650   0.5625   0.6646   0.7161   0.6208  
0.45       0.6393   0.6650   0.5625   0.6646   0.7161   0.6208  
0.50       0.6393   0.6650   0.5625   0.6646   0.7161   0.6208  
0.55       0.6393   0.6650   0.5625   0.6646   0.7161   0.6208  
0.60       0.6393   0.6650   0.5625   0.6646   0.7161   0.6208  
0.65       0.6393   0.6650   0.5625   0.6646   0.7161   0.6208  
0.70       0.6393   0.6650   0.5625   0.6646   0.7161   0.6208  
0.75       0.6393   0.6650   0.5625   0.6646   0.7161   0.6208  
0.80       0.6393   0.6650   0.5625   0.6646   0.7161   0.6208  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6393, F1=0.6650, Normal Recall=0.5625, Normal Precision=0.6646, Attack Recall=0.7161, Attack Precision=0.6208

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
0.15       0.1056   0.1828   0.0063   1.0000   1.0000   0.1006   <--
0.20       0.1056   0.1828   0.0063   1.0000   1.0000   0.1006  
0.25       0.1056   0.1828   0.0063   1.0000   1.0000   0.1006  
0.30       0.1056   0.1828   0.0063   1.0000   1.0000   0.1006  
0.35       0.1056   0.1828   0.0063   1.0000   1.0000   0.1006  
0.40       0.1056   0.1828   0.0063   1.0000   1.0000   0.1006  
0.45       0.1056   0.1828   0.0063   1.0000   1.0000   0.1006  
0.50       0.1056   0.1828   0.0063   1.0000   1.0000   0.1006  
0.55       0.1056   0.1828   0.0063   1.0000   1.0000   0.1006  
0.60       0.1056   0.1828   0.0063   1.0000   1.0000   0.1006  
0.65       0.1056   0.1828   0.0063   1.0000   1.0000   0.1006  
0.70       0.1056   0.1828   0.0063   1.0000   1.0000   0.1006  
0.75       0.1056   0.1828   0.0063   1.0000   1.0000   0.1006  
0.80       0.1056   0.1828   0.0063   1.0000   1.0000   0.1006  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.1056, F1=0.1828, Normal Recall=0.0063, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1006

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
0.15       0.2050   0.3347   0.0062   1.0000   1.0000   0.2010   <--
0.20       0.2050   0.3347   0.0062   1.0000   1.0000   0.2010  
0.25       0.2050   0.3347   0.0062   1.0000   1.0000   0.2010  
0.30       0.2050   0.3347   0.0062   1.0000   1.0000   0.2010  
0.35       0.2050   0.3347   0.0062   1.0000   1.0000   0.2010  
0.40       0.2050   0.3347   0.0062   1.0000   1.0000   0.2010  
0.45       0.2050   0.3347   0.0062   1.0000   1.0000   0.2010  
0.50       0.2050   0.3347   0.0062   1.0000   1.0000   0.2010  
0.55       0.2050   0.3347   0.0062   1.0000   1.0000   0.2010  
0.60       0.2050   0.3347   0.0062   1.0000   1.0000   0.2010  
0.65       0.2050   0.3347   0.0062   1.0000   1.0000   0.2010  
0.70       0.2050   0.3347   0.0062   1.0000   1.0000   0.2010  
0.75       0.2050   0.3347   0.0062   1.0000   1.0000   0.2010  
0.80       0.2050   0.3347   0.0062   1.0000   1.0000   0.2010  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2050, F1=0.3347, Normal Recall=0.0062, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.2010

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
0.15       0.3044   0.4631   0.0063   1.0000   1.0000   0.3013   <--
0.20       0.3044   0.4631   0.0063   1.0000   1.0000   0.3013  
0.25       0.3044   0.4631   0.0063   1.0000   1.0000   0.3013  
0.30       0.3044   0.4631   0.0063   1.0000   1.0000   0.3013  
0.35       0.3044   0.4631   0.0063   1.0000   1.0000   0.3013  
0.40       0.3044   0.4631   0.0063   1.0000   1.0000   0.3013  
0.45       0.3044   0.4631   0.0063   1.0000   1.0000   0.3013  
0.50       0.3044   0.4631   0.0063   1.0000   1.0000   0.3013  
0.55       0.3044   0.4631   0.0063   1.0000   1.0000   0.3013  
0.60       0.3044   0.4631   0.0063   1.0000   1.0000   0.3013  
0.65       0.3044   0.4631   0.0063   1.0000   1.0000   0.3013  
0.70       0.3044   0.4631   0.0063   1.0000   1.0000   0.3013  
0.75       0.3044   0.4631   0.0063   1.0000   1.0000   0.3013  
0.80       0.3044   0.4631   0.0063   1.0000   1.0000   0.3013  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3044, F1=0.4631, Normal Recall=0.0063, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3013

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
0.15       0.4039   0.5730   0.0065   1.0000   1.0000   0.4016   <--
0.20       0.4039   0.5730   0.0065   1.0000   1.0000   0.4016  
0.25       0.4039   0.5730   0.0065   1.0000   1.0000   0.4016  
0.30       0.4039   0.5730   0.0065   1.0000   1.0000   0.4016  
0.35       0.4039   0.5730   0.0065   1.0000   1.0000   0.4016  
0.40       0.4039   0.5730   0.0065   1.0000   1.0000   0.4016  
0.45       0.4039   0.5730   0.0065   1.0000   1.0000   0.4016  
0.50       0.4039   0.5730   0.0065   1.0000   1.0000   0.4016  
0.55       0.4039   0.5730   0.0065   1.0000   1.0000   0.4016  
0.60       0.4039   0.5730   0.0065   1.0000   1.0000   0.4016  
0.65       0.4039   0.5730   0.0065   1.0000   1.0000   0.4016  
0.70       0.4039   0.5730   0.0065   1.0000   1.0000   0.4016  
0.75       0.4039   0.5730   0.0065   1.0000   1.0000   0.4016  
0.80       0.4039   0.5730   0.0065   1.0000   1.0000   0.4016  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4039, F1=0.5730, Normal Recall=0.0065, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4016

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
0.15       0.5032   0.6681   0.0064   1.0000   1.0000   0.5016   <--
0.20       0.5032   0.6681   0.0064   1.0000   1.0000   0.5016  
0.25       0.5032   0.6681   0.0064   1.0000   1.0000   0.5016  
0.30       0.5032   0.6681   0.0064   1.0000   1.0000   0.5016  
0.35       0.5032   0.6681   0.0064   1.0000   1.0000   0.5016  
0.40       0.5032   0.6681   0.0064   1.0000   1.0000   0.5016  
0.45       0.5032   0.6681   0.0064   1.0000   1.0000   0.5016  
0.50       0.5032   0.6681   0.0064   1.0000   1.0000   0.5016  
0.55       0.5032   0.6681   0.0064   1.0000   1.0000   0.5016  
0.60       0.5032   0.6681   0.0064   1.0000   1.0000   0.5016  
0.65       0.5032   0.6681   0.0064   1.0000   1.0000   0.5016  
0.70       0.5032   0.6681   0.0064   1.0000   1.0000   0.5016  
0.75       0.5032   0.6681   0.0064   1.0000   1.0000   0.5016  
0.80       0.5032   0.6681   0.0064   1.0000   1.0000   0.5016  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5032, F1=0.6681, Normal Recall=0.0064, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5016

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
0.15       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006   <--
0.20       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.25       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.30       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.35       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.40       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.45       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.50       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.55       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.60       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.65       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.70       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.75       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.80       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.1058, F1=0.1828, Normal Recall=0.0065, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1006

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
0.15       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010   <--
0.20       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.25       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.30       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.35       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.40       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.45       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.50       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.55       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.60       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.65       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.70       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.75       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.80       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2051, F1=0.3348, Normal Recall=0.0064, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.2010

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
0.15       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014   <--
0.20       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.25       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.30       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.35       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.40       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.45       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.50       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.55       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.60       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.65       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.70       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.75       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.80       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3046, F1=0.4632, Normal Recall=0.0065, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3014

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
0.15       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016   <--
0.20       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.25       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.30       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.35       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.40       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.45       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.50       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.55       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.60       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.65       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.70       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.75       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.80       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4040, F1=0.5731, Normal Recall=0.0067, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4016

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
0.15       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017   <--
0.20       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.25       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.30       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.35       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.40       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.45       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.50       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.55       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.60       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.65       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.70       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.75       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.80       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5033, F1=0.6681, Normal Recall=0.0066, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5017

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
0.15       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006   <--
0.20       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.25       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.30       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.35       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.40       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.45       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.50       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.55       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.60       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.65       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.70       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.75       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
0.80       0.1058   0.1828   0.0065   1.0000   1.0000   0.1006  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.1058, F1=0.1828, Normal Recall=0.0065, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1006

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
0.15       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010   <--
0.20       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.25       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.30       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.35       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.40       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.45       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.50       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.55       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.60       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.65       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.70       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.75       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
0.80       0.2051   0.3348   0.0064   1.0000   1.0000   0.2010  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2051, F1=0.3348, Normal Recall=0.0064, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.2010

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
0.15       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014   <--
0.20       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.25       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.30       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.35       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.40       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.45       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.50       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.55       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.60       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.65       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.70       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.75       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
0.80       0.3046   0.4632   0.0065   1.0000   1.0000   0.3014  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3046, F1=0.4632, Normal Recall=0.0065, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3014

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
0.15       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016   <--
0.20       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.25       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.30       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.35       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.40       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.45       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.50       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.55       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.60       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.65       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.70       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.75       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
0.80       0.4040   0.5731   0.0067   1.0000   1.0000   0.4016  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4040, F1=0.5731, Normal Recall=0.0067, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4016

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
0.15       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017   <--
0.20       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.25       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.30       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.35       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.40       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.45       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.50       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.55       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.60       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.65       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.70       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.75       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
0.80       0.5033   0.6681   0.0066   1.0000   1.0000   0.5017  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5033, F1=0.6681, Normal Recall=0.0066, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5017

```

