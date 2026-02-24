# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e1_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-19 19:59:17 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4541 | 0.4548 | 0.4573 | 0.4588 | 0.4603 | 0.4616 | 0.4625 | 0.4645 | 0.4678 | 0.4692 | 0.4706 |
| QAT+Prune only | 0.6967 | 0.7277 | 0.7572 | 0.7870 | 0.8174 | 0.8470 | 0.8773 | 0.9078 | 0.9381 | 0.9674 | 0.9981 |
| QAT+PTQ | 0.6971 | 0.7282 | 0.7576 | 0.7875 | 0.8178 | 0.8472 | 0.8776 | 0.9080 | 0.9381 | 0.9674 | 0.9981 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6971 | 0.7282 | 0.7576 | 0.7875 | 0.8178 | 0.8472 | 0.8776 | 0.9080 | 0.9381 | 0.9674 | 0.9981 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1471 | 0.2575 | 0.3429 | 0.4109 | 0.4664 | 0.5123 | 0.5517 | 0.5859 | 0.6148 | 0.6400 |
| QAT+Prune only | 0.0000 | 0.4230 | 0.6218 | 0.7376 | 0.8139 | 0.8671 | 0.9071 | 0.9381 | 0.9627 | 0.9822 | 0.9991 |
| QAT+PTQ | 0.0000 | 0.4235 | 0.6223 | 0.7381 | 0.8142 | 0.8673 | 0.9073 | 0.9382 | 0.9627 | 0.9822 | 0.9991 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4235 | 0.6223 | 0.7381 | 0.8142 | 0.8673 | 0.9073 | 0.9382 | 0.9627 | 0.9822 | 0.9991 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4541 | 0.4532 | 0.4540 | 0.4538 | 0.4534 | 0.4526 | 0.4503 | 0.4504 | 0.4564 | 0.4566 | 0.0000 |
| QAT+Prune only | 0.6967 | 0.6977 | 0.6969 | 0.6965 | 0.6969 | 0.6958 | 0.6960 | 0.6969 | 0.6977 | 0.6906 | 0.0000 |
| QAT+PTQ | 0.6971 | 0.6983 | 0.6975 | 0.6972 | 0.6976 | 0.6963 | 0.6968 | 0.6977 | 0.6977 | 0.6906 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6971 | 0.6983 | 0.6975 | 0.6972 | 0.6976 | 0.6963 | 0.6968 | 0.6977 | 0.6977 | 0.6906 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4541 | 0.0000 | 0.0000 | 0.0000 | 0.4541 | 1.0000 |
| 90 | 10 | 299,940 | 0.4548 | 0.0872 | 0.4701 | 0.1471 | 0.4532 | 0.8850 |
| 80 | 20 | 291,350 | 0.4573 | 0.1773 | 0.4706 | 0.2575 | 0.4540 | 0.7743 |
| 70 | 30 | 194,230 | 0.4588 | 0.2697 | 0.4706 | 0.3429 | 0.4538 | 0.6667 |
| 60 | 40 | 145,675 | 0.4603 | 0.3647 | 0.4706 | 0.4109 | 0.4534 | 0.5623 |
| 50 | 50 | 116,540 | 0.4616 | 0.4623 | 0.4706 | 0.4664 | 0.4526 | 0.4609 |
| 40 | 60 | 97,115 | 0.4625 | 0.5622 | 0.4706 | 0.5123 | 0.4503 | 0.3619 |
| 30 | 70 | 83,240 | 0.4645 | 0.6664 | 0.4706 | 0.5517 | 0.4504 | 0.2672 |
| 20 | 80 | 72,835 | 0.4678 | 0.7759 | 0.4706 | 0.5859 | 0.4564 | 0.1773 |
| 10 | 90 | 64,740 | 0.4692 | 0.8863 | 0.4706 | 0.6148 | 0.4566 | 0.0874 |
| 0 | 100 | 58,270 | 0.4706 | 1.0000 | 0.4706 | 0.6400 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6967 | 0.0000 | 0.0000 | 0.0000 | 0.6967 | 1.0000 |
| 90 | 10 | 299,940 | 0.7277 | 0.2684 | 0.9981 | 0.4230 | 0.6977 | 0.9997 |
| 80 | 20 | 291,350 | 0.7572 | 0.4516 | 0.9981 | 0.6218 | 0.6969 | 0.9993 |
| 70 | 30 | 194,230 | 0.7870 | 0.5850 | 0.9981 | 0.7376 | 0.6965 | 0.9989 |
| 60 | 40 | 145,675 | 0.8174 | 0.6870 | 0.9981 | 0.8139 | 0.6969 | 0.9982 |
| 50 | 50 | 116,540 | 0.8470 | 0.7664 | 0.9981 | 0.8671 | 0.6958 | 0.9973 |
| 40 | 60 | 97,115 | 0.8773 | 0.8312 | 0.9981 | 0.9071 | 0.6960 | 0.9960 |
| 30 | 70 | 83,240 | 0.9078 | 0.8849 | 0.9981 | 0.9381 | 0.6969 | 0.9938 |
| 20 | 80 | 72,835 | 0.9381 | 0.9296 | 0.9981 | 0.9627 | 0.6977 | 0.9895 |
| 10 | 90 | 64,740 | 0.9674 | 0.9667 | 0.9981 | 0.9822 | 0.6906 | 0.9764 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9991 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6971 | 0.0000 | 0.0000 | 0.0000 | 0.6971 | 1.0000 |
| 90 | 10 | 299,940 | 0.7282 | 0.2688 | 0.9981 | 0.4235 | 0.6983 | 0.9997 |
| 80 | 20 | 291,350 | 0.7576 | 0.4520 | 0.9981 | 0.6223 | 0.6975 | 0.9993 |
| 70 | 30 | 194,230 | 0.7875 | 0.5855 | 0.9981 | 0.7381 | 0.6972 | 0.9989 |
| 60 | 40 | 145,675 | 0.8178 | 0.6876 | 0.9981 | 0.8142 | 0.6976 | 0.9982 |
| 50 | 50 | 116,540 | 0.8472 | 0.7667 | 0.9981 | 0.8673 | 0.6963 | 0.9973 |
| 40 | 60 | 97,115 | 0.8776 | 0.8316 | 0.9981 | 0.9073 | 0.6968 | 0.9960 |
| 30 | 70 | 83,240 | 0.9080 | 0.8851 | 0.9981 | 0.9382 | 0.6977 | 0.9938 |
| 20 | 80 | 72,835 | 0.9381 | 0.9296 | 0.9981 | 0.9627 | 0.6977 | 0.9895 |
| 10 | 90 | 64,740 | 0.9674 | 0.9667 | 0.9981 | 0.9822 | 0.6906 | 0.9764 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9991 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6971 | 0.0000 | 0.0000 | 0.0000 | 0.6971 | 1.0000 |
| 90 | 10 | 299,940 | 0.7282 | 0.2688 | 0.9981 | 0.4235 | 0.6983 | 0.9997 |
| 80 | 20 | 291,350 | 0.7576 | 0.4520 | 0.9981 | 0.6223 | 0.6975 | 0.9993 |
| 70 | 30 | 194,230 | 0.7875 | 0.5855 | 0.9981 | 0.7381 | 0.6972 | 0.9989 |
| 60 | 40 | 145,675 | 0.8178 | 0.6876 | 0.9981 | 0.8142 | 0.6976 | 0.9982 |
| 50 | 50 | 116,540 | 0.8472 | 0.7667 | 0.9981 | 0.8673 | 0.6963 | 0.9973 |
| 40 | 60 | 97,115 | 0.8776 | 0.8316 | 0.9981 | 0.9073 | 0.6968 | 0.9960 |
| 30 | 70 | 83,240 | 0.9080 | 0.8851 | 0.9981 | 0.9382 | 0.6977 | 0.9938 |
| 20 | 80 | 72,835 | 0.9381 | 0.9296 | 0.9981 | 0.9627 | 0.6977 | 0.9895 |
| 10 | 90 | 64,740 | 0.9674 | 0.9667 | 0.9981 | 0.9822 | 0.6906 | 0.9764 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9991 | 0.0000 | 0.0000 |


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
0.15       0.4548   0.1470   0.4531   0.8850   0.4698   0.0871   <--
0.20       0.4548   0.1470   0.4531   0.8850   0.4698   0.0871  
0.25       0.4548   0.1470   0.4531   0.8850   0.4698   0.0871  
0.30       0.4548   0.1470   0.4531   0.8850   0.4698   0.0871  
0.35       0.4548   0.1470   0.4531   0.8850   0.4698   0.0871  
0.40       0.4548   0.1470   0.4531   0.8850   0.4698   0.0871  
0.45       0.4548   0.1470   0.4531   0.8850   0.4698   0.0871  
0.50       0.4548   0.1470   0.4531   0.8850   0.4698   0.0871  
0.55       0.4548   0.1470   0.4531   0.8850   0.4698   0.0871  
0.60       0.4548   0.1470   0.4531   0.8850   0.4698   0.0871  
0.65       0.4548   0.1470   0.4531   0.8850   0.4698   0.0871  
0.70       0.4548   0.1470   0.4531   0.8850   0.4698   0.0871  
0.75       0.4548   0.1470   0.4531   0.8850   0.4698   0.0871  
0.80       0.4548   0.1470   0.4531   0.8850   0.4698   0.0871  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4548, F1=0.1470, Normal Recall=0.4531, Normal Precision=0.8850, Attack Recall=0.4698, Attack Precision=0.0871

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
0.15       0.4560   0.2571   0.4524   0.7736   0.4706   0.1768   <--
0.20       0.4560   0.2571   0.4524   0.7736   0.4706   0.1768  
0.25       0.4560   0.2571   0.4524   0.7736   0.4706   0.1768  
0.30       0.4560   0.2571   0.4524   0.7736   0.4706   0.1768  
0.35       0.4560   0.2571   0.4524   0.7736   0.4706   0.1768  
0.40       0.4560   0.2571   0.4524   0.7736   0.4706   0.1768  
0.45       0.4560   0.2571   0.4524   0.7736   0.4706   0.1768  
0.50       0.4560   0.2571   0.4524   0.7736   0.4706   0.1768  
0.55       0.4560   0.2571   0.4524   0.7736   0.4706   0.1768  
0.60       0.4560   0.2571   0.4524   0.7736   0.4706   0.1768  
0.65       0.4560   0.2571   0.4524   0.7736   0.4706   0.1768  
0.70       0.4560   0.2571   0.4524   0.7736   0.4706   0.1768  
0.75       0.4560   0.2571   0.4524   0.7736   0.4706   0.1768  
0.80       0.4560   0.2571   0.4524   0.7736   0.4706   0.1768  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4560, F1=0.2571, Normal Recall=0.4524, Normal Precision=0.7736, Attack Recall=0.4706, Attack Precision=0.1768

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
0.15       0.4581   0.3426   0.4528   0.6662   0.4706   0.2693   <--
0.20       0.4581   0.3426   0.4528   0.6662   0.4706   0.2693  
0.25       0.4581   0.3426   0.4528   0.6662   0.4706   0.2693  
0.30       0.4581   0.3426   0.4528   0.6662   0.4706   0.2693  
0.35       0.4581   0.3426   0.4528   0.6662   0.4706   0.2693  
0.40       0.4581   0.3426   0.4528   0.6662   0.4706   0.2693  
0.45       0.4581   0.3426   0.4528   0.6662   0.4706   0.2693  
0.50       0.4581   0.3426   0.4528   0.6662   0.4706   0.2693  
0.55       0.4581   0.3426   0.4528   0.6662   0.4706   0.2693  
0.60       0.4581   0.3426   0.4528   0.6662   0.4706   0.2693  
0.65       0.4581   0.3426   0.4528   0.6662   0.4706   0.2693  
0.70       0.4581   0.3426   0.4528   0.6662   0.4706   0.2693  
0.75       0.4581   0.3426   0.4528   0.6662   0.4706   0.2693  
0.80       0.4581   0.3426   0.4528   0.6662   0.4706   0.2693  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4581, F1=0.3426, Normal Recall=0.4528, Normal Precision=0.6662, Attack Recall=0.4706, Attack Precision=0.2693

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
0.15       0.4610   0.4112   0.4546   0.5630   0.4706   0.3652   <--
0.20       0.4610   0.4112   0.4546   0.5630   0.4706   0.3652  
0.25       0.4610   0.4112   0.4546   0.5630   0.4706   0.3652  
0.30       0.4610   0.4112   0.4546   0.5630   0.4706   0.3652  
0.35       0.4610   0.4112   0.4546   0.5630   0.4706   0.3652  
0.40       0.4610   0.4112   0.4546   0.5630   0.4706   0.3652  
0.45       0.4610   0.4112   0.4546   0.5630   0.4706   0.3652  
0.50       0.4610   0.4112   0.4546   0.5630   0.4706   0.3652  
0.55       0.4610   0.4112   0.4546   0.5630   0.4706   0.3652  
0.60       0.4610   0.4112   0.4546   0.5630   0.4706   0.3652  
0.65       0.4610   0.4112   0.4546   0.5630   0.4706   0.3652  
0.70       0.4610   0.4112   0.4546   0.5630   0.4706   0.3652  
0.75       0.4610   0.4112   0.4546   0.5630   0.4706   0.3652  
0.80       0.4610   0.4112   0.4546   0.5630   0.4706   0.3652  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4610, F1=0.4112, Normal Recall=0.4546, Normal Precision=0.5630, Attack Recall=0.4706, Attack Precision=0.3652

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
0.15       0.4626   0.4669   0.4547   0.4620   0.4706   0.4632   <--
0.20       0.4626   0.4669   0.4547   0.4620   0.4706   0.4632  
0.25       0.4626   0.4669   0.4547   0.4620   0.4706   0.4632  
0.30       0.4626   0.4669   0.4547   0.4620   0.4706   0.4632  
0.35       0.4626   0.4669   0.4547   0.4620   0.4706   0.4632  
0.40       0.4626   0.4669   0.4547   0.4620   0.4706   0.4632  
0.45       0.4626   0.4669   0.4547   0.4620   0.4706   0.4632  
0.50       0.4626   0.4669   0.4547   0.4620   0.4706   0.4632  
0.55       0.4626   0.4669   0.4547   0.4620   0.4706   0.4632  
0.60       0.4626   0.4669   0.4547   0.4620   0.4706   0.4632  
0.65       0.4626   0.4669   0.4547   0.4620   0.4706   0.4632  
0.70       0.4626   0.4669   0.4547   0.4620   0.4706   0.4632  
0.75       0.4626   0.4669   0.4547   0.4620   0.4706   0.4632  
0.80       0.4626   0.4669   0.4547   0.4620   0.4706   0.4632  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4626, F1=0.4669, Normal Recall=0.4547, Normal Precision=0.4620, Attack Recall=0.4706, Attack Precision=0.4632

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
0.15       0.7277   0.4231   0.6977   0.9997   0.9983   0.2684   <--
0.20       0.7277   0.4231   0.6977   0.9997   0.9983   0.2684  
0.25       0.7277   0.4231   0.6977   0.9997   0.9983   0.2684  
0.30       0.7277   0.4231   0.6977   0.9997   0.9983   0.2684  
0.35       0.7277   0.4231   0.6977   0.9997   0.9983   0.2684  
0.40       0.7277   0.4231   0.6977   0.9997   0.9983   0.2684  
0.45       0.7277   0.4231   0.6977   0.9997   0.9983   0.2684  
0.50       0.7277   0.4231   0.6977   0.9997   0.9983   0.2684  
0.55       0.7277   0.4231   0.6977   0.9997   0.9983   0.2684  
0.60       0.7277   0.4231   0.6977   0.9997   0.9983   0.2684  
0.65       0.7277   0.4231   0.6977   0.9997   0.9983   0.2684  
0.70       0.7277   0.4231   0.6977   0.9997   0.9983   0.2684  
0.75       0.7277   0.4231   0.6977   0.9997   0.9983   0.2684  
0.80       0.7277   0.4231   0.6977   0.9997   0.9983   0.2684  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7277, F1=0.4231, Normal Recall=0.6977, Normal Precision=0.9997, Attack Recall=0.9983, Attack Precision=0.2684

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
0.15       0.7583   0.6229   0.6983   0.9993   0.9981   0.4527   <--
0.20       0.7583   0.6229   0.6983   0.9993   0.9981   0.4527  
0.25       0.7583   0.6229   0.6983   0.9993   0.9981   0.4527  
0.30       0.7583   0.6229   0.6983   0.9993   0.9981   0.4527  
0.35       0.7583   0.6229   0.6983   0.9993   0.9981   0.4527  
0.40       0.7583   0.6229   0.6983   0.9993   0.9981   0.4527  
0.45       0.7583   0.6229   0.6983   0.9993   0.9981   0.4527  
0.50       0.7583   0.6229   0.6983   0.9993   0.9981   0.4527  
0.55       0.7583   0.6229   0.6983   0.9993   0.9981   0.4527  
0.60       0.7583   0.6229   0.6983   0.9993   0.9981   0.4527  
0.65       0.7583   0.6229   0.6983   0.9993   0.9981   0.4527  
0.70       0.7583   0.6229   0.6983   0.9993   0.9981   0.4527  
0.75       0.7583   0.6229   0.6983   0.9993   0.9981   0.4527  
0.80       0.7583   0.6229   0.6983   0.9993   0.9981   0.4527  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7583, F1=0.6229, Normal Recall=0.6983, Normal Precision=0.9993, Attack Recall=0.9981, Attack Precision=0.4527

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
0.15       0.7877   0.7383   0.6975   0.9989   0.9981   0.5857   <--
0.20       0.7877   0.7383   0.6975   0.9989   0.9981   0.5857  
0.25       0.7877   0.7383   0.6975   0.9989   0.9981   0.5857  
0.30       0.7877   0.7383   0.6975   0.9989   0.9981   0.5857  
0.35       0.7877   0.7383   0.6975   0.9989   0.9981   0.5857  
0.40       0.7877   0.7383   0.6975   0.9989   0.9981   0.5857  
0.45       0.7877   0.7383   0.6975   0.9989   0.9981   0.5857  
0.50       0.7877   0.7383   0.6975   0.9989   0.9981   0.5857  
0.55       0.7877   0.7383   0.6975   0.9989   0.9981   0.5857  
0.60       0.7877   0.7383   0.6975   0.9989   0.9981   0.5857  
0.65       0.7877   0.7383   0.6975   0.9989   0.9981   0.5857  
0.70       0.7877   0.7383   0.6975   0.9989   0.9981   0.5857  
0.75       0.7877   0.7383   0.6975   0.9989   0.9981   0.5857  
0.80       0.7877   0.7383   0.6975   0.9989   0.9981   0.5857  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7877, F1=0.7383, Normal Recall=0.6975, Normal Precision=0.9989, Attack Recall=0.9981, Attack Precision=0.5857

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
0.15       0.8172   0.8137   0.6965   0.9982   0.9981   0.6868   <--
0.20       0.8172   0.8137   0.6965   0.9982   0.9981   0.6868  
0.25       0.8172   0.8137   0.6965   0.9982   0.9981   0.6868  
0.30       0.8172   0.8137   0.6965   0.9982   0.9981   0.6868  
0.35       0.8172   0.8137   0.6965   0.9982   0.9981   0.6868  
0.40       0.8172   0.8137   0.6965   0.9982   0.9981   0.6868  
0.45       0.8172   0.8137   0.6965   0.9982   0.9981   0.6868  
0.50       0.8172   0.8137   0.6965   0.9982   0.9981   0.6868  
0.55       0.8172   0.8137   0.6965   0.9982   0.9981   0.6868  
0.60       0.8172   0.8137   0.6965   0.9982   0.9981   0.6868  
0.65       0.8172   0.8137   0.6965   0.9982   0.9981   0.6868  
0.70       0.8172   0.8137   0.6965   0.9982   0.9981   0.6868  
0.75       0.8172   0.8137   0.6965   0.9982   0.9981   0.6868  
0.80       0.8172   0.8137   0.6965   0.9982   0.9981   0.6868  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8172, F1=0.8137, Normal Recall=0.6965, Normal Precision=0.9982, Attack Recall=0.9981, Attack Precision=0.6868

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
0.15       0.8466   0.8668   0.6950   0.9973   0.9981   0.7660   <--
0.20       0.8466   0.8668   0.6950   0.9973   0.9981   0.7660  
0.25       0.8466   0.8668   0.6950   0.9973   0.9981   0.7660  
0.30       0.8466   0.8668   0.6950   0.9973   0.9981   0.7660  
0.35       0.8466   0.8668   0.6950   0.9973   0.9981   0.7660  
0.40       0.8466   0.8668   0.6950   0.9973   0.9981   0.7660  
0.45       0.8466   0.8668   0.6950   0.9973   0.9981   0.7660  
0.50       0.8466   0.8668   0.6950   0.9973   0.9981   0.7660  
0.55       0.8466   0.8668   0.6950   0.9973   0.9981   0.7660  
0.60       0.8466   0.8668   0.6950   0.9973   0.9981   0.7660  
0.65       0.8466   0.8668   0.6950   0.9973   0.9981   0.7660  
0.70       0.8466   0.8668   0.6950   0.9973   0.9981   0.7660  
0.75       0.8466   0.8668   0.6950   0.9973   0.9981   0.7660  
0.80       0.8466   0.8668   0.6950   0.9973   0.9981   0.7660  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8466, F1=0.8668, Normal Recall=0.6950, Normal Precision=0.9973, Attack Recall=0.9981, Attack Precision=0.7660

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
0.15       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688   <--
0.20       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.25       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.30       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.35       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.40       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.45       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.50       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.55       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.60       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.65       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.70       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.75       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.80       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7283, F1=0.4236, Normal Recall=0.6983, Normal Precision=0.9997, Attack Recall=0.9983, Attack Precision=0.2688

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
0.15       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532   <--
0.20       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.25       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.30       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.35       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.40       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.45       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.50       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.55       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.60       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.65       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.70       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.75       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.80       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7588, F1=0.6234, Normal Recall=0.6989, Normal Precision=0.9993, Attack Recall=0.9981, Attack Precision=0.4532

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
0.15       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861   <--
0.20       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.25       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.30       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.35       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.40       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.45       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.50       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.55       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.60       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.65       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.70       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.75       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.80       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7880, F1=0.7385, Normal Recall=0.6979, Normal Precision=0.9989, Attack Recall=0.9981, Attack Precision=0.5861

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
0.15       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871   <--
0.20       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.25       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.30       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.35       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.40       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.45       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.50       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.55       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.60       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.65       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.70       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.75       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.80       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8174, F1=0.8139, Normal Recall=0.6969, Normal Precision=0.9982, Attack Recall=0.9981, Attack Precision=0.6871

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
0.15       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663   <--
0.20       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.25       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.30       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.35       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.40       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.45       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.50       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.55       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.60       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.65       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.70       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.75       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.80       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8468, F1=0.8670, Normal Recall=0.6955, Normal Precision=0.9973, Attack Recall=0.9981, Attack Precision=0.7663

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
0.15       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688   <--
0.20       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.25       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.30       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.35       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.40       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.45       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.50       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.55       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.60       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.65       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.70       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.75       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
0.80       0.7283   0.4236   0.6983   0.9997   0.9983   0.2688  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7283, F1=0.4236, Normal Recall=0.6983, Normal Precision=0.9997, Attack Recall=0.9983, Attack Precision=0.2688

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
0.15       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532   <--
0.20       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.25       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.30       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.35       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.40       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.45       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.50       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.55       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.60       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.65       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.70       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.75       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
0.80       0.7588   0.6234   0.6989   0.9993   0.9981   0.4532  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7588, F1=0.6234, Normal Recall=0.6989, Normal Precision=0.9993, Attack Recall=0.9981, Attack Precision=0.4532

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
0.15       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861   <--
0.20       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.25       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.30       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.35       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.40       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.45       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.50       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.55       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.60       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.65       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.70       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.75       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
0.80       0.7880   0.7385   0.6979   0.9989   0.9981   0.5861  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7880, F1=0.7385, Normal Recall=0.6979, Normal Precision=0.9989, Attack Recall=0.9981, Attack Precision=0.5861

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
0.15       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871   <--
0.20       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.25       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.30       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.35       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.40       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.45       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.50       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.55       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.60       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.65       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.70       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.75       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
0.80       0.8174   0.8139   0.6969   0.9982   0.9981   0.6871  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8174, F1=0.8139, Normal Recall=0.6969, Normal Precision=0.9982, Attack Recall=0.9981, Attack Precision=0.6871

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
0.15       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663   <--
0.20       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.25       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.30       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.35       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.40       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.45       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.50       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.55       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.60       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.65       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.70       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.75       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
0.80       0.8468   0.8670   0.6955   0.9973   0.9981   0.7663  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8468, F1=0.8670, Normal Recall=0.6955, Normal Precision=0.9973, Attack Recall=0.9981, Attack Precision=0.7663

```

