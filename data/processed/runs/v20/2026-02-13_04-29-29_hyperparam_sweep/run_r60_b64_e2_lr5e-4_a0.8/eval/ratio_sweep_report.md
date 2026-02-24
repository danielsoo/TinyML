# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e2_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-14 01:03:38 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7326 | 0.6979 | 0.6634 | 0.6286 | 0.5933 | 0.5571 | 0.5234 | 0.4871 | 0.4528 | 0.4179 | 0.3830 |
| QAT+Prune only | 0.9165 | 0.9224 | 0.9283 | 0.9352 | 0.9408 | 0.9469 | 0.9538 | 0.9604 | 0.9663 | 0.9726 | 0.9792 |
| QAT+PTQ | 0.9162 | 0.9220 | 0.9279 | 0.9349 | 0.9404 | 0.9465 | 0.9534 | 0.9601 | 0.9660 | 0.9722 | 0.9788 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9162 | 0.9220 | 0.9279 | 0.9349 | 0.9404 | 0.9465 | 0.9534 | 0.9601 | 0.9660 | 0.9722 | 0.9788 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2005 | 0.3127 | 0.3822 | 0.4297 | 0.4637 | 0.4909 | 0.5111 | 0.5283 | 0.5422 | 0.5538 |
| QAT+Prune only | 0.0000 | 0.7162 | 0.8453 | 0.9006 | 0.9298 | 0.9485 | 0.9621 | 0.9720 | 0.9790 | 0.9847 | 0.9895 |
| QAT+PTQ | 0.0000 | 0.7151 | 0.8445 | 0.9002 | 0.9293 | 0.9482 | 0.9618 | 0.9717 | 0.9788 | 0.9845 | 0.9893 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.7151 | 0.8445 | 0.9002 | 0.9293 | 0.9482 | 0.9618 | 0.9717 | 0.9788 | 0.9845 | 0.9893 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7326 | 0.7334 | 0.7335 | 0.7339 | 0.7336 | 0.7313 | 0.7341 | 0.7302 | 0.7323 | 0.7326 | 0.0000 |
| QAT+Prune only | 0.9165 | 0.9160 | 0.9156 | 0.9163 | 0.9153 | 0.9145 | 0.9156 | 0.9166 | 0.9149 | 0.9126 | 0.0000 |
| QAT+PTQ | 0.9162 | 0.9156 | 0.9152 | 0.9161 | 0.9149 | 0.9142 | 0.9152 | 0.9165 | 0.9147 | 0.9123 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9162 | 0.9156 | 0.9152 | 0.9161 | 0.9149 | 0.9142 | 0.9152 | 0.9165 | 0.9147 | 0.9123 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7326 | 0.0000 | 0.0000 | 0.0000 | 0.7326 | 1.0000 |
| 90 | 10 | 299,940 | 0.6979 | 0.1364 | 0.3788 | 0.2005 | 0.7334 | 0.9140 |
| 80 | 20 | 291,350 | 0.6634 | 0.2643 | 0.3830 | 0.3127 | 0.7335 | 0.8262 |
| 70 | 30 | 194,230 | 0.6286 | 0.3815 | 0.3829 | 0.3822 | 0.7339 | 0.7351 |
| 60 | 40 | 145,675 | 0.5933 | 0.4893 | 0.3830 | 0.4297 | 0.7336 | 0.6407 |
| 50 | 50 | 116,540 | 0.5571 | 0.5876 | 0.3830 | 0.4637 | 0.7313 | 0.5424 |
| 40 | 60 | 97,115 | 0.5234 | 0.6836 | 0.3830 | 0.4909 | 0.7341 | 0.4423 |
| 30 | 70 | 83,240 | 0.4871 | 0.7681 | 0.3830 | 0.5111 | 0.7302 | 0.3365 |
| 20 | 80 | 72,835 | 0.4528 | 0.8512 | 0.3830 | 0.5283 | 0.7323 | 0.2288 |
| 10 | 90 | 64,740 | 0.4179 | 0.9280 | 0.3830 | 0.5422 | 0.7326 | 0.1166 |
| 0 | 100 | 58,270 | 0.3830 | 1.0000 | 0.3830 | 0.5538 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9165 | 0.0000 | 0.0000 | 0.0000 | 0.9165 | 1.0000 |
| 90 | 10 | 299,940 | 0.9224 | 0.5645 | 0.9794 | 0.7162 | 0.9160 | 0.9975 |
| 80 | 20 | 291,350 | 0.9283 | 0.7436 | 0.9792 | 0.8453 | 0.9156 | 0.9944 |
| 70 | 30 | 194,230 | 0.9352 | 0.8337 | 0.9792 | 0.9006 | 0.9163 | 0.9904 |
| 60 | 40 | 145,675 | 0.9408 | 0.8851 | 0.9792 | 0.9298 | 0.9153 | 0.9851 |
| 50 | 50 | 116,540 | 0.9469 | 0.9197 | 0.9792 | 0.9485 | 0.9145 | 0.9778 |
| 40 | 60 | 97,115 | 0.9538 | 0.9456 | 0.9792 | 0.9621 | 0.9156 | 0.9671 |
| 30 | 70 | 83,240 | 0.9604 | 0.9648 | 0.9792 | 0.9720 | 0.9166 | 0.9498 |
| 20 | 80 | 72,835 | 0.9663 | 0.9787 | 0.9792 | 0.9790 | 0.9149 | 0.9167 |
| 10 | 90 | 64,740 | 0.9726 | 0.9902 | 0.9792 | 0.9847 | 0.9126 | 0.8300 |
| 0 | 100 | 58,270 | 0.9792 | 1.0000 | 0.9792 | 0.9895 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9162 | 0.0000 | 0.0000 | 0.0000 | 0.9162 | 1.0000 |
| 90 | 10 | 299,940 | 0.9220 | 0.5632 | 0.9790 | 0.7151 | 0.9156 | 0.9975 |
| 80 | 20 | 291,350 | 0.9279 | 0.7426 | 0.9788 | 0.8445 | 0.9152 | 0.9942 |
| 70 | 30 | 194,230 | 0.9349 | 0.8333 | 0.9788 | 0.9002 | 0.9161 | 0.9902 |
| 60 | 40 | 145,675 | 0.9404 | 0.8846 | 0.9788 | 0.9293 | 0.9149 | 0.9848 |
| 50 | 50 | 116,540 | 0.9465 | 0.9194 | 0.9788 | 0.9482 | 0.9142 | 0.9774 |
| 40 | 60 | 97,115 | 0.9534 | 0.9454 | 0.9788 | 0.9618 | 0.9152 | 0.9665 |
| 30 | 70 | 83,240 | 0.9601 | 0.9647 | 0.9788 | 0.9717 | 0.9165 | 0.9488 |
| 20 | 80 | 72,835 | 0.9660 | 0.9787 | 0.9788 | 0.9788 | 0.9147 | 0.9152 |
| 10 | 90 | 64,740 | 0.9722 | 0.9901 | 0.9788 | 0.9845 | 0.9123 | 0.8273 |
| 0 | 100 | 58,270 | 0.9788 | 1.0000 | 0.9788 | 0.9893 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9162 | 0.0000 | 0.0000 | 0.0000 | 0.9162 | 1.0000 |
| 90 | 10 | 299,940 | 0.9220 | 0.5632 | 0.9790 | 0.7151 | 0.9156 | 0.9975 |
| 80 | 20 | 291,350 | 0.9279 | 0.7426 | 0.9788 | 0.8445 | 0.9152 | 0.9942 |
| 70 | 30 | 194,230 | 0.9349 | 0.8333 | 0.9788 | 0.9002 | 0.9161 | 0.9902 |
| 60 | 40 | 145,675 | 0.9404 | 0.8846 | 0.9788 | 0.9293 | 0.9149 | 0.9848 |
| 50 | 50 | 116,540 | 0.9465 | 0.9194 | 0.9788 | 0.9482 | 0.9142 | 0.9774 |
| 40 | 60 | 97,115 | 0.9534 | 0.9454 | 0.9788 | 0.9618 | 0.9152 | 0.9665 |
| 30 | 70 | 83,240 | 0.9601 | 0.9647 | 0.9788 | 0.9717 | 0.9165 | 0.9488 |
| 20 | 80 | 72,835 | 0.9660 | 0.9787 | 0.9788 | 0.9788 | 0.9147 | 0.9152 |
| 10 | 90 | 64,740 | 0.9722 | 0.9901 | 0.9788 | 0.9845 | 0.9123 | 0.8273 |
| 0 | 100 | 58,270 | 0.9788 | 1.0000 | 0.9788 | 0.9893 | 0.0000 | 0.0000 |


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
0.15       0.6983   0.2024   0.7334   0.9145   0.3828   0.1376   <--
0.20       0.6983   0.2024   0.7334   0.9145   0.3828   0.1376  
0.25       0.6983   0.2024   0.7334   0.9145   0.3828   0.1376  
0.30       0.6983   0.2024   0.7334   0.9145   0.3828   0.1376  
0.35       0.6983   0.2024   0.7334   0.9145   0.3828   0.1376  
0.40       0.6983   0.2024   0.7334   0.9145   0.3828   0.1376  
0.45       0.6983   0.2024   0.7334   0.9145   0.3828   0.1376  
0.50       0.6983   0.2024   0.7334   0.9145   0.3828   0.1376  
0.55       0.6983   0.2024   0.7334   0.9145   0.3828   0.1376  
0.60       0.6983   0.2024   0.7334   0.9145   0.3828   0.1376  
0.65       0.6983   0.2024   0.7334   0.9145   0.3828   0.1376  
0.70       0.6983   0.2024   0.7334   0.9145   0.3828   0.1376  
0.75       0.6983   0.2024   0.7334   0.9145   0.3828   0.1376  
0.80       0.6983   0.2024   0.7334   0.9145   0.3828   0.1376  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6983, F1=0.2024, Normal Recall=0.7334, Normal Precision=0.9145, Attack Recall=0.3828, Attack Precision=0.1376

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
0.15       0.6632   0.3126   0.7333   0.8262   0.3830   0.2641   <--
0.20       0.6632   0.3126   0.7333   0.8262   0.3830   0.2641  
0.25       0.6632   0.3126   0.7333   0.8262   0.3830   0.2641  
0.30       0.6632   0.3126   0.7333   0.8262   0.3830   0.2641  
0.35       0.6632   0.3126   0.7333   0.8262   0.3830   0.2641  
0.40       0.6632   0.3126   0.7333   0.8262   0.3830   0.2641  
0.45       0.6632   0.3126   0.7333   0.8262   0.3830   0.2641  
0.50       0.6632   0.3126   0.7333   0.8262   0.3830   0.2641  
0.55       0.6632   0.3126   0.7333   0.8262   0.3830   0.2641  
0.60       0.6632   0.3126   0.7333   0.8262   0.3830   0.2641  
0.65       0.6632   0.3126   0.7333   0.8262   0.3830   0.2641  
0.70       0.6632   0.3126   0.7333   0.8262   0.3830   0.2641  
0.75       0.6632   0.3126   0.7333   0.8262   0.3830   0.2641  
0.80       0.6632   0.3126   0.7333   0.8262   0.3830   0.2641  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6632, F1=0.3126, Normal Recall=0.7333, Normal Precision=0.8262, Attack Recall=0.3830, Attack Precision=0.2641

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
0.15       0.6283   0.3821   0.7335   0.7350   0.3830   0.3811   <--
0.20       0.6283   0.3821   0.7335   0.7350   0.3830   0.3811  
0.25       0.6283   0.3821   0.7335   0.7350   0.3830   0.3811  
0.30       0.6283   0.3821   0.7335   0.7350   0.3830   0.3811  
0.35       0.6283   0.3821   0.7335   0.7350   0.3830   0.3811  
0.40       0.6283   0.3821   0.7335   0.7350   0.3830   0.3811  
0.45       0.6283   0.3821   0.7335   0.7350   0.3830   0.3811  
0.50       0.6283   0.3821   0.7335   0.7350   0.3830   0.3811  
0.55       0.6283   0.3821   0.7335   0.7350   0.3830   0.3811  
0.60       0.6283   0.3821   0.7335   0.7350   0.3830   0.3811  
0.65       0.6283   0.3821   0.7335   0.7350   0.3830   0.3811  
0.70       0.6283   0.3821   0.7335   0.7350   0.3830   0.3811  
0.75       0.6283   0.3821   0.7335   0.7350   0.3830   0.3811  
0.80       0.6283   0.3821   0.7335   0.7350   0.3830   0.3811  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6283, F1=0.3821, Normal Recall=0.7335, Normal Precision=0.7350, Attack Recall=0.3830, Attack Precision=0.3811

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
0.15       0.5928   0.4293   0.7326   0.6404   0.3830   0.4885   <--
0.20       0.5928   0.4293   0.7326   0.6404   0.3830   0.4885  
0.25       0.5928   0.4293   0.7326   0.6404   0.3830   0.4885  
0.30       0.5928   0.4293   0.7326   0.6404   0.3830   0.4885  
0.35       0.5928   0.4293   0.7326   0.6404   0.3830   0.4885  
0.40       0.5928   0.4293   0.7326   0.6404   0.3830   0.4885  
0.45       0.5928   0.4293   0.7326   0.6404   0.3830   0.4885  
0.50       0.5928   0.4293   0.7326   0.6404   0.3830   0.4885  
0.55       0.5928   0.4293   0.7326   0.6404   0.3830   0.4885  
0.60       0.5928   0.4293   0.7326   0.6404   0.3830   0.4885  
0.65       0.5928   0.4293   0.7326   0.6404   0.3830   0.4885  
0.70       0.5928   0.4293   0.7326   0.6404   0.3830   0.4885  
0.75       0.5928   0.4293   0.7326   0.6404   0.3830   0.4885  
0.80       0.5928   0.4293   0.7326   0.6404   0.3830   0.4885  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5928, F1=0.4293, Normal Recall=0.7326, Normal Precision=0.6404, Attack Recall=0.3830, Attack Precision=0.4885

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
0.15       0.5581   0.4643   0.7332   0.5430   0.3830   0.5894   <--
0.20       0.5581   0.4643   0.7332   0.5430   0.3830   0.5894  
0.25       0.5581   0.4643   0.7332   0.5430   0.3830   0.5894  
0.30       0.5581   0.4643   0.7332   0.5430   0.3830   0.5894  
0.35       0.5581   0.4643   0.7332   0.5430   0.3830   0.5894  
0.40       0.5581   0.4643   0.7332   0.5430   0.3830   0.5894  
0.45       0.5581   0.4643   0.7332   0.5430   0.3830   0.5894  
0.50       0.5581   0.4643   0.7332   0.5430   0.3830   0.5894  
0.55       0.5581   0.4643   0.7332   0.5430   0.3830   0.5894  
0.60       0.5581   0.4643   0.7332   0.5430   0.3830   0.5894  
0.65       0.5581   0.4643   0.7332   0.5430   0.3830   0.5894  
0.70       0.5581   0.4643   0.7332   0.5430   0.3830   0.5894  
0.75       0.5581   0.4643   0.7332   0.5430   0.3830   0.5894  
0.80       0.5581   0.4643   0.7332   0.5430   0.3830   0.5894  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5581, F1=0.4643, Normal Recall=0.7332, Normal Precision=0.5430, Attack Recall=0.3830, Attack Precision=0.5894

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
0.15       0.9223   0.7161   0.9160   0.9975   0.9792   0.5644   <--
0.20       0.9223   0.7161   0.9160   0.9975   0.9792   0.5644  
0.25       0.9223   0.7161   0.9160   0.9975   0.9792   0.5644  
0.30       0.9223   0.7161   0.9160   0.9975   0.9792   0.5644  
0.35       0.9223   0.7161   0.9160   0.9975   0.9792   0.5644  
0.40       0.9223   0.7161   0.9160   0.9975   0.9792   0.5644  
0.45       0.9223   0.7161   0.9160   0.9975   0.9792   0.5644  
0.50       0.9223   0.7161   0.9160   0.9975   0.9792   0.5644  
0.55       0.9223   0.7161   0.9160   0.9975   0.9792   0.5644  
0.60       0.9223   0.7161   0.9160   0.9975   0.9792   0.5644  
0.65       0.9223   0.7161   0.9160   0.9975   0.9792   0.5644  
0.70       0.9223   0.7161   0.9160   0.9975   0.9792   0.5644  
0.75       0.9223   0.7161   0.9160   0.9975   0.9792   0.5644  
0.80       0.9223   0.7161   0.9160   0.9975   0.9792   0.5644  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9223, F1=0.7161, Normal Recall=0.9160, Normal Precision=0.9975, Attack Recall=0.9792, Attack Precision=0.5644

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
0.15       0.9287   0.8461   0.9161   0.9944   0.9792   0.7448   <--
0.20       0.9287   0.8461   0.9161   0.9944   0.9792   0.7448  
0.25       0.9287   0.8461   0.9161   0.9944   0.9792   0.7448  
0.30       0.9287   0.8461   0.9161   0.9944   0.9792   0.7448  
0.35       0.9287   0.8461   0.9161   0.9944   0.9792   0.7448  
0.40       0.9287   0.8461   0.9161   0.9944   0.9792   0.7448  
0.45       0.9287   0.8461   0.9161   0.9944   0.9792   0.7448  
0.50       0.9287   0.8461   0.9161   0.9944   0.9792   0.7448  
0.55       0.9287   0.8461   0.9161   0.9944   0.9792   0.7448  
0.60       0.9287   0.8461   0.9161   0.9944   0.9792   0.7448  
0.65       0.9287   0.8461   0.9161   0.9944   0.9792   0.7448  
0.70       0.9287   0.8461   0.9161   0.9944   0.9792   0.7448  
0.75       0.9287   0.8461   0.9161   0.9944   0.9792   0.7448  
0.80       0.9287   0.8461   0.9161   0.9944   0.9792   0.7448  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9287, F1=0.8461, Normal Recall=0.9161, Normal Precision=0.9944, Attack Recall=0.9792, Attack Precision=0.7448

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
0.15       0.9353   0.9008   0.9165   0.9904   0.9792   0.8340   <--
0.20       0.9353   0.9008   0.9165   0.9904   0.9792   0.8340  
0.25       0.9353   0.9008   0.9165   0.9904   0.9792   0.8340  
0.30       0.9353   0.9008   0.9165   0.9904   0.9792   0.8340  
0.35       0.9353   0.9008   0.9165   0.9904   0.9792   0.8340  
0.40       0.9353   0.9008   0.9165   0.9904   0.9792   0.8340  
0.45       0.9353   0.9008   0.9165   0.9904   0.9792   0.8340  
0.50       0.9353   0.9008   0.9165   0.9904   0.9792   0.8340  
0.55       0.9353   0.9008   0.9165   0.9904   0.9792   0.8340  
0.60       0.9353   0.9008   0.9165   0.9904   0.9792   0.8340  
0.65       0.9353   0.9008   0.9165   0.9904   0.9792   0.8340  
0.70       0.9353   0.9008   0.9165   0.9904   0.9792   0.8340  
0.75       0.9353   0.9008   0.9165   0.9904   0.9792   0.8340  
0.80       0.9353   0.9008   0.9165   0.9904   0.9792   0.8340  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9353, F1=0.9008, Normal Recall=0.9165, Normal Precision=0.9904, Attack Recall=0.9792, Attack Precision=0.8340

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
0.15       0.9416   0.9306   0.9165   0.9851   0.9792   0.8866   <--
0.20       0.9416   0.9306   0.9165   0.9851   0.9792   0.8866  
0.25       0.9416   0.9306   0.9165   0.9851   0.9792   0.8866  
0.30       0.9416   0.9306   0.9165   0.9851   0.9792   0.8866  
0.35       0.9416   0.9306   0.9165   0.9851   0.9792   0.8866  
0.40       0.9416   0.9306   0.9165   0.9851   0.9792   0.8866  
0.45       0.9416   0.9306   0.9165   0.9851   0.9792   0.8866  
0.50       0.9416   0.9306   0.9165   0.9851   0.9792   0.8866  
0.55       0.9416   0.9306   0.9165   0.9851   0.9792   0.8866  
0.60       0.9416   0.9306   0.9165   0.9851   0.9792   0.8866  
0.65       0.9416   0.9306   0.9165   0.9851   0.9792   0.8866  
0.70       0.9416   0.9306   0.9165   0.9851   0.9792   0.8866  
0.75       0.9416   0.9306   0.9165   0.9851   0.9792   0.8866  
0.80       0.9416   0.9306   0.9165   0.9851   0.9792   0.8866  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9416, F1=0.9306, Normal Recall=0.9165, Normal Precision=0.9851, Attack Recall=0.9792, Attack Precision=0.8866

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
0.15       0.9477   0.9493   0.9161   0.9778   0.9792   0.9211   <--
0.20       0.9477   0.9493   0.9161   0.9778   0.9792   0.9211  
0.25       0.9477   0.9493   0.9161   0.9778   0.9792   0.9211  
0.30       0.9477   0.9493   0.9161   0.9778   0.9792   0.9211  
0.35       0.9477   0.9493   0.9161   0.9778   0.9792   0.9211  
0.40       0.9477   0.9493   0.9161   0.9778   0.9792   0.9211  
0.45       0.9477   0.9493   0.9161   0.9778   0.9792   0.9211  
0.50       0.9477   0.9493   0.9161   0.9778   0.9792   0.9211  
0.55       0.9477   0.9493   0.9161   0.9778   0.9792   0.9211  
0.60       0.9477   0.9493   0.9161   0.9778   0.9792   0.9211  
0.65       0.9477   0.9493   0.9161   0.9778   0.9792   0.9211  
0.70       0.9477   0.9493   0.9161   0.9778   0.9792   0.9211  
0.75       0.9477   0.9493   0.9161   0.9778   0.9792   0.9211  
0.80       0.9477   0.9493   0.9161   0.9778   0.9792   0.9211  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9477, F1=0.9493, Normal Recall=0.9161, Normal Precision=0.9778, Attack Recall=0.9792, Attack Precision=0.9211

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
0.15       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632   <--
0.20       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.25       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.30       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.35       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.40       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.45       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.50       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.55       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.60       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.65       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.70       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.75       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.80       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9220, F1=0.7150, Normal Recall=0.9156, Normal Precision=0.9974, Attack Recall=0.9788, Attack Precision=0.5632

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
0.15       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439   <--
0.20       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.25       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.30       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.35       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.40       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.45       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.50       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.55       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.60       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.65       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.70       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.75       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.80       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9284, F1=0.8453, Normal Recall=0.9158, Normal Precision=0.9943, Attack Recall=0.9788, Attack Precision=0.7439

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
0.15       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335   <--
0.20       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.25       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.30       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.35       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.40       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.45       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.50       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.55       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.60       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.65       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.70       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.75       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.80       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9350, F1=0.9003, Normal Recall=0.9162, Normal Precision=0.9902, Attack Recall=0.9788, Attack Precision=0.8335

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
0.15       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862   <--
0.20       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.25       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.30       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.35       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.40       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.45       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.50       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.55       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.60       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.65       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.70       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.75       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.80       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9413, F1=0.9302, Normal Recall=0.9162, Normal Precision=0.9848, Attack Recall=0.9788, Attack Precision=0.8862

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
0.15       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208   <--
0.20       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.25       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.30       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.35       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.40       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.45       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.50       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.55       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.60       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.65       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.70       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.75       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.80       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9473, F1=0.9489, Normal Recall=0.9158, Normal Precision=0.9774, Attack Recall=0.9788, Attack Precision=0.9208

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
0.15       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632   <--
0.20       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.25       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.30       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.35       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.40       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.45       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.50       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.55       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.60       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.65       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.70       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.75       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
0.80       0.9220   0.7150   0.9156   0.9974   0.9788   0.5632  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9220, F1=0.7150, Normal Recall=0.9156, Normal Precision=0.9974, Attack Recall=0.9788, Attack Precision=0.5632

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
0.15       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439   <--
0.20       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.25       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.30       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.35       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.40       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.45       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.50       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.55       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.60       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.65       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.70       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.75       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
0.80       0.9284   0.8453   0.9158   0.9943   0.9788   0.7439  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9284, F1=0.8453, Normal Recall=0.9158, Normal Precision=0.9943, Attack Recall=0.9788, Attack Precision=0.7439

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
0.15       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335   <--
0.20       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.25       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.30       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.35       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.40       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.45       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.50       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.55       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.60       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.65       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.70       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.75       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
0.80       0.9350   0.9003   0.9162   0.9902   0.9788   0.8335  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9350, F1=0.9003, Normal Recall=0.9162, Normal Precision=0.9902, Attack Recall=0.9788, Attack Precision=0.8335

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
0.15       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862   <--
0.20       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.25       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.30       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.35       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.40       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.45       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.50       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.55       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.60       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.65       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.70       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.75       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
0.80       0.9413   0.9302   0.9162   0.9848   0.9788   0.8862  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9413, F1=0.9302, Normal Recall=0.9162, Normal Precision=0.9848, Attack Recall=0.9788, Attack Precision=0.8862

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
0.15       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208   <--
0.20       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.25       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.30       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.35       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.40       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.45       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.50       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.55       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.60       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.65       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.70       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.75       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
0.80       0.9473   0.9489   0.9158   0.9774   0.9788   0.9208  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9473, F1=0.9489, Normal Recall=0.9158, Normal Precision=0.9774, Attack Recall=0.9788, Attack Precision=0.9208

```

