# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e1_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-14 13:03:24 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3157 | 0.3606 | 0.4062 | 0.4526 | 0.4989 | 0.5443 | 0.5887 | 0.6350 | 0.6806 | 0.7264 | 0.7715 |
| QAT+Prune only | 0.3638 | 0.4252 | 0.4877 | 0.5519 | 0.6147 | 0.6757 | 0.7392 | 0.8014 | 0.8626 | 0.9258 | 0.9885 |
| QAT+PTQ | 0.3527 | 0.4155 | 0.4791 | 0.5442 | 0.6081 | 0.6703 | 0.7344 | 0.7982 | 0.8602 | 0.9249 | 0.9885 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.3527 | 0.4155 | 0.4791 | 0.5442 | 0.6081 | 0.6703 | 0.7344 | 0.7982 | 0.8602 | 0.9249 | 0.9885 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1942 | 0.3420 | 0.4582 | 0.5519 | 0.6287 | 0.6924 | 0.7474 | 0.7944 | 0.8354 | 0.8710 |
| QAT+Prune only | 0.0000 | 0.2559 | 0.4356 | 0.5697 | 0.6724 | 0.7530 | 0.8198 | 0.8745 | 0.9201 | 0.9600 | 0.9942 |
| QAT+PTQ | 0.0000 | 0.2527 | 0.4315 | 0.5654 | 0.6687 | 0.7499 | 0.8171 | 0.8727 | 0.9188 | 0.9595 | 0.9942 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2527 | 0.4315 | 0.5654 | 0.6687 | 0.7499 | 0.8171 | 0.8727 | 0.9188 | 0.9595 | 0.9942 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3157 | 0.3151 | 0.3149 | 0.3159 | 0.3171 | 0.3171 | 0.3145 | 0.3164 | 0.3169 | 0.3204 | 0.0000 |
| QAT+Prune only | 0.3638 | 0.3626 | 0.3625 | 0.3648 | 0.3654 | 0.3629 | 0.3652 | 0.3647 | 0.3592 | 0.3616 | 0.0000 |
| QAT+PTQ | 0.3527 | 0.3518 | 0.3518 | 0.3538 | 0.3546 | 0.3520 | 0.3533 | 0.3542 | 0.3469 | 0.3530 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.3527 | 0.3518 | 0.3518 | 0.3538 | 0.3546 | 0.3520 | 0.3533 | 0.3542 | 0.3469 | 0.3530 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3157 | 0.0000 | 0.0000 | 0.0000 | 0.3157 | 1.0000 |
| 90 | 10 | 299,940 | 0.3606 | 0.1111 | 0.7706 | 0.1942 | 0.3151 | 0.9251 |
| 80 | 20 | 291,350 | 0.4062 | 0.2197 | 0.7715 | 0.3420 | 0.3149 | 0.8465 |
| 70 | 30 | 194,230 | 0.4526 | 0.3259 | 0.7715 | 0.4582 | 0.3159 | 0.7634 |
| 60 | 40 | 145,675 | 0.4989 | 0.4296 | 0.7715 | 0.5519 | 0.3171 | 0.6755 |
| 50 | 50 | 116,540 | 0.5443 | 0.5305 | 0.7715 | 0.6287 | 0.3171 | 0.5812 |
| 40 | 60 | 97,115 | 0.5887 | 0.6280 | 0.7715 | 0.6924 | 0.3145 | 0.4786 |
| 30 | 70 | 83,240 | 0.6350 | 0.7248 | 0.7715 | 0.7474 | 0.3164 | 0.3724 |
| 20 | 80 | 72,835 | 0.6806 | 0.8188 | 0.7715 | 0.7944 | 0.3169 | 0.2575 |
| 10 | 90 | 64,740 | 0.7264 | 0.9108 | 0.7715 | 0.8354 | 0.3204 | 0.1348 |
| 0 | 100 | 58,270 | 0.7715 | 1.0000 | 0.7715 | 0.8710 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3638 | 0.0000 | 0.0000 | 0.0000 | 0.3638 | 1.0000 |
| 90 | 10 | 299,940 | 0.4252 | 0.1470 | 0.9884 | 0.2559 | 0.3626 | 0.9965 |
| 80 | 20 | 291,350 | 0.4877 | 0.2794 | 0.9885 | 0.4356 | 0.3625 | 0.9921 |
| 70 | 30 | 194,230 | 0.5519 | 0.4001 | 0.9885 | 0.5697 | 0.3648 | 0.9867 |
| 60 | 40 | 145,675 | 0.6147 | 0.5094 | 0.9885 | 0.6724 | 0.3654 | 0.9795 |
| 50 | 50 | 116,540 | 0.6757 | 0.6081 | 0.9885 | 0.7530 | 0.3629 | 0.9693 |
| 40 | 60 | 97,115 | 0.7392 | 0.7002 | 0.9885 | 0.8198 | 0.3652 | 0.9550 |
| 30 | 70 | 83,240 | 0.8014 | 0.7840 | 0.9885 | 0.8745 | 0.3647 | 0.9316 |
| 20 | 80 | 72,835 | 0.8626 | 0.8605 | 0.9885 | 0.9201 | 0.3592 | 0.8866 |
| 10 | 90 | 64,740 | 0.9258 | 0.9330 | 0.9885 | 0.9600 | 0.3616 | 0.7777 |
| 0 | 100 | 58,270 | 0.9885 | 1.0000 | 0.9885 | 0.9942 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3527 | 0.0000 | 0.0000 | 0.0000 | 0.3527 | 1.0000 |
| 90 | 10 | 299,940 | 0.4155 | 0.1449 | 0.9884 | 0.2527 | 0.3518 | 0.9963 |
| 80 | 20 | 291,350 | 0.4791 | 0.2760 | 0.9885 | 0.4315 | 0.3518 | 0.9919 |
| 70 | 30 | 194,230 | 0.5442 | 0.3960 | 0.9885 | 0.5654 | 0.3538 | 0.9862 |
| 60 | 40 | 145,675 | 0.6081 | 0.5052 | 0.9885 | 0.6687 | 0.3546 | 0.9788 |
| 50 | 50 | 116,540 | 0.6703 | 0.6040 | 0.9885 | 0.7499 | 0.3520 | 0.9683 |
| 40 | 60 | 97,115 | 0.7344 | 0.6963 | 0.9885 | 0.8171 | 0.3533 | 0.9534 |
| 30 | 70 | 83,240 | 0.7982 | 0.7813 | 0.9885 | 0.8727 | 0.3542 | 0.9295 |
| 20 | 80 | 72,835 | 0.8602 | 0.8582 | 0.9885 | 0.9188 | 0.3469 | 0.8828 |
| 10 | 90 | 64,740 | 0.9249 | 0.9322 | 0.9885 | 0.9595 | 0.3530 | 0.7730 |
| 0 | 100 | 58,270 | 0.9885 | 1.0000 | 0.9885 | 0.9942 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.3527 | 0.0000 | 0.0000 | 0.0000 | 0.3527 | 1.0000 |
| 90 | 10 | 299,940 | 0.4155 | 0.1449 | 0.9884 | 0.2527 | 0.3518 | 0.9963 |
| 80 | 20 | 291,350 | 0.4791 | 0.2760 | 0.9885 | 0.4315 | 0.3518 | 0.9919 |
| 70 | 30 | 194,230 | 0.5442 | 0.3960 | 0.9885 | 0.5654 | 0.3538 | 0.9862 |
| 60 | 40 | 145,675 | 0.6081 | 0.5052 | 0.9885 | 0.6687 | 0.3546 | 0.9788 |
| 50 | 50 | 116,540 | 0.6703 | 0.6040 | 0.9885 | 0.7499 | 0.3520 | 0.9683 |
| 40 | 60 | 97,115 | 0.7344 | 0.6963 | 0.9885 | 0.8171 | 0.3533 | 0.9534 |
| 30 | 70 | 83,240 | 0.7982 | 0.7813 | 0.9885 | 0.8727 | 0.3542 | 0.9295 |
| 20 | 80 | 72,835 | 0.8602 | 0.8582 | 0.9885 | 0.9188 | 0.3469 | 0.8828 |
| 10 | 90 | 64,740 | 0.9249 | 0.9322 | 0.9885 | 0.9595 | 0.3530 | 0.7730 |
| 0 | 100 | 58,270 | 0.9885 | 1.0000 | 0.9885 | 0.9942 | 0.0000 | 0.0000 |


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
0.15       0.3607   0.1943   0.3151   0.9253   0.7711   0.1112   <--
0.20       0.3607   0.1943   0.3151   0.9253   0.7711   0.1112  
0.25       0.3607   0.1943   0.3151   0.9253   0.7711   0.1112  
0.30       0.3607   0.1943   0.3151   0.9253   0.7711   0.1112  
0.35       0.3607   0.1943   0.3151   0.9253   0.7711   0.1112  
0.40       0.3607   0.1943   0.3151   0.9253   0.7711   0.1112  
0.45       0.3607   0.1943   0.3151   0.9253   0.7711   0.1112  
0.50       0.3607   0.1943   0.3151   0.9253   0.7711   0.1112  
0.55       0.3607   0.1943   0.3151   0.9253   0.7711   0.1112  
0.60       0.3607   0.1943   0.3151   0.9253   0.7711   0.1112  
0.65       0.3607   0.1943   0.3151   0.9253   0.7711   0.1112  
0.70       0.3607   0.1943   0.3151   0.9253   0.7711   0.1112  
0.75       0.3607   0.1943   0.3151   0.9253   0.7711   0.1112  
0.80       0.3607   0.1943   0.3151   0.9253   0.7711   0.1112  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3607, F1=0.1943, Normal Recall=0.3151, Normal Precision=0.9253, Attack Recall=0.7711, Attack Precision=0.1112

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
0.15       0.4066   0.3422   0.3154   0.8467   0.7715   0.2198   <--
0.20       0.4066   0.3422   0.3154   0.8467   0.7715   0.2198  
0.25       0.4066   0.3422   0.3154   0.8467   0.7715   0.2198  
0.30       0.4066   0.3422   0.3154   0.8467   0.7715   0.2198  
0.35       0.4066   0.3422   0.3154   0.8467   0.7715   0.2198  
0.40       0.4066   0.3422   0.3154   0.8467   0.7715   0.2198  
0.45       0.4066   0.3422   0.3154   0.8467   0.7715   0.2198  
0.50       0.4066   0.3422   0.3154   0.8467   0.7715   0.2198  
0.55       0.4066   0.3422   0.3154   0.8467   0.7715   0.2198  
0.60       0.4066   0.3422   0.3154   0.8467   0.7715   0.2198  
0.65       0.4066   0.3422   0.3154   0.8467   0.7715   0.2198  
0.70       0.4066   0.3422   0.3154   0.8467   0.7715   0.2198  
0.75       0.4066   0.3422   0.3154   0.8467   0.7715   0.2198  
0.80       0.4066   0.3422   0.3154   0.8467   0.7715   0.2198  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4066, F1=0.3422, Normal Recall=0.3154, Normal Precision=0.8467, Attack Recall=0.7715, Attack Precision=0.2198

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
0.15       0.4530   0.4584   0.3165   0.7637   0.7715   0.3260   <--
0.20       0.4530   0.4584   0.3165   0.7637   0.7715   0.3260  
0.25       0.4530   0.4584   0.3165   0.7637   0.7715   0.3260  
0.30       0.4530   0.4584   0.3165   0.7637   0.7715   0.3260  
0.35       0.4530   0.4584   0.3165   0.7637   0.7715   0.3260  
0.40       0.4530   0.4584   0.3165   0.7637   0.7715   0.3260  
0.45       0.4530   0.4584   0.3165   0.7637   0.7715   0.3260  
0.50       0.4530   0.4584   0.3165   0.7637   0.7715   0.3260  
0.55       0.4530   0.4584   0.3165   0.7637   0.7715   0.3260  
0.60       0.4530   0.4584   0.3165   0.7637   0.7715   0.3260  
0.65       0.4530   0.4584   0.3165   0.7637   0.7715   0.3260  
0.70       0.4530   0.4584   0.3165   0.7637   0.7715   0.3260  
0.75       0.4530   0.4584   0.3165   0.7637   0.7715   0.3260  
0.80       0.4530   0.4584   0.3165   0.7637   0.7715   0.3260  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4530, F1=0.4584, Normal Recall=0.3165, Normal Precision=0.7637, Attack Recall=0.7715, Attack Precision=0.3260

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
0.15       0.4982   0.5516   0.3160   0.6747   0.7715   0.4292   <--
0.20       0.4982   0.5516   0.3160   0.6747   0.7715   0.4292  
0.25       0.4982   0.5516   0.3160   0.6747   0.7715   0.4292  
0.30       0.4982   0.5516   0.3160   0.6747   0.7715   0.4292  
0.35       0.4982   0.5516   0.3160   0.6747   0.7715   0.4292  
0.40       0.4982   0.5516   0.3160   0.6747   0.7715   0.4292  
0.45       0.4982   0.5516   0.3160   0.6747   0.7715   0.4292  
0.50       0.4982   0.5516   0.3160   0.6747   0.7715   0.4292  
0.55       0.4982   0.5516   0.3160   0.6747   0.7715   0.4292  
0.60       0.4982   0.5516   0.3160   0.6747   0.7715   0.4292  
0.65       0.4982   0.5516   0.3160   0.6747   0.7715   0.4292  
0.70       0.4982   0.5516   0.3160   0.6747   0.7715   0.4292  
0.75       0.4982   0.5516   0.3160   0.6747   0.7715   0.4292  
0.80       0.4982   0.5516   0.3160   0.6747   0.7715   0.4292  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4982, F1=0.5516, Normal Recall=0.3160, Normal Precision=0.6747, Attack Recall=0.7715, Attack Precision=0.4292

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
0.15       0.5443   0.6287   0.3171   0.5813   0.7715   0.5305   <--
0.20       0.5443   0.6287   0.3171   0.5813   0.7715   0.5305  
0.25       0.5443   0.6287   0.3171   0.5813   0.7715   0.5305  
0.30       0.5443   0.6287   0.3171   0.5813   0.7715   0.5305  
0.35       0.5443   0.6287   0.3171   0.5813   0.7715   0.5305  
0.40       0.5443   0.6287   0.3171   0.5813   0.7715   0.5305  
0.45       0.5443   0.6287   0.3171   0.5813   0.7715   0.5305  
0.50       0.5443   0.6287   0.3171   0.5813   0.7715   0.5305  
0.55       0.5443   0.6287   0.3171   0.5813   0.7715   0.5305  
0.60       0.5443   0.6287   0.3171   0.5813   0.7715   0.5305  
0.65       0.5443   0.6287   0.3171   0.5813   0.7715   0.5305  
0.70       0.5443   0.6287   0.3171   0.5813   0.7715   0.5305  
0.75       0.5443   0.6287   0.3171   0.5813   0.7715   0.5305  
0.80       0.5443   0.6287   0.3171   0.5813   0.7715   0.5305  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5443, F1=0.6287, Normal Recall=0.3171, Normal Precision=0.5813, Attack Recall=0.7715, Attack Precision=0.5305

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
0.15       0.4253   0.2560   0.3626   0.9966   0.9888   0.1470   <--
0.20       0.4253   0.2560   0.3626   0.9966   0.9888   0.1470  
0.25       0.4253   0.2560   0.3626   0.9966   0.9888   0.1470  
0.30       0.4253   0.2560   0.3626   0.9966   0.9888   0.1470  
0.35       0.4253   0.2560   0.3626   0.9966   0.9888   0.1470  
0.40       0.4253   0.2560   0.3626   0.9966   0.9888   0.1470  
0.45       0.4253   0.2560   0.3626   0.9966   0.9888   0.1470  
0.50       0.4253   0.2560   0.3626   0.9966   0.9888   0.1470  
0.55       0.4253   0.2560   0.3626   0.9966   0.9888   0.1470  
0.60       0.4253   0.2560   0.3626   0.9966   0.9888   0.1470  
0.65       0.4253   0.2560   0.3626   0.9966   0.9888   0.1470  
0.70       0.4253   0.2560   0.3626   0.9966   0.9888   0.1470  
0.75       0.4253   0.2560   0.3626   0.9966   0.9888   0.1470  
0.80       0.4253   0.2560   0.3626   0.9966   0.9888   0.1470  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4253, F1=0.2560, Normal Recall=0.3626, Normal Precision=0.9966, Attack Recall=0.9888, Attack Precision=0.1470

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
0.15       0.4880   0.4358   0.3629   0.9922   0.9885   0.2795   <--
0.20       0.4880   0.4358   0.3629   0.9922   0.9885   0.2795  
0.25       0.4880   0.4358   0.3629   0.9922   0.9885   0.2795  
0.30       0.4880   0.4358   0.3629   0.9922   0.9885   0.2795  
0.35       0.4880   0.4358   0.3629   0.9922   0.9885   0.2795  
0.40       0.4880   0.4358   0.3629   0.9922   0.9885   0.2795  
0.45       0.4880   0.4358   0.3629   0.9922   0.9885   0.2795  
0.50       0.4880   0.4358   0.3629   0.9922   0.9885   0.2795  
0.55       0.4880   0.4358   0.3629   0.9922   0.9885   0.2795  
0.60       0.4880   0.4358   0.3629   0.9922   0.9885   0.2795  
0.65       0.4880   0.4358   0.3629   0.9922   0.9885   0.2795  
0.70       0.4880   0.4358   0.3629   0.9922   0.9885   0.2795  
0.75       0.4880   0.4358   0.3629   0.9922   0.9885   0.2795  
0.80       0.4880   0.4358   0.3629   0.9922   0.9885   0.2795  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4880, F1=0.4358, Normal Recall=0.3629, Normal Precision=0.9922, Attack Recall=0.9885, Attack Precision=0.2795

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
0.15       0.5512   0.5693   0.3638   0.9867   0.9885   0.3997   <--
0.20       0.5512   0.5693   0.3638   0.9867   0.9885   0.3997  
0.25       0.5512   0.5693   0.3638   0.9867   0.9885   0.3997  
0.30       0.5512   0.5693   0.3638   0.9867   0.9885   0.3997  
0.35       0.5512   0.5693   0.3638   0.9867   0.9885   0.3997  
0.40       0.5512   0.5693   0.3638   0.9867   0.9885   0.3997  
0.45       0.5512   0.5693   0.3638   0.9867   0.9885   0.3997  
0.50       0.5512   0.5693   0.3638   0.9867   0.9885   0.3997  
0.55       0.5512   0.5693   0.3638   0.9867   0.9885   0.3997  
0.60       0.5512   0.5693   0.3638   0.9867   0.9885   0.3997  
0.65       0.5512   0.5693   0.3638   0.9867   0.9885   0.3997  
0.70       0.5512   0.5693   0.3638   0.9867   0.9885   0.3997  
0.75       0.5512   0.5693   0.3638   0.9867   0.9885   0.3997  
0.80       0.5512   0.5693   0.3638   0.9867   0.9885   0.3997  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5512, F1=0.5693, Normal Recall=0.3638, Normal Precision=0.9867, Attack Recall=0.9885, Attack Precision=0.3997

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
0.15       0.6134   0.6717   0.3633   0.9794   0.9885   0.5086   <--
0.20       0.6134   0.6717   0.3633   0.9794   0.9885   0.5086  
0.25       0.6134   0.6717   0.3633   0.9794   0.9885   0.5086  
0.30       0.6134   0.6717   0.3633   0.9794   0.9885   0.5086  
0.35       0.6134   0.6717   0.3633   0.9794   0.9885   0.5086  
0.40       0.6134   0.6717   0.3633   0.9794   0.9885   0.5086  
0.45       0.6134   0.6717   0.3633   0.9794   0.9885   0.5086  
0.50       0.6134   0.6717   0.3633   0.9794   0.9885   0.5086  
0.55       0.6134   0.6717   0.3633   0.9794   0.9885   0.5086  
0.60       0.6134   0.6717   0.3633   0.9794   0.9885   0.5086  
0.65       0.6134   0.6717   0.3633   0.9794   0.9885   0.5086  
0.70       0.6134   0.6717   0.3633   0.9794   0.9885   0.5086  
0.75       0.6134   0.6717   0.3633   0.9794   0.9885   0.5086  
0.80       0.6134   0.6717   0.3633   0.9794   0.9885   0.5086  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6134, F1=0.6717, Normal Recall=0.3633, Normal Precision=0.9794, Attack Recall=0.9885, Attack Precision=0.5086

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
0.15       0.6753   0.7528   0.3621   0.9693   0.9885   0.6078   <--
0.20       0.6753   0.7528   0.3621   0.9693   0.9885   0.6078  
0.25       0.6753   0.7528   0.3621   0.9693   0.9885   0.6078  
0.30       0.6753   0.7528   0.3621   0.9693   0.9885   0.6078  
0.35       0.6753   0.7528   0.3621   0.9693   0.9885   0.6078  
0.40       0.6753   0.7528   0.3621   0.9693   0.9885   0.6078  
0.45       0.6753   0.7528   0.3621   0.9693   0.9885   0.6078  
0.50       0.6753   0.7528   0.3621   0.9693   0.9885   0.6078  
0.55       0.6753   0.7528   0.3621   0.9693   0.9885   0.6078  
0.60       0.6753   0.7528   0.3621   0.9693   0.9885   0.6078  
0.65       0.6753   0.7528   0.3621   0.9693   0.9885   0.6078  
0.70       0.6753   0.7528   0.3621   0.9693   0.9885   0.6078  
0.75       0.6753   0.7528   0.3621   0.9693   0.9885   0.6078  
0.80       0.6753   0.7528   0.3621   0.9693   0.9885   0.6078  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6753, F1=0.7528, Normal Recall=0.3621, Normal Precision=0.9693, Attack Recall=0.9885, Attack Precision=0.6078

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
0.15       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449   <--
0.20       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.25       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.30       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.35       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.40       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.45       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.50       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.55       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.60       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.65       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.70       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.75       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.80       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4155, F1=0.2528, Normal Recall=0.3518, Normal Precision=0.9965, Attack Recall=0.9887, Attack Precision=0.1449

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
0.15       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762   <--
0.20       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.25       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.30       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.35       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.40       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.45       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.50       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.55       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.60       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.65       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.70       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.75       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.80       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4795, F1=0.4317, Normal Recall=0.3523, Normal Precision=0.9919, Attack Recall=0.9885, Attack Precision=0.2762

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
0.15       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956   <--
0.20       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.25       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.30       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.35       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.40       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.45       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.50       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.55       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.60       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.65       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.70       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.75       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.80       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5435, F1=0.5651, Normal Recall=0.3528, Normal Precision=0.9862, Attack Recall=0.9885, Attack Precision=0.3956

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
0.15       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043   <--
0.20       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.25       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.30       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.35       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.40       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.45       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.50       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.55       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.60       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.65       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.70       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.75       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.80       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6068, F1=0.6679, Normal Recall=0.3523, Normal Precision=0.9787, Attack Recall=0.9885, Attack Precision=0.5043

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
0.15       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037   <--
0.20       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.25       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.30       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.35       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.40       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.45       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.50       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.55       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.60       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.65       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.70       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.75       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.80       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6698, F1=0.7496, Normal Recall=0.3511, Normal Precision=0.9682, Attack Recall=0.9885, Attack Precision=0.6037

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
0.15       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449   <--
0.20       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.25       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.30       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.35       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.40       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.45       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.50       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.55       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.60       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.65       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.70       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.75       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
0.80       0.4155   0.2528   0.3518   0.9965   0.9887   0.1449  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4155, F1=0.2528, Normal Recall=0.3518, Normal Precision=0.9965, Attack Recall=0.9887, Attack Precision=0.1449

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
0.15       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762   <--
0.20       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.25       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.30       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.35       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.40       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.45       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.50       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.55       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.60       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.65       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.70       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.75       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
0.80       0.4795   0.4317   0.3523   0.9919   0.9885   0.2762  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4795, F1=0.4317, Normal Recall=0.3523, Normal Precision=0.9919, Attack Recall=0.9885, Attack Precision=0.2762

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
0.15       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956   <--
0.20       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.25       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.30       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.35       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.40       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.45       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.50       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.55       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.60       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.65       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.70       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.75       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
0.80       0.5435   0.5651   0.3528   0.9862   0.9885   0.3956  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5435, F1=0.5651, Normal Recall=0.3528, Normal Precision=0.9862, Attack Recall=0.9885, Attack Precision=0.3956

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
0.15       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043   <--
0.20       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.25       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.30       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.35       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.40       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.45       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.50       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.55       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.60       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.65       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.70       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.75       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
0.80       0.6068   0.6679   0.3523   0.9787   0.9885   0.5043  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6068, F1=0.6679, Normal Recall=0.3523, Normal Precision=0.9787, Attack Recall=0.9885, Attack Precision=0.5043

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
0.15       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037   <--
0.20       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.25       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.30       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.35       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.40       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.45       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.50       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.55       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.60       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.65       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.70       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.75       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
0.80       0.6698   0.7496   0.3511   0.9682   0.9885   0.6037  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6698, F1=0.7496, Normal Recall=0.3511, Normal Precision=0.9682, Attack Recall=0.9885, Attack Precision=0.6037

```

