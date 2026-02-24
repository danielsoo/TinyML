# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e1_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-21 04:26:44 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9237 | 0.9211 | 0.9189 | 0.9178 | 0.9151 | 0.9132 | 0.9114 | 0.9101 | 0.9082 | 0.9057 | 0.9045 |
| QAT+Prune only | 0.7458 | 0.7700 | 0.7940 | 0.8188 | 0.8430 | 0.8657 | 0.8920 | 0.9162 | 0.9412 | 0.9645 | 0.9895 |
| QAT+PTQ | 0.7454 | 0.7694 | 0.7936 | 0.8185 | 0.8428 | 0.8655 | 0.8920 | 0.9162 | 0.9410 | 0.9647 | 0.9897 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7454 | 0.7694 | 0.7936 | 0.8185 | 0.8428 | 0.8655 | 0.8920 | 0.9162 | 0.9410 | 0.9647 | 0.9897 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6964 | 0.8168 | 0.8685 | 0.8950 | 0.9125 | 0.9245 | 0.9337 | 0.9404 | 0.9453 | 0.9499 |
| QAT+Prune only | 0.0000 | 0.4624 | 0.6577 | 0.7662 | 0.8345 | 0.8805 | 0.9166 | 0.9430 | 0.9642 | 0.9804 | 0.9947 |
| QAT+PTQ | 0.0000 | 0.4618 | 0.6572 | 0.7659 | 0.8343 | 0.8804 | 0.9166 | 0.9429 | 0.9641 | 0.9805 | 0.9948 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4618 | 0.6572 | 0.7659 | 0.8343 | 0.8804 | 0.9166 | 0.9429 | 0.9641 | 0.9805 | 0.9948 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9237 | 0.9229 | 0.9224 | 0.9235 | 0.9222 | 0.9219 | 0.9217 | 0.9230 | 0.9230 | 0.9166 | 0.0000 |
| QAT+Prune only | 0.7458 | 0.7456 | 0.7452 | 0.7457 | 0.7454 | 0.7420 | 0.7458 | 0.7452 | 0.7480 | 0.7394 | 0.0000 |
| QAT+PTQ | 0.7454 | 0.7450 | 0.7445 | 0.7452 | 0.7449 | 0.7414 | 0.7455 | 0.7446 | 0.7465 | 0.7396 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7454 | 0.7450 | 0.7445 | 0.7452 | 0.7449 | 0.7414 | 0.7455 | 0.7446 | 0.7465 | 0.7396 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9237 | 0.0000 | 0.0000 | 0.0000 | 0.9237 | 1.0000 |
| 90 | 10 | 299,940 | 0.9211 | 0.5660 | 0.9048 | 0.6964 | 0.9229 | 0.9887 |
| 80 | 20 | 291,350 | 0.9189 | 0.7446 | 0.9045 | 0.8168 | 0.9224 | 0.9748 |
| 70 | 30 | 194,230 | 0.9178 | 0.8352 | 0.9045 | 0.8685 | 0.9235 | 0.9576 |
| 60 | 40 | 145,675 | 0.9151 | 0.8857 | 0.9045 | 0.8950 | 0.9222 | 0.9354 |
| 50 | 50 | 116,540 | 0.9132 | 0.9205 | 0.9045 | 0.9125 | 0.9219 | 0.9061 |
| 40 | 60 | 97,115 | 0.9114 | 0.9454 | 0.9045 | 0.9245 | 0.9217 | 0.8655 |
| 30 | 70 | 83,240 | 0.9101 | 0.9648 | 0.9045 | 0.9337 | 0.9230 | 0.8055 |
| 20 | 80 | 72,835 | 0.9082 | 0.9792 | 0.9045 | 0.9404 | 0.9230 | 0.7073 |
| 10 | 90 | 64,740 | 0.9057 | 0.9899 | 0.9045 | 0.9453 | 0.9166 | 0.5161 |
| 0 | 100 | 58,270 | 0.9045 | 1.0000 | 0.9045 | 0.9499 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7458 | 0.0000 | 0.0000 | 0.0000 | 0.7458 | 1.0000 |
| 90 | 10 | 299,940 | 0.7700 | 0.3017 | 0.9892 | 0.4624 | 0.7456 | 0.9984 |
| 80 | 20 | 291,350 | 0.7940 | 0.4926 | 0.9895 | 0.6577 | 0.7452 | 0.9965 |
| 70 | 30 | 194,230 | 0.8188 | 0.6251 | 0.9895 | 0.7662 | 0.7457 | 0.9940 |
| 60 | 40 | 145,675 | 0.8430 | 0.7215 | 0.9895 | 0.8345 | 0.7454 | 0.9907 |
| 50 | 50 | 116,540 | 0.8657 | 0.7932 | 0.9895 | 0.8805 | 0.7420 | 0.9860 |
| 40 | 60 | 97,115 | 0.8920 | 0.8538 | 0.9895 | 0.9166 | 0.7458 | 0.9792 |
| 30 | 70 | 83,240 | 0.9162 | 0.9006 | 0.9895 | 0.9430 | 0.7452 | 0.9681 |
| 20 | 80 | 72,835 | 0.9412 | 0.9401 | 0.9895 | 0.9642 | 0.7480 | 0.9467 |
| 10 | 90 | 64,740 | 0.9645 | 0.9716 | 0.9895 | 0.9804 | 0.7394 | 0.8863 |
| 0 | 100 | 58,270 | 0.9895 | 1.0000 | 0.9895 | 0.9947 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7454 | 0.0000 | 0.0000 | 0.0000 | 0.7454 | 1.0000 |
| 90 | 10 | 299,940 | 0.7694 | 0.3012 | 0.9893 | 0.4618 | 0.7450 | 0.9984 |
| 80 | 20 | 291,350 | 0.7936 | 0.4920 | 0.9897 | 0.6572 | 0.7445 | 0.9965 |
| 70 | 30 | 194,230 | 0.8185 | 0.6247 | 0.9897 | 0.7659 | 0.7452 | 0.9941 |
| 60 | 40 | 145,675 | 0.8428 | 0.7212 | 0.9897 | 0.8343 | 0.7449 | 0.9908 |
| 50 | 50 | 116,540 | 0.8655 | 0.7928 | 0.9897 | 0.8804 | 0.7414 | 0.9863 |
| 40 | 60 | 97,115 | 0.8920 | 0.8536 | 0.9897 | 0.9166 | 0.7455 | 0.9796 |
| 30 | 70 | 83,240 | 0.9162 | 0.9004 | 0.9897 | 0.9429 | 0.7446 | 0.9686 |
| 20 | 80 | 72,835 | 0.9410 | 0.9398 | 0.9897 | 0.9641 | 0.7465 | 0.9475 |
| 10 | 90 | 64,740 | 0.9647 | 0.9716 | 0.9897 | 0.9805 | 0.7396 | 0.8883 |
| 0 | 100 | 58,270 | 0.9897 | 1.0000 | 0.9897 | 0.9948 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7454 | 0.0000 | 0.0000 | 0.0000 | 0.7454 | 1.0000 |
| 90 | 10 | 299,940 | 0.7694 | 0.3012 | 0.9893 | 0.4618 | 0.7450 | 0.9984 |
| 80 | 20 | 291,350 | 0.7936 | 0.4920 | 0.9897 | 0.6572 | 0.7445 | 0.9965 |
| 70 | 30 | 194,230 | 0.8185 | 0.6247 | 0.9897 | 0.7659 | 0.7452 | 0.9941 |
| 60 | 40 | 145,675 | 0.8428 | 0.7212 | 0.9897 | 0.8343 | 0.7449 | 0.9908 |
| 50 | 50 | 116,540 | 0.8655 | 0.7928 | 0.9897 | 0.8804 | 0.7414 | 0.9863 |
| 40 | 60 | 97,115 | 0.8920 | 0.8536 | 0.9897 | 0.9166 | 0.7455 | 0.9796 |
| 30 | 70 | 83,240 | 0.9162 | 0.9004 | 0.9897 | 0.9429 | 0.7446 | 0.9686 |
| 20 | 80 | 72,835 | 0.9410 | 0.9398 | 0.9897 | 0.9641 | 0.7465 | 0.9475 |
| 10 | 90 | 64,740 | 0.9647 | 0.9716 | 0.9897 | 0.9805 | 0.7396 | 0.8883 |
| 0 | 100 | 58,270 | 0.9897 | 1.0000 | 0.9897 | 0.9948 | 0.0000 | 0.0000 |


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
0.15       0.9212   0.6967   0.9229   0.9888   0.9055   0.5662   <--
0.20       0.9212   0.6967   0.9229   0.9888   0.9055   0.5662  
0.25       0.9212   0.6967   0.9229   0.9888   0.9055   0.5662  
0.30       0.9212   0.6967   0.9229   0.9888   0.9055   0.5662  
0.35       0.9212   0.6967   0.9229   0.9888   0.9055   0.5662  
0.40       0.9212   0.6967   0.9229   0.9888   0.9055   0.5662  
0.45       0.9212   0.6967   0.9229   0.9888   0.9055   0.5662  
0.50       0.9212   0.6967   0.9229   0.9888   0.9055   0.5662  
0.55       0.9212   0.6967   0.9229   0.9888   0.9055   0.5662  
0.60       0.9212   0.6967   0.9229   0.9888   0.9055   0.5662  
0.65       0.9212   0.6967   0.9229   0.9888   0.9055   0.5662  
0.70       0.9212   0.6967   0.9229   0.9888   0.9055   0.5662  
0.75       0.9212   0.6967   0.9229   0.9888   0.9055   0.5662  
0.80       0.9212   0.6967   0.9229   0.9888   0.9055   0.5662  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9212, F1=0.6967, Normal Recall=0.9229, Normal Precision=0.9888, Attack Recall=0.9055, Attack Precision=0.5662

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
0.15       0.9193   0.8176   0.9230   0.9748   0.9045   0.7459   <--
0.20       0.9193   0.8176   0.9230   0.9748   0.9045   0.7459  
0.25       0.9193   0.8176   0.9230   0.9748   0.9045   0.7459  
0.30       0.9193   0.8176   0.9230   0.9748   0.9045   0.7459  
0.35       0.9193   0.8176   0.9230   0.9748   0.9045   0.7459  
0.40       0.9193   0.8176   0.9230   0.9748   0.9045   0.7459  
0.45       0.9193   0.8176   0.9230   0.9748   0.9045   0.7459  
0.50       0.9193   0.8176   0.9230   0.9748   0.9045   0.7459  
0.55       0.9193   0.8176   0.9230   0.9748   0.9045   0.7459  
0.60       0.9193   0.8176   0.9230   0.9748   0.9045   0.7459  
0.65       0.9193   0.8176   0.9230   0.9748   0.9045   0.7459  
0.70       0.9193   0.8176   0.9230   0.9748   0.9045   0.7459  
0.75       0.9193   0.8176   0.9230   0.9748   0.9045   0.7459  
0.80       0.9193   0.8176   0.9230   0.9748   0.9045   0.7459  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9193, F1=0.8176, Normal Recall=0.9230, Normal Precision=0.9748, Attack Recall=0.9045, Attack Precision=0.7459

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
0.15       0.9179   0.8687   0.9237   0.9576   0.9045   0.8356   <--
0.20       0.9179   0.8687   0.9237   0.9576   0.9045   0.8356  
0.25       0.9179   0.8687   0.9237   0.9576   0.9045   0.8356  
0.30       0.9179   0.8687   0.9237   0.9576   0.9045   0.8356  
0.35       0.9179   0.8687   0.9237   0.9576   0.9045   0.8356  
0.40       0.9179   0.8687   0.9237   0.9576   0.9045   0.8356  
0.45       0.9179   0.8687   0.9237   0.9576   0.9045   0.8356  
0.50       0.9179   0.8687   0.9237   0.9576   0.9045   0.8356  
0.55       0.9179   0.8687   0.9237   0.9576   0.9045   0.8356  
0.60       0.9179   0.8687   0.9237   0.9576   0.9045   0.8356  
0.65       0.9179   0.8687   0.9237   0.9576   0.9045   0.8356  
0.70       0.9179   0.8687   0.9237   0.9576   0.9045   0.8356  
0.75       0.9179   0.8687   0.9237   0.9576   0.9045   0.8356  
0.80       0.9179   0.8687   0.9237   0.9576   0.9045   0.8356  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9179, F1=0.8687, Normal Recall=0.9237, Normal Precision=0.9576, Attack Recall=0.9045, Attack Precision=0.8356

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
0.15       0.9159   0.8959   0.9236   0.9355   0.9045   0.8875   <--
0.20       0.9159   0.8959   0.9236   0.9355   0.9045   0.8875  
0.25       0.9159   0.8959   0.9236   0.9355   0.9045   0.8875  
0.30       0.9159   0.8959   0.9236   0.9355   0.9045   0.8875  
0.35       0.9159   0.8959   0.9236   0.9355   0.9045   0.8875  
0.40       0.9159   0.8959   0.9236   0.9355   0.9045   0.8875  
0.45       0.9159   0.8959   0.9236   0.9355   0.9045   0.8875  
0.50       0.9159   0.8959   0.9236   0.9355   0.9045   0.8875  
0.55       0.9159   0.8959   0.9236   0.9355   0.9045   0.8875  
0.60       0.9159   0.8959   0.9236   0.9355   0.9045   0.8875  
0.65       0.9159   0.8959   0.9236   0.9355   0.9045   0.8875  
0.70       0.9159   0.8959   0.9236   0.9355   0.9045   0.8875  
0.75       0.9159   0.8959   0.9236   0.9355   0.9045   0.8875  
0.80       0.9159   0.8959   0.9236   0.9355   0.9045   0.8875  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9159, F1=0.8959, Normal Recall=0.9236, Normal Precision=0.9355, Attack Recall=0.9045, Attack Precision=0.8875

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
0.15       0.9140   0.9132   0.9235   0.9063   0.9045   0.9220   <--
0.20       0.9140   0.9132   0.9235   0.9063   0.9045   0.9220  
0.25       0.9140   0.9132   0.9235   0.9063   0.9045   0.9220  
0.30       0.9140   0.9132   0.9235   0.9063   0.9045   0.9220  
0.35       0.9140   0.9132   0.9235   0.9063   0.9045   0.9220  
0.40       0.9140   0.9132   0.9235   0.9063   0.9045   0.9220  
0.45       0.9140   0.9132   0.9235   0.9063   0.9045   0.9220  
0.50       0.9140   0.9132   0.9235   0.9063   0.9045   0.9220  
0.55       0.9140   0.9132   0.9235   0.9063   0.9045   0.9220  
0.60       0.9140   0.9132   0.9235   0.9063   0.9045   0.9220  
0.65       0.9140   0.9132   0.9235   0.9063   0.9045   0.9220  
0.70       0.9140   0.9132   0.9235   0.9063   0.9045   0.9220  
0.75       0.9140   0.9132   0.9235   0.9063   0.9045   0.9220  
0.80       0.9140   0.9132   0.9235   0.9063   0.9045   0.9220  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9140, F1=0.9132, Normal Recall=0.9235, Normal Precision=0.9063, Attack Recall=0.9045, Attack Precision=0.9220

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
0.15       0.7700   0.4625   0.7456   0.9984   0.9895   0.3017   <--
0.20       0.7700   0.4625   0.7456   0.9984   0.9895   0.3017  
0.25       0.7700   0.4625   0.7456   0.9984   0.9895   0.3017  
0.30       0.7700   0.4625   0.7456   0.9984   0.9895   0.3017  
0.35       0.7700   0.4625   0.7456   0.9984   0.9895   0.3017  
0.40       0.7700   0.4625   0.7456   0.9984   0.9895   0.3017  
0.45       0.7700   0.4625   0.7456   0.9984   0.9895   0.3017  
0.50       0.7700   0.4625   0.7456   0.9984   0.9895   0.3017  
0.55       0.7700   0.4625   0.7456   0.9984   0.9895   0.3017  
0.60       0.7700   0.4625   0.7456   0.9984   0.9895   0.3017  
0.65       0.7700   0.4625   0.7456   0.9984   0.9895   0.3017  
0.70       0.7700   0.4625   0.7456   0.9984   0.9895   0.3017  
0.75       0.7700   0.4625   0.7456   0.9984   0.9895   0.3017  
0.80       0.7700   0.4625   0.7456   0.9984   0.9895   0.3017  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7700, F1=0.4625, Normal Recall=0.7456, Normal Precision=0.9984, Attack Recall=0.9895, Attack Precision=0.3017

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
0.15       0.7946   0.6584   0.7459   0.9965   0.9895   0.4933   <--
0.20       0.7946   0.6584   0.7459   0.9965   0.9895   0.4933  
0.25       0.7946   0.6584   0.7459   0.9965   0.9895   0.4933  
0.30       0.7946   0.6584   0.7459   0.9965   0.9895   0.4933  
0.35       0.7946   0.6584   0.7459   0.9965   0.9895   0.4933  
0.40       0.7946   0.6584   0.7459   0.9965   0.9895   0.4933  
0.45       0.7946   0.6584   0.7459   0.9965   0.9895   0.4933  
0.50       0.7946   0.6584   0.7459   0.9965   0.9895   0.4933  
0.55       0.7946   0.6584   0.7459   0.9965   0.9895   0.4933  
0.60       0.7946   0.6584   0.7459   0.9965   0.9895   0.4933  
0.65       0.7946   0.6584   0.7459   0.9965   0.9895   0.4933  
0.70       0.7946   0.6584   0.7459   0.9965   0.9895   0.4933  
0.75       0.7946   0.6584   0.7459   0.9965   0.9895   0.4933  
0.80       0.7946   0.6584   0.7459   0.9965   0.9895   0.4933  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7946, F1=0.6584, Normal Recall=0.7459, Normal Precision=0.9965, Attack Recall=0.9895, Attack Precision=0.4933

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
0.15       0.8187   0.7661   0.7456   0.9940   0.9895   0.6250   <--
0.20       0.8187   0.7661   0.7456   0.9940   0.9895   0.6250  
0.25       0.8187   0.7661   0.7456   0.9940   0.9895   0.6250  
0.30       0.8187   0.7661   0.7456   0.9940   0.9895   0.6250  
0.35       0.8187   0.7661   0.7456   0.9940   0.9895   0.6250  
0.40       0.8187   0.7661   0.7456   0.9940   0.9895   0.6250  
0.45       0.8187   0.7661   0.7456   0.9940   0.9895   0.6250  
0.50       0.8187   0.7661   0.7456   0.9940   0.9895   0.6250  
0.55       0.8187   0.7661   0.7456   0.9940   0.9895   0.6250  
0.60       0.8187   0.7661   0.7456   0.9940   0.9895   0.6250  
0.65       0.8187   0.7661   0.7456   0.9940   0.9895   0.6250  
0.70       0.8187   0.7661   0.7456   0.9940   0.9895   0.6250  
0.75       0.8187   0.7661   0.7456   0.9940   0.9895   0.6250  
0.80       0.8187   0.7661   0.7456   0.9940   0.9895   0.6250  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8187, F1=0.7661, Normal Recall=0.7456, Normal Precision=0.9940, Attack Recall=0.9895, Attack Precision=0.6250

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
0.15       0.8430   0.8344   0.7453   0.9907   0.9895   0.7214   <--
0.20       0.8430   0.8344   0.7453   0.9907   0.9895   0.7214  
0.25       0.8430   0.8344   0.7453   0.9907   0.9895   0.7214  
0.30       0.8430   0.8344   0.7453   0.9907   0.9895   0.7214  
0.35       0.8430   0.8344   0.7453   0.9907   0.9895   0.7214  
0.40       0.8430   0.8344   0.7453   0.9907   0.9895   0.7214  
0.45       0.8430   0.8344   0.7453   0.9907   0.9895   0.7214  
0.50       0.8430   0.8344   0.7453   0.9907   0.9895   0.7214  
0.55       0.8430   0.8344   0.7453   0.9907   0.9895   0.7214  
0.60       0.8430   0.8344   0.7453   0.9907   0.9895   0.7214  
0.65       0.8430   0.8344   0.7453   0.9907   0.9895   0.7214  
0.70       0.8430   0.8344   0.7453   0.9907   0.9895   0.7214  
0.75       0.8430   0.8344   0.7453   0.9907   0.9895   0.7214  
0.80       0.8430   0.8344   0.7453   0.9907   0.9895   0.7214  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8430, F1=0.8344, Normal Recall=0.7453, Normal Precision=0.9907, Attack Recall=0.9895, Attack Precision=0.7214

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
0.15       0.8664   0.8810   0.7433   0.9860   0.9895   0.7940   <--
0.20       0.8664   0.8810   0.7433   0.9860   0.9895   0.7940  
0.25       0.8664   0.8810   0.7433   0.9860   0.9895   0.7940  
0.30       0.8664   0.8810   0.7433   0.9860   0.9895   0.7940  
0.35       0.8664   0.8810   0.7433   0.9860   0.9895   0.7940  
0.40       0.8664   0.8810   0.7433   0.9860   0.9895   0.7940  
0.45       0.8664   0.8810   0.7433   0.9860   0.9895   0.7940  
0.50       0.8664   0.8810   0.7433   0.9860   0.9895   0.7940  
0.55       0.8664   0.8810   0.7433   0.9860   0.9895   0.7940  
0.60       0.8664   0.8810   0.7433   0.9860   0.9895   0.7940  
0.65       0.8664   0.8810   0.7433   0.9860   0.9895   0.7940  
0.70       0.8664   0.8810   0.7433   0.9860   0.9895   0.7940  
0.75       0.8664   0.8810   0.7433   0.9860   0.9895   0.7940  
0.80       0.8664   0.8810   0.7433   0.9860   0.9895   0.7940  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8664, F1=0.8810, Normal Recall=0.7433, Normal Precision=0.9860, Attack Recall=0.9895, Attack Precision=0.7940

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
0.15       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013   <--
0.20       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.25       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.30       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.35       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.40       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.45       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.50       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.55       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.60       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.65       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.70       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.75       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.80       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7695, F1=0.4620, Normal Recall=0.7450, Normal Precision=0.9985, Attack Recall=0.9897, Attack Precision=0.3013

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
0.15       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928   <--
0.20       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.25       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.30       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.35       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.40       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.45       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.50       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.55       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.60       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.65       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.70       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.75       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.80       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7942, F1=0.6580, Normal Recall=0.7454, Normal Precision=0.9965, Attack Recall=0.9897, Attack Precision=0.4928

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
0.15       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246   <--
0.20       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.25       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.30       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.35       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.40       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.45       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.50       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.55       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.60       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.65       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.70       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.75       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.80       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8185, F1=0.7659, Normal Recall=0.7451, Normal Precision=0.9941, Attack Recall=0.9897, Attack Precision=0.6246

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
0.15       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212   <--
0.20       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.25       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.30       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.35       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.40       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.45       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.50       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.55       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.60       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.65       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.70       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.75       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.80       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8429, F1=0.8344, Normal Recall=0.7450, Normal Precision=0.9908, Attack Recall=0.9897, Attack Precision=0.7212

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
0.15       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939   <--
0.20       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.25       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.30       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.35       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.40       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.45       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.50       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.55       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.60       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.65       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.70       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.75       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.80       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8663, F1=0.8810, Normal Recall=0.7430, Normal Precision=0.9863, Attack Recall=0.9897, Attack Precision=0.7939

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
0.15       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013   <--
0.20       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.25       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.30       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.35       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.40       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.45       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.50       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.55       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.60       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.65       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.70       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.75       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
0.80       0.7695   0.4620   0.7450   0.9985   0.9897   0.3013  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7695, F1=0.4620, Normal Recall=0.7450, Normal Precision=0.9985, Attack Recall=0.9897, Attack Precision=0.3013

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
0.15       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928   <--
0.20       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.25       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.30       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.35       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.40       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.45       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.50       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.55       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.60       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.65       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.70       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.75       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
0.80       0.7942   0.6580   0.7454   0.9965   0.9897   0.4928  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7942, F1=0.6580, Normal Recall=0.7454, Normal Precision=0.9965, Attack Recall=0.9897, Attack Precision=0.4928

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
0.15       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246   <--
0.20       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.25       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.30       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.35       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.40       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.45       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.50       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.55       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.60       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.65       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.70       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.75       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
0.80       0.8185   0.7659   0.7451   0.9941   0.9897   0.6246  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8185, F1=0.7659, Normal Recall=0.7451, Normal Precision=0.9941, Attack Recall=0.9897, Attack Precision=0.6246

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
0.15       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212   <--
0.20       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.25       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.30       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.35       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.40       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.45       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.50       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.55       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.60       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.65       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.70       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.75       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
0.80       0.8429   0.8344   0.7450   0.9908   0.9897   0.7212  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8429, F1=0.8344, Normal Recall=0.7450, Normal Precision=0.9908, Attack Recall=0.9897, Attack Precision=0.7212

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
0.15       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939   <--
0.20       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.25       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.30       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.35       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.40       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.45       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.50       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.55       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.60       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.65       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.70       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.75       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
0.80       0.8663   0.8810   0.7430   0.9863   0.9897   0.7939  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8663, F1=0.8810, Normal Recall=0.7430, Normal Precision=0.9863, Attack Recall=0.9897, Attack Precision=0.7939

```

