# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e1_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-18 08:22:54 |

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
| Original (TFLite) | 0.9463 | 0.9208 | 0.8945 | 0.8686 | 0.8428 | 0.8164 | 0.7910 | 0.7653 | 0.7394 | 0.7129 | 0.6871 |
| QAT+Prune only | 0.5692 | 0.6074 | 0.6458 | 0.6856 | 0.7225 | 0.7610 | 0.8001 | 0.8391 | 0.8774 | 0.9160 | 0.9550 |
| QAT+PTQ | 0.5690 | 0.6072 | 0.6457 | 0.6856 | 0.7227 | 0.7612 | 0.8006 | 0.8396 | 0.8780 | 0.9167 | 0.9558 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.5690 | 0.6072 | 0.6457 | 0.6856 | 0.7227 | 0.7612 | 0.8006 | 0.8396 | 0.8780 | 0.9167 | 0.9558 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6347 | 0.7226 | 0.7584 | 0.7776 | 0.7891 | 0.7978 | 0.8039 | 0.8084 | 0.8116 | 0.8145 |
| QAT+Prune only | 0.0000 | 0.3271 | 0.5189 | 0.6457 | 0.7335 | 0.7998 | 0.8515 | 0.8926 | 0.9257 | 0.9534 | 0.9770 |
| QAT+PTQ | 0.0000 | 0.3272 | 0.5190 | 0.6459 | 0.7339 | 0.8001 | 0.8519 | 0.8930 | 0.9261 | 0.9538 | 0.9774 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3272 | 0.5190 | 0.6459 | 0.7339 | 0.8001 | 0.8519 | 0.8930 | 0.9261 | 0.9538 | 0.9774 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9463 | 0.9466 | 0.9464 | 0.9464 | 0.9465 | 0.9457 | 0.9468 | 0.9477 | 0.9484 | 0.9453 | 0.0000 |
| QAT+Prune only | 0.5692 | 0.5688 | 0.5685 | 0.5702 | 0.5675 | 0.5670 | 0.5678 | 0.5688 | 0.5671 | 0.5649 | 0.0000 |
| QAT+PTQ | 0.5690 | 0.5685 | 0.5682 | 0.5698 | 0.5673 | 0.5667 | 0.5678 | 0.5686 | 0.5667 | 0.5644 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.5690 | 0.5685 | 0.5682 | 0.5698 | 0.5673 | 0.5667 | 0.5678 | 0.5686 | 0.5667 | 0.5644 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9463 | 0.0000 | 0.0000 | 0.0000 | 0.9463 | 1.0000 |
| 90 | 10 | 299,940 | 0.9208 | 0.5889 | 0.6882 | 0.6347 | 0.9466 | 0.9647 |
| 80 | 20 | 291,350 | 0.8945 | 0.7621 | 0.6871 | 0.7226 | 0.9464 | 0.9237 |
| 70 | 30 | 194,230 | 0.8686 | 0.8461 | 0.6871 | 0.7584 | 0.9464 | 0.8759 |
| 60 | 40 | 145,675 | 0.8428 | 0.8955 | 0.6871 | 0.7776 | 0.9465 | 0.8194 |
| 50 | 50 | 116,540 | 0.8164 | 0.9267 | 0.6871 | 0.7891 | 0.9457 | 0.7514 |
| 40 | 60 | 97,115 | 0.7910 | 0.9509 | 0.6871 | 0.7978 | 0.9468 | 0.6686 |
| 30 | 70 | 83,240 | 0.7653 | 0.9684 | 0.6871 | 0.8039 | 0.9477 | 0.5648 |
| 20 | 80 | 72,835 | 0.7394 | 0.9816 | 0.6871 | 0.8084 | 0.9484 | 0.4311 |
| 10 | 90 | 64,740 | 0.7129 | 0.9912 | 0.6871 | 0.8116 | 0.9453 | 0.2513 |
| 0 | 100 | 58,270 | 0.6871 | 1.0000 | 0.6871 | 0.8145 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5692 | 0.0000 | 0.0000 | 0.0000 | 0.5692 | 1.0000 |
| 90 | 10 | 299,940 | 0.6074 | 0.1974 | 0.9543 | 0.3271 | 0.5688 | 0.9912 |
| 80 | 20 | 291,350 | 0.6458 | 0.3562 | 0.9550 | 0.5189 | 0.5685 | 0.9806 |
| 70 | 30 | 194,230 | 0.6856 | 0.4877 | 0.9550 | 0.6457 | 0.5702 | 0.9673 |
| 60 | 40 | 145,675 | 0.7225 | 0.5955 | 0.9550 | 0.7335 | 0.5675 | 0.9498 |
| 50 | 50 | 116,540 | 0.7610 | 0.6880 | 0.9550 | 0.7998 | 0.5670 | 0.9264 |
| 40 | 60 | 97,115 | 0.8001 | 0.7682 | 0.9550 | 0.8515 | 0.5678 | 0.8937 |
| 30 | 70 | 83,240 | 0.8391 | 0.8378 | 0.9550 | 0.8926 | 0.5688 | 0.8441 |
| 20 | 80 | 72,835 | 0.8774 | 0.8982 | 0.9550 | 0.9257 | 0.5671 | 0.7589 |
| 10 | 90 | 64,740 | 0.9160 | 0.9518 | 0.9550 | 0.9534 | 0.5649 | 0.5822 |
| 0 | 100 | 58,270 | 0.9550 | 1.0000 | 0.9550 | 0.9770 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5690 | 0.0000 | 0.0000 | 0.0000 | 0.5690 | 1.0000 |
| 90 | 10 | 299,940 | 0.6072 | 0.1974 | 0.9551 | 0.3272 | 0.5685 | 0.9913 |
| 80 | 20 | 291,350 | 0.6457 | 0.3562 | 0.9558 | 0.5190 | 0.5682 | 0.9809 |
| 70 | 30 | 194,230 | 0.6856 | 0.4878 | 0.9558 | 0.6459 | 0.5698 | 0.9678 |
| 60 | 40 | 145,675 | 0.7227 | 0.5956 | 0.9558 | 0.7339 | 0.5673 | 0.9506 |
| 50 | 50 | 116,540 | 0.7612 | 0.6881 | 0.9558 | 0.8001 | 0.5667 | 0.9277 |
| 40 | 60 | 97,115 | 0.8006 | 0.7684 | 0.9558 | 0.8519 | 0.5678 | 0.8955 |
| 30 | 70 | 83,240 | 0.8396 | 0.8379 | 0.9558 | 0.8930 | 0.5686 | 0.8465 |
| 20 | 80 | 72,835 | 0.8780 | 0.8982 | 0.9558 | 0.9261 | 0.5667 | 0.7622 |
| 10 | 90 | 64,740 | 0.9167 | 0.9518 | 0.9558 | 0.9538 | 0.5644 | 0.5866 |
| 0 | 100 | 58,270 | 0.9558 | 1.0000 | 0.9558 | 0.9774 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.5690 | 0.0000 | 0.0000 | 0.0000 | 0.5690 | 1.0000 |
| 90 | 10 | 299,940 | 0.6072 | 0.1974 | 0.9551 | 0.3272 | 0.5685 | 0.9913 |
| 80 | 20 | 291,350 | 0.6457 | 0.3562 | 0.9558 | 0.5190 | 0.5682 | 0.9809 |
| 70 | 30 | 194,230 | 0.6856 | 0.4878 | 0.9558 | 0.6459 | 0.5698 | 0.9678 |
| 60 | 40 | 145,675 | 0.7227 | 0.5956 | 0.9558 | 0.7339 | 0.5673 | 0.9506 |
| 50 | 50 | 116,540 | 0.7612 | 0.6881 | 0.9558 | 0.8001 | 0.5667 | 0.9277 |
| 40 | 60 | 97,115 | 0.8006 | 0.7684 | 0.9558 | 0.8519 | 0.5678 | 0.8955 |
| 30 | 70 | 83,240 | 0.8396 | 0.8379 | 0.9558 | 0.8930 | 0.5686 | 0.8465 |
| 20 | 80 | 72,835 | 0.8780 | 0.8982 | 0.9558 | 0.9261 | 0.5667 | 0.7622 |
| 10 | 90 | 64,740 | 0.9167 | 0.9518 | 0.9558 | 0.9538 | 0.5644 | 0.5866 |
| 0 | 100 | 58,270 | 0.9558 | 1.0000 | 0.9558 | 0.9774 | 0.0000 | 0.0000 |


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
0.15       0.9211   0.6368   0.9466   0.9651   0.6916   0.5901   <--
0.20       0.9211   0.6368   0.9466   0.9651   0.6916   0.5901  
0.25       0.9211   0.6368   0.9466   0.9651   0.6916   0.5901  
0.30       0.9211   0.6368   0.9466   0.9651   0.6916   0.5901  
0.35       0.9211   0.6368   0.9466   0.9651   0.6916   0.5901  
0.40       0.9211   0.6368   0.9466   0.9651   0.6916   0.5901  
0.45       0.9211   0.6368   0.9466   0.9651   0.6916   0.5901  
0.50       0.9211   0.6368   0.9466   0.9651   0.6916   0.5901  
0.55       0.9211   0.6368   0.9466   0.9651   0.6916   0.5901  
0.60       0.9211   0.6368   0.9466   0.9651   0.6916   0.5901  
0.65       0.9211   0.6368   0.9466   0.9651   0.6916   0.5901  
0.70       0.9211   0.6368   0.9466   0.9651   0.6916   0.5901  
0.75       0.9211   0.6368   0.9466   0.9651   0.6916   0.5901  
0.80       0.9211   0.6368   0.9466   0.9651   0.6916   0.5901  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9211, F1=0.6368, Normal Recall=0.9466, Normal Precision=0.9651, Attack Recall=0.6916, Attack Precision=0.5901

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
0.15       0.8949   0.7233   0.9468   0.9237   0.6871   0.7635   <--
0.20       0.8949   0.7233   0.9468   0.9237   0.6871   0.7635  
0.25       0.8949   0.7233   0.9468   0.9237   0.6871   0.7635  
0.30       0.8949   0.7233   0.9468   0.9237   0.6871   0.7635  
0.35       0.8949   0.7233   0.9468   0.9237   0.6871   0.7635  
0.40       0.8949   0.7233   0.9468   0.9237   0.6871   0.7635  
0.45       0.8949   0.7233   0.9468   0.9237   0.6871   0.7635  
0.50       0.8949   0.7233   0.9468   0.9237   0.6871   0.7635  
0.55       0.8949   0.7233   0.9468   0.9237   0.6871   0.7635  
0.60       0.8949   0.7233   0.9468   0.9237   0.6871   0.7635  
0.65       0.8949   0.7233   0.9468   0.9237   0.6871   0.7635  
0.70       0.8949   0.7233   0.9468   0.9237   0.6871   0.7635  
0.75       0.8949   0.7233   0.9468   0.9237   0.6871   0.7635  
0.80       0.8949   0.7233   0.9468   0.9237   0.6871   0.7635  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8949, F1=0.7233, Normal Recall=0.9468, Normal Precision=0.9237, Attack Recall=0.6871, Attack Precision=0.7635

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
0.15       0.8690   0.7588   0.9469   0.8760   0.6871   0.8472   <--
0.20       0.8690   0.7588   0.9469   0.8760   0.6871   0.8472  
0.25       0.8690   0.7588   0.9469   0.8760   0.6871   0.8472  
0.30       0.8690   0.7588   0.9469   0.8760   0.6871   0.8472  
0.35       0.8690   0.7588   0.9469   0.8760   0.6871   0.8472  
0.40       0.8690   0.7588   0.9469   0.8760   0.6871   0.8472  
0.45       0.8690   0.7588   0.9469   0.8760   0.6871   0.8472  
0.50       0.8690   0.7588   0.9469   0.8760   0.6871   0.8472  
0.55       0.8690   0.7588   0.9469   0.8760   0.6871   0.8472  
0.60       0.8690   0.7588   0.9469   0.8760   0.6871   0.8472  
0.65       0.8690   0.7588   0.9469   0.8760   0.6871   0.8472  
0.70       0.8690   0.7588   0.9469   0.8760   0.6871   0.8472  
0.75       0.8690   0.7588   0.9469   0.8760   0.6871   0.8472  
0.80       0.8690   0.7588   0.9469   0.8760   0.6871   0.8472  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8690, F1=0.7588, Normal Recall=0.9469, Normal Precision=0.8760, Attack Recall=0.6871, Attack Precision=0.8472

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
0.15       0.8426   0.7774   0.9462   0.8194   0.6871   0.8949   <--
0.20       0.8426   0.7774   0.9462   0.8194   0.6871   0.8949  
0.25       0.8426   0.7774   0.9462   0.8194   0.6871   0.8949  
0.30       0.8426   0.7774   0.9462   0.8194   0.6871   0.8949  
0.35       0.8426   0.7774   0.9462   0.8194   0.6871   0.8949  
0.40       0.8426   0.7774   0.9462   0.8194   0.6871   0.8949  
0.45       0.8426   0.7774   0.9462   0.8194   0.6871   0.8949  
0.50       0.8426   0.7774   0.9462   0.8194   0.6871   0.8949  
0.55       0.8426   0.7774   0.9462   0.8194   0.6871   0.8949  
0.60       0.8426   0.7774   0.9462   0.8194   0.6871   0.8949  
0.65       0.8426   0.7774   0.9462   0.8194   0.6871   0.8949  
0.70       0.8426   0.7774   0.9462   0.8194   0.6871   0.8949  
0.75       0.8426   0.7774   0.9462   0.8194   0.6871   0.8949  
0.80       0.8426   0.7774   0.9462   0.8194   0.6871   0.8949  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8426, F1=0.7774, Normal Recall=0.9462, Normal Precision=0.8194, Attack Recall=0.6871, Attack Precision=0.8949

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
0.15       0.8167   0.7894   0.9463   0.7515   0.6871   0.9274   <--
0.20       0.8167   0.7894   0.9463   0.7515   0.6871   0.9274  
0.25       0.8167   0.7894   0.9463   0.7515   0.6871   0.9274  
0.30       0.8167   0.7894   0.9463   0.7515   0.6871   0.9274  
0.35       0.8167   0.7894   0.9463   0.7515   0.6871   0.9274  
0.40       0.8167   0.7894   0.9463   0.7515   0.6871   0.9274  
0.45       0.8167   0.7894   0.9463   0.7515   0.6871   0.9274  
0.50       0.8167   0.7894   0.9463   0.7515   0.6871   0.9274  
0.55       0.8167   0.7894   0.9463   0.7515   0.6871   0.9274  
0.60       0.8167   0.7894   0.9463   0.7515   0.6871   0.9274  
0.65       0.8167   0.7894   0.9463   0.7515   0.6871   0.9274  
0.70       0.8167   0.7894   0.9463   0.7515   0.6871   0.9274  
0.75       0.8167   0.7894   0.9463   0.7515   0.6871   0.9274  
0.80       0.8167   0.7894   0.9463   0.7515   0.6871   0.9274  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8167, F1=0.7894, Normal Recall=0.9463, Normal Precision=0.7515, Attack Recall=0.6871, Attack Precision=0.9274

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
0.15       0.6073   0.3269   0.5689   0.9910   0.9537   0.1973   <--
0.20       0.6073   0.3269   0.5689   0.9910   0.9537   0.1973  
0.25       0.6073   0.3269   0.5689   0.9910   0.9537   0.1973  
0.30       0.6073   0.3269   0.5689   0.9910   0.9537   0.1973  
0.35       0.6073   0.3269   0.5689   0.9910   0.9537   0.1973  
0.40       0.6073   0.3269   0.5689   0.9910   0.9537   0.1973  
0.45       0.6073   0.3269   0.5689   0.9910   0.9537   0.1973  
0.50       0.6073   0.3269   0.5689   0.9910   0.9537   0.1973  
0.55       0.6073   0.3269   0.5689   0.9910   0.9537   0.1973  
0.60       0.6073   0.3269   0.5689   0.9910   0.9537   0.1973  
0.65       0.6073   0.3269   0.5689   0.9910   0.9537   0.1973  
0.70       0.6073   0.3269   0.5689   0.9910   0.9537   0.1973  
0.75       0.6073   0.3269   0.5689   0.9910   0.9537   0.1973  
0.80       0.6073   0.3269   0.5689   0.9910   0.9537   0.1973  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6073, F1=0.3269, Normal Recall=0.5689, Normal Precision=0.9910, Attack Recall=0.9537, Attack Precision=0.1973

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
0.15       0.6467   0.5195   0.5697   0.9806   0.9550   0.3568   <--
0.20       0.6467   0.5195   0.5697   0.9806   0.9550   0.3568  
0.25       0.6467   0.5195   0.5697   0.9806   0.9550   0.3568  
0.30       0.6467   0.5195   0.5697   0.9806   0.9550   0.3568  
0.35       0.6467   0.5195   0.5697   0.9806   0.9550   0.3568  
0.40       0.6467   0.5195   0.5697   0.9806   0.9550   0.3568  
0.45       0.6467   0.5195   0.5697   0.9806   0.9550   0.3568  
0.50       0.6467   0.5195   0.5697   0.9806   0.9550   0.3568  
0.55       0.6467   0.5195   0.5697   0.9806   0.9550   0.3568  
0.60       0.6467   0.5195   0.5697   0.9806   0.9550   0.3568  
0.65       0.6467   0.5195   0.5697   0.9806   0.9550   0.3568  
0.70       0.6467   0.5195   0.5697   0.9806   0.9550   0.3568  
0.75       0.6467   0.5195   0.5697   0.9806   0.9550   0.3568  
0.80       0.6467   0.5195   0.5697   0.9806   0.9550   0.3568  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6467, F1=0.5195, Normal Recall=0.5697, Normal Precision=0.9806, Attack Recall=0.9550, Attack Precision=0.3568

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
0.15       0.6851   0.6454   0.5695   0.9672   0.9550   0.4874   <--
0.20       0.6851   0.6454   0.5695   0.9672   0.9550   0.4874  
0.25       0.6851   0.6454   0.5695   0.9672   0.9550   0.4874  
0.30       0.6851   0.6454   0.5695   0.9672   0.9550   0.4874  
0.35       0.6851   0.6454   0.5695   0.9672   0.9550   0.4874  
0.40       0.6851   0.6454   0.5695   0.9672   0.9550   0.4874  
0.45       0.6851   0.6454   0.5695   0.9672   0.9550   0.4874  
0.50       0.6851   0.6454   0.5695   0.9672   0.9550   0.4874  
0.55       0.6851   0.6454   0.5695   0.9672   0.9550   0.4874  
0.60       0.6851   0.6454   0.5695   0.9672   0.9550   0.4874  
0.65       0.6851   0.6454   0.5695   0.9672   0.9550   0.4874  
0.70       0.6851   0.6454   0.5695   0.9672   0.9550   0.4874  
0.75       0.6851   0.6454   0.5695   0.9672   0.9550   0.4874  
0.80       0.6851   0.6454   0.5695   0.9672   0.9550   0.4874  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6851, F1=0.6454, Normal Recall=0.5695, Normal Precision=0.9672, Attack Recall=0.9550, Attack Precision=0.4874

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
0.15       0.7235   0.7342   0.5691   0.9499   0.9550   0.5964   <--
0.20       0.7235   0.7342   0.5691   0.9499   0.9550   0.5964  
0.25       0.7235   0.7342   0.5691   0.9499   0.9550   0.5964  
0.30       0.7235   0.7342   0.5691   0.9499   0.9550   0.5964  
0.35       0.7235   0.7342   0.5691   0.9499   0.9550   0.5964  
0.40       0.7235   0.7342   0.5691   0.9499   0.9550   0.5964  
0.45       0.7235   0.7342   0.5691   0.9499   0.9550   0.5964  
0.50       0.7235   0.7342   0.5691   0.9499   0.9550   0.5964  
0.55       0.7235   0.7342   0.5691   0.9499   0.9550   0.5964  
0.60       0.7235   0.7342   0.5691   0.9499   0.9550   0.5964  
0.65       0.7235   0.7342   0.5691   0.9499   0.9550   0.5964  
0.70       0.7235   0.7342   0.5691   0.9499   0.9550   0.5964  
0.75       0.7235   0.7342   0.5691   0.9499   0.9550   0.5964  
0.80       0.7235   0.7342   0.5691   0.9499   0.9550   0.5964  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7235, F1=0.7342, Normal Recall=0.5691, Normal Precision=0.9499, Attack Recall=0.9550, Attack Precision=0.5964

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
0.15       0.7613   0.8000   0.5676   0.9265   0.9550   0.6883   <--
0.20       0.7613   0.8000   0.5676   0.9265   0.9550   0.6883  
0.25       0.7613   0.8000   0.5676   0.9265   0.9550   0.6883  
0.30       0.7613   0.8000   0.5676   0.9265   0.9550   0.6883  
0.35       0.7613   0.8000   0.5676   0.9265   0.9550   0.6883  
0.40       0.7613   0.8000   0.5676   0.9265   0.9550   0.6883  
0.45       0.7613   0.8000   0.5676   0.9265   0.9550   0.6883  
0.50       0.7613   0.8000   0.5676   0.9265   0.9550   0.6883  
0.55       0.7613   0.8000   0.5676   0.9265   0.9550   0.6883  
0.60       0.7613   0.8000   0.5676   0.9265   0.9550   0.6883  
0.65       0.7613   0.8000   0.5676   0.9265   0.9550   0.6883  
0.70       0.7613   0.8000   0.5676   0.9265   0.9550   0.6883  
0.75       0.7613   0.8000   0.5676   0.9265   0.9550   0.6883  
0.80       0.7613   0.8000   0.5676   0.9265   0.9550   0.6883  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7613, F1=0.8000, Normal Recall=0.5676, Normal Precision=0.9265, Attack Recall=0.9550, Attack Precision=0.6883

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
0.15       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973   <--
0.20       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.25       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.30       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.35       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.40       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.45       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.50       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.55       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.60       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.65       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.70       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.75       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.80       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6071, F1=0.3269, Normal Recall=0.5685, Normal Precision=0.9911, Attack Recall=0.9543, Attack Precision=0.1973

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
0.15       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569   <--
0.20       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.25       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.30       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.35       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.40       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.45       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.50       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.55       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.60       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.65       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.70       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.75       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.80       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6467, F1=0.5197, Normal Recall=0.5694, Normal Precision=0.9810, Attack Recall=0.9558, Attack Precision=0.3569

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
0.15       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875   <--
0.20       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.25       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.30       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.35       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.40       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.45       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.50       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.55       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.60       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.65       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.70       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.75       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.80       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6852, F1=0.6456, Normal Recall=0.5693, Normal Precision=0.9678, Attack Recall=0.9558, Attack Precision=0.4875

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
0.15       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965   <--
0.20       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.25       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.30       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.35       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.40       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.45       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.50       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.55       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.60       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.65       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.70       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.75       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.80       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7237, F1=0.7346, Normal Recall=0.5690, Normal Precision=0.9508, Attack Recall=0.9558, Attack Precision=0.5965

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
0.15       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884   <--
0.20       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.25       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.30       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.35       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.40       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.45       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.50       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.55       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.60       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.65       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.70       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.75       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.80       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7616, F1=0.8003, Normal Recall=0.5673, Normal Precision=0.9277, Attack Recall=0.9558, Attack Precision=0.6884

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
0.15       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973   <--
0.20       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.25       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.30       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.35       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.40       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.45       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.50       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.55       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.60       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.65       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.70       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.75       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
0.80       0.6071   0.3269   0.5685   0.9911   0.9543   0.1973  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6071, F1=0.3269, Normal Recall=0.5685, Normal Precision=0.9911, Attack Recall=0.9543, Attack Precision=0.1973

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
0.15       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569   <--
0.20       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.25       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.30       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.35       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.40       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.45       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.50       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.55       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.60       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.65       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.70       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.75       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
0.80       0.6467   0.5197   0.5694   0.9810   0.9558   0.3569  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6467, F1=0.5197, Normal Recall=0.5694, Normal Precision=0.9810, Attack Recall=0.9558, Attack Precision=0.3569

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
0.15       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875   <--
0.20       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.25       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.30       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.35       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.40       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.45       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.50       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.55       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.60       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.65       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.70       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.75       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
0.80       0.6852   0.6456   0.5693   0.9678   0.9558   0.4875  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6852, F1=0.6456, Normal Recall=0.5693, Normal Precision=0.9678, Attack Recall=0.9558, Attack Precision=0.4875

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
0.15       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965   <--
0.20       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.25       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.30       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.35       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.40       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.45       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.50       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.55       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.60       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.65       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.70       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.75       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
0.80       0.7237   0.7346   0.5690   0.9508   0.9558   0.5965  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7237, F1=0.7346, Normal Recall=0.5690, Normal Precision=0.9508, Attack Recall=0.9558, Attack Precision=0.5965

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
0.15       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884   <--
0.20       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.25       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.30       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.35       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.40       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.45       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.50       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.55       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.60       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.65       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.70       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.75       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
0.80       0.7616   0.8003   0.5673   0.9277   0.9558   0.6884  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7616, F1=0.8003, Normal Recall=0.5673, Normal Precision=0.9277, Attack Recall=0.9558, Attack Precision=0.6884

```

