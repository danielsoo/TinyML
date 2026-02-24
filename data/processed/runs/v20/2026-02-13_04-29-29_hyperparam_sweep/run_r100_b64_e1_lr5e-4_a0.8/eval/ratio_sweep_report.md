# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e1_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-19 15:17:13 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2426 | 0.3182 | 0.3934 | 0.4683 | 0.5440 | 0.6175 | 0.6939 | 0.7682 | 0.8417 | 0.9172 | 0.9920 |
| QAT+Prune only | 0.7883 | 0.8092 | 0.8294 | 0.8513 | 0.8716 | 0.8915 | 0.9137 | 0.9346 | 0.9549 | 0.9765 | 0.9975 |
| QAT+PTQ | 0.7870 | 0.8081 | 0.8284 | 0.8505 | 0.8709 | 0.8909 | 0.9133 | 0.9344 | 0.9547 | 0.9765 | 0.9976 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7870 | 0.8081 | 0.8284 | 0.8505 | 0.8709 | 0.8909 | 0.9133 | 0.9344 | 0.9547 | 0.9765 | 0.9976 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2256 | 0.3955 | 0.5282 | 0.6351 | 0.7217 | 0.7954 | 0.8570 | 0.9093 | 0.9557 | 0.9960 |
| QAT+Prune only | 0.0000 | 0.5111 | 0.7005 | 0.8010 | 0.8614 | 0.9019 | 0.9327 | 0.9553 | 0.9725 | 0.9871 | 0.9988 |
| QAT+PTQ | 0.0000 | 0.5097 | 0.6993 | 0.8001 | 0.8607 | 0.9014 | 0.9325 | 0.9551 | 0.9724 | 0.9871 | 0.9988 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5097 | 0.6993 | 0.8001 | 0.8607 | 0.9014 | 0.9325 | 0.9551 | 0.9724 | 0.9871 | 0.9988 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2426 | 0.2432 | 0.2438 | 0.2439 | 0.2454 | 0.2431 | 0.2467 | 0.2460 | 0.2403 | 0.2441 | 0.0000 |
| QAT+Prune only | 0.7883 | 0.7882 | 0.7873 | 0.7886 | 0.7877 | 0.7854 | 0.7879 | 0.7879 | 0.7845 | 0.7873 | 0.0000 |
| QAT+PTQ | 0.7870 | 0.7870 | 0.7861 | 0.7874 | 0.7864 | 0.7843 | 0.7868 | 0.7868 | 0.7829 | 0.7864 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7870 | 0.7870 | 0.7861 | 0.7874 | 0.7864 | 0.7843 | 0.7868 | 0.7868 | 0.7829 | 0.7864 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2426 | 0.0000 | 0.0000 | 0.0000 | 0.2426 | 1.0000 |
| 90 | 10 | 299,940 | 0.3182 | 0.1272 | 0.9929 | 0.2256 | 0.2432 | 0.9968 |
| 80 | 20 | 291,350 | 0.3934 | 0.2470 | 0.9920 | 0.3955 | 0.2438 | 0.9919 |
| 70 | 30 | 194,230 | 0.4683 | 0.3599 | 0.9920 | 0.5282 | 0.2439 | 0.9861 |
| 60 | 40 | 145,675 | 0.5440 | 0.4671 | 0.9920 | 0.6351 | 0.2454 | 0.9787 |
| 50 | 50 | 116,540 | 0.6175 | 0.5672 | 0.9920 | 0.7217 | 0.2431 | 0.9681 |
| 40 | 60 | 97,115 | 0.6939 | 0.6639 | 0.9920 | 0.7954 | 0.2467 | 0.9536 |
| 30 | 70 | 83,240 | 0.7682 | 0.7543 | 0.9920 | 0.8570 | 0.2460 | 0.9295 |
| 20 | 80 | 72,835 | 0.8417 | 0.8393 | 0.9920 | 0.9093 | 0.2403 | 0.8825 |
| 10 | 90 | 64,740 | 0.9172 | 0.9219 | 0.9920 | 0.9557 | 0.2441 | 0.7722 |
| 0 | 100 | 58,270 | 0.9920 | 1.0000 | 0.9920 | 0.9960 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7883 | 0.0000 | 0.0000 | 0.0000 | 0.7883 | 1.0000 |
| 90 | 10 | 299,940 | 0.8092 | 0.3436 | 0.9975 | 0.5111 | 0.7882 | 0.9997 |
| 80 | 20 | 291,350 | 0.8294 | 0.5397 | 0.9975 | 0.7005 | 0.7873 | 0.9992 |
| 70 | 30 | 194,230 | 0.8513 | 0.6692 | 0.9975 | 0.8010 | 0.7886 | 0.9987 |
| 60 | 40 | 145,675 | 0.8716 | 0.7580 | 0.9975 | 0.8614 | 0.7877 | 0.9979 |
| 50 | 50 | 116,540 | 0.8915 | 0.8230 | 0.9975 | 0.9019 | 0.7854 | 0.9969 |
| 40 | 60 | 97,115 | 0.9137 | 0.8759 | 0.9975 | 0.9327 | 0.7879 | 0.9953 |
| 30 | 70 | 83,240 | 0.9346 | 0.9165 | 0.9975 | 0.9553 | 0.7879 | 0.9928 |
| 20 | 80 | 72,835 | 0.9549 | 0.9488 | 0.9975 | 0.9725 | 0.7845 | 0.9876 |
| 10 | 90 | 64,740 | 0.9765 | 0.9769 | 0.9975 | 0.9871 | 0.7873 | 0.9727 |
| 0 | 100 | 58,270 | 0.9975 | 1.0000 | 0.9975 | 0.9988 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7870 | 0.0000 | 0.0000 | 0.0000 | 0.7870 | 1.0000 |
| 90 | 10 | 299,940 | 0.8081 | 0.3423 | 0.9976 | 0.5097 | 0.7870 | 0.9997 |
| 80 | 20 | 291,350 | 0.8284 | 0.5383 | 0.9976 | 0.6993 | 0.7861 | 0.9992 |
| 70 | 30 | 194,230 | 0.8505 | 0.6679 | 0.9976 | 0.8001 | 0.7874 | 0.9987 |
| 60 | 40 | 145,675 | 0.8709 | 0.7569 | 0.9976 | 0.8607 | 0.7864 | 0.9980 |
| 50 | 50 | 116,540 | 0.8909 | 0.8222 | 0.9976 | 0.9014 | 0.7843 | 0.9969 |
| 40 | 60 | 97,115 | 0.9133 | 0.8753 | 0.9976 | 0.9325 | 0.7868 | 0.9954 |
| 30 | 70 | 83,240 | 0.9344 | 0.9161 | 0.9976 | 0.9551 | 0.7868 | 0.9929 |
| 20 | 80 | 72,835 | 0.9547 | 0.9484 | 0.9976 | 0.9724 | 0.7829 | 0.9879 |
| 10 | 90 | 64,740 | 0.9765 | 0.9768 | 0.9976 | 0.9871 | 0.7864 | 0.9732 |
| 0 | 100 | 58,270 | 0.9976 | 1.0000 | 0.9976 | 0.9988 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7870 | 0.0000 | 0.0000 | 0.0000 | 0.7870 | 1.0000 |
| 90 | 10 | 299,940 | 0.8081 | 0.3423 | 0.9976 | 0.5097 | 0.7870 | 0.9997 |
| 80 | 20 | 291,350 | 0.8284 | 0.5383 | 0.9976 | 0.6993 | 0.7861 | 0.9992 |
| 70 | 30 | 194,230 | 0.8505 | 0.6679 | 0.9976 | 0.8001 | 0.7874 | 0.9987 |
| 60 | 40 | 145,675 | 0.8709 | 0.7569 | 0.9976 | 0.8607 | 0.7864 | 0.9980 |
| 50 | 50 | 116,540 | 0.8909 | 0.8222 | 0.9976 | 0.9014 | 0.7843 | 0.9969 |
| 40 | 60 | 97,115 | 0.9133 | 0.8753 | 0.9976 | 0.9325 | 0.7868 | 0.9954 |
| 30 | 70 | 83,240 | 0.9344 | 0.9161 | 0.9976 | 0.9551 | 0.7868 | 0.9929 |
| 20 | 80 | 72,835 | 0.9547 | 0.9484 | 0.9976 | 0.9724 | 0.7829 | 0.9879 |
| 10 | 90 | 64,740 | 0.9765 | 0.9768 | 0.9976 | 0.9871 | 0.7864 | 0.9732 |
| 0 | 100 | 58,270 | 0.9976 | 1.0000 | 0.9976 | 0.9988 | 0.0000 | 0.0000 |


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
0.15       0.3181   0.2254   0.2432   0.9964   0.9922   0.1272   <--
0.20       0.3181   0.2254   0.2432   0.9964   0.9922   0.1272  
0.25       0.3181   0.2254   0.2432   0.9964   0.9922   0.1272  
0.30       0.3181   0.2254   0.2432   0.9964   0.9922   0.1272  
0.35       0.3181   0.2254   0.2432   0.9964   0.9922   0.1272  
0.40       0.3181   0.2254   0.2432   0.9964   0.9922   0.1272  
0.45       0.3181   0.2254   0.2432   0.9964   0.9922   0.1272  
0.50       0.3181   0.2254   0.2432   0.9964   0.9922   0.1272  
0.55       0.3181   0.2254   0.2432   0.9964   0.9922   0.1272  
0.60       0.3181   0.2254   0.2432   0.9964   0.9922   0.1272  
0.65       0.3181   0.2254   0.2432   0.9964   0.9922   0.1272  
0.70       0.3181   0.2254   0.2432   0.9964   0.9922   0.1272  
0.75       0.3181   0.2254   0.2432   0.9964   0.9922   0.1272  
0.80       0.3181   0.2254   0.2432   0.9964   0.9922   0.1272  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3181, F1=0.2254, Normal Recall=0.2432, Normal Precision=0.9964, Attack Recall=0.9922, Attack Precision=0.1272

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
0.15       0.3928   0.3952   0.2430   0.9918   0.9920   0.2468   <--
0.20       0.3928   0.3952   0.2430   0.9918   0.9920   0.2468  
0.25       0.3928   0.3952   0.2430   0.9918   0.9920   0.2468  
0.30       0.3928   0.3952   0.2430   0.9918   0.9920   0.2468  
0.35       0.3928   0.3952   0.2430   0.9918   0.9920   0.2468  
0.40       0.3928   0.3952   0.2430   0.9918   0.9920   0.2468  
0.45       0.3928   0.3952   0.2430   0.9918   0.9920   0.2468  
0.50       0.3928   0.3952   0.2430   0.9918   0.9920   0.2468  
0.55       0.3928   0.3952   0.2430   0.9918   0.9920   0.2468  
0.60       0.3928   0.3952   0.2430   0.9918   0.9920   0.2468  
0.65       0.3928   0.3952   0.2430   0.9918   0.9920   0.2468  
0.70       0.3928   0.3952   0.2430   0.9918   0.9920   0.2468  
0.75       0.3928   0.3952   0.2430   0.9918   0.9920   0.2468  
0.80       0.3928   0.3952   0.2430   0.9918   0.9920   0.2468  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3928, F1=0.3952, Normal Recall=0.2430, Normal Precision=0.9918, Attack Recall=0.9920, Attack Precision=0.2468

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
0.15       0.4680   0.5280   0.2434   0.9861   0.9920   0.3597   <--
0.20       0.4680   0.5280   0.2434   0.9861   0.9920   0.3597  
0.25       0.4680   0.5280   0.2434   0.9861   0.9920   0.3597  
0.30       0.4680   0.5280   0.2434   0.9861   0.9920   0.3597  
0.35       0.4680   0.5280   0.2434   0.9861   0.9920   0.3597  
0.40       0.4680   0.5280   0.2434   0.9861   0.9920   0.3597  
0.45       0.4680   0.5280   0.2434   0.9861   0.9920   0.3597  
0.50       0.4680   0.5280   0.2434   0.9861   0.9920   0.3597  
0.55       0.4680   0.5280   0.2434   0.9861   0.9920   0.3597  
0.60       0.4680   0.5280   0.2434   0.9861   0.9920   0.3597  
0.65       0.4680   0.5280   0.2434   0.9861   0.9920   0.3597  
0.70       0.4680   0.5280   0.2434   0.9861   0.9920   0.3597  
0.75       0.4680   0.5280   0.2434   0.9861   0.9920   0.3597  
0.80       0.4680   0.5280   0.2434   0.9861   0.9920   0.3597  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4680, F1=0.5280, Normal Recall=0.2434, Normal Precision=0.9861, Attack Recall=0.9920, Attack Precision=0.3597

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
0.15       0.5425   0.6343   0.2428   0.9785   0.9920   0.4662   <--
0.20       0.5425   0.6343   0.2428   0.9785   0.9920   0.4662  
0.25       0.5425   0.6343   0.2428   0.9785   0.9920   0.4662  
0.30       0.5425   0.6343   0.2428   0.9785   0.9920   0.4662  
0.35       0.5425   0.6343   0.2428   0.9785   0.9920   0.4662  
0.40       0.5425   0.6343   0.2428   0.9785   0.9920   0.4662  
0.45       0.5425   0.6343   0.2428   0.9785   0.9920   0.4662  
0.50       0.5425   0.6343   0.2428   0.9785   0.9920   0.4662  
0.55       0.5425   0.6343   0.2428   0.9785   0.9920   0.4662  
0.60       0.5425   0.6343   0.2428   0.9785   0.9920   0.4662  
0.65       0.5425   0.6343   0.2428   0.9785   0.9920   0.4662  
0.70       0.5425   0.6343   0.2428   0.9785   0.9920   0.4662  
0.75       0.5425   0.6343   0.2428   0.9785   0.9920   0.4662  
0.80       0.5425   0.6343   0.2428   0.9785   0.9920   0.4662  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5425, F1=0.6343, Normal Recall=0.2428, Normal Precision=0.9785, Attack Recall=0.9920, Attack Precision=0.4662

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
0.15       0.6170   0.7214   0.2419   0.9680   0.9920   0.5668   <--
0.20       0.6170   0.7214   0.2419   0.9680   0.9920   0.5668  
0.25       0.6170   0.7214   0.2419   0.9680   0.9920   0.5668  
0.30       0.6170   0.7214   0.2419   0.9680   0.9920   0.5668  
0.35       0.6170   0.7214   0.2419   0.9680   0.9920   0.5668  
0.40       0.6170   0.7214   0.2419   0.9680   0.9920   0.5668  
0.45       0.6170   0.7214   0.2419   0.9680   0.9920   0.5668  
0.50       0.6170   0.7214   0.2419   0.9680   0.9920   0.5668  
0.55       0.6170   0.7214   0.2419   0.9680   0.9920   0.5668  
0.60       0.6170   0.7214   0.2419   0.9680   0.9920   0.5668  
0.65       0.6170   0.7214   0.2419   0.9680   0.9920   0.5668  
0.70       0.6170   0.7214   0.2419   0.9680   0.9920   0.5668  
0.75       0.6170   0.7214   0.2419   0.9680   0.9920   0.5668  
0.80       0.6170   0.7214   0.2419   0.9680   0.9920   0.5668  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6170, F1=0.7214, Normal Recall=0.2419, Normal Precision=0.9680, Attack Recall=0.9920, Attack Precision=0.5668

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
0.15       0.8092   0.5111   0.7882   0.9996   0.9974   0.3435   <--
0.20       0.8092   0.5111   0.7882   0.9996   0.9974   0.3435  
0.25       0.8092   0.5111   0.7882   0.9996   0.9974   0.3435  
0.30       0.8092   0.5111   0.7882   0.9996   0.9974   0.3435  
0.35       0.8092   0.5111   0.7882   0.9996   0.9974   0.3435  
0.40       0.8092   0.5111   0.7882   0.9996   0.9974   0.3435  
0.45       0.8092   0.5111   0.7882   0.9996   0.9974   0.3435  
0.50       0.8092   0.5111   0.7882   0.9996   0.9974   0.3435  
0.55       0.8092   0.5111   0.7882   0.9996   0.9974   0.3435  
0.60       0.8092   0.5111   0.7882   0.9996   0.9974   0.3435  
0.65       0.8092   0.5111   0.7882   0.9996   0.9974   0.3435  
0.70       0.8092   0.5111   0.7882   0.9996   0.9974   0.3435  
0.75       0.8092   0.5111   0.7882   0.9996   0.9974   0.3435  
0.80       0.8092   0.5111   0.7882   0.9996   0.9974   0.3435  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8092, F1=0.5111, Normal Recall=0.7882, Normal Precision=0.9996, Attack Recall=0.9974, Attack Precision=0.3435

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
0.15       0.8306   0.7019   0.7888   0.9992   0.9975   0.5415   <--
0.20       0.8306   0.7019   0.7888   0.9992   0.9975   0.5415  
0.25       0.8306   0.7019   0.7888   0.9992   0.9975   0.5415  
0.30       0.8306   0.7019   0.7888   0.9992   0.9975   0.5415  
0.35       0.8306   0.7019   0.7888   0.9992   0.9975   0.5415  
0.40       0.8306   0.7019   0.7888   0.9992   0.9975   0.5415  
0.45       0.8306   0.7019   0.7888   0.9992   0.9975   0.5415  
0.50       0.8306   0.7019   0.7888   0.9992   0.9975   0.5415  
0.55       0.8306   0.7019   0.7888   0.9992   0.9975   0.5415  
0.60       0.8306   0.7019   0.7888   0.9992   0.9975   0.5415  
0.65       0.8306   0.7019   0.7888   0.9992   0.9975   0.5415  
0.70       0.8306   0.7019   0.7888   0.9992   0.9975   0.5415  
0.75       0.8306   0.7019   0.7888   0.9992   0.9975   0.5415  
0.80       0.8306   0.7019   0.7888   0.9992   0.9975   0.5415  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8306, F1=0.7019, Normal Recall=0.7888, Normal Precision=0.9992, Attack Recall=0.9975, Attack Precision=0.5415

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
0.15       0.8510   0.8006   0.7881   0.9987   0.9975   0.6686   <--
0.20       0.8510   0.8006   0.7881   0.9987   0.9975   0.6686  
0.25       0.8510   0.8006   0.7881   0.9987   0.9975   0.6686  
0.30       0.8510   0.8006   0.7881   0.9987   0.9975   0.6686  
0.35       0.8510   0.8006   0.7881   0.9987   0.9975   0.6686  
0.40       0.8510   0.8006   0.7881   0.9987   0.9975   0.6686  
0.45       0.8510   0.8006   0.7881   0.9987   0.9975   0.6686  
0.50       0.8510   0.8006   0.7881   0.9987   0.9975   0.6686  
0.55       0.8510   0.8006   0.7881   0.9987   0.9975   0.6686  
0.60       0.8510   0.8006   0.7881   0.9987   0.9975   0.6686  
0.65       0.8510   0.8006   0.7881   0.9987   0.9975   0.6686  
0.70       0.8510   0.8006   0.7881   0.9987   0.9975   0.6686  
0.75       0.8510   0.8006   0.7881   0.9987   0.9975   0.6686  
0.80       0.8510   0.8006   0.7881   0.9987   0.9975   0.6686  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8510, F1=0.8006, Normal Recall=0.7881, Normal Precision=0.9987, Attack Recall=0.9975, Attack Precision=0.6686

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
0.15       0.8717   0.8615   0.7877   0.9979   0.9975   0.7581   <--
0.20       0.8717   0.8615   0.7877   0.9979   0.9975   0.7581  
0.25       0.8717   0.8615   0.7877   0.9979   0.9975   0.7581  
0.30       0.8717   0.8615   0.7877   0.9979   0.9975   0.7581  
0.35       0.8717   0.8615   0.7877   0.9979   0.9975   0.7581  
0.40       0.8717   0.8615   0.7877   0.9979   0.9975   0.7581  
0.45       0.8717   0.8615   0.7877   0.9979   0.9975   0.7581  
0.50       0.8717   0.8615   0.7877   0.9979   0.9975   0.7581  
0.55       0.8717   0.8615   0.7877   0.9979   0.9975   0.7581  
0.60       0.8717   0.8615   0.7877   0.9979   0.9975   0.7581  
0.65       0.8717   0.8615   0.7877   0.9979   0.9975   0.7581  
0.70       0.8717   0.8615   0.7877   0.9979   0.9975   0.7581  
0.75       0.8717   0.8615   0.7877   0.9979   0.9975   0.7581  
0.80       0.8717   0.8615   0.7877   0.9979   0.9975   0.7581  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8717, F1=0.8615, Normal Recall=0.7877, Normal Precision=0.9979, Attack Recall=0.9975, Attack Precision=0.7581

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
0.15       0.8915   0.9019   0.7856   0.9969   0.9975   0.8231   <--
0.20       0.8915   0.9019   0.7856   0.9969   0.9975   0.8231  
0.25       0.8915   0.9019   0.7856   0.9969   0.9975   0.8231  
0.30       0.8915   0.9019   0.7856   0.9969   0.9975   0.8231  
0.35       0.8915   0.9019   0.7856   0.9969   0.9975   0.8231  
0.40       0.8915   0.9019   0.7856   0.9969   0.9975   0.8231  
0.45       0.8915   0.9019   0.7856   0.9969   0.9975   0.8231  
0.50       0.8915   0.9019   0.7856   0.9969   0.9975   0.8231  
0.55       0.8915   0.9019   0.7856   0.9969   0.9975   0.8231  
0.60       0.8915   0.9019   0.7856   0.9969   0.9975   0.8231  
0.65       0.8915   0.9019   0.7856   0.9969   0.9975   0.8231  
0.70       0.8915   0.9019   0.7856   0.9969   0.9975   0.8231  
0.75       0.8915   0.9019   0.7856   0.9969   0.9975   0.8231  
0.80       0.8915   0.9019   0.7856   0.9969   0.9975   0.8231  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8915, F1=0.9019, Normal Recall=0.7856, Normal Precision=0.9969, Attack Recall=0.9975, Attack Precision=0.8231

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
0.15       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423   <--
0.20       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.25       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.30       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.35       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.40       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.45       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.50       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.55       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.60       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.65       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.70       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.75       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.80       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8081, F1=0.5097, Normal Recall=0.7870, Normal Precision=0.9997, Attack Recall=0.9975, Attack Precision=0.3423

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
0.15       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402   <--
0.20       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.25       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.30       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.35       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.40       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.45       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.50       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.55       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.60       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.65       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.70       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.75       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.80       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8297, F1=0.7008, Normal Recall=0.7877, Normal Precision=0.9992, Attack Recall=0.9976, Attack Precision=0.5402

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
0.15       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673   <--
0.20       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.25       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.30       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.35       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.40       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.45       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.50       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.55       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.60       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.65       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.70       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.75       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.80       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8501, F1=0.7997, Normal Recall=0.7868, Normal Precision=0.9987, Attack Recall=0.9976, Attack Precision=0.6673

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
0.15       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570   <--
0.20       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.25       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.30       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.35       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.40       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.45       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.50       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.55       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.60       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.65       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.70       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.75       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.80       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8709, F1=0.8608, Normal Recall=0.7865, Normal Precision=0.9980, Attack Recall=0.9976, Attack Precision=0.7570

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
0.15       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222   <--
0.20       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.25       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.30       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.35       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.40       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.45       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.50       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.55       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.60       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.65       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.70       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.75       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.80       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8909, F1=0.9014, Normal Recall=0.7843, Normal Precision=0.9969, Attack Recall=0.9976, Attack Precision=0.8222

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
0.15       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423   <--
0.20       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.25       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.30       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.35       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.40       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.45       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.50       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.55       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.60       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.65       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.70       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.75       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
0.80       0.8081   0.5097   0.7870   0.9997   0.9975   0.3423  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8081, F1=0.5097, Normal Recall=0.7870, Normal Precision=0.9997, Attack Recall=0.9975, Attack Precision=0.3423

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
0.15       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402   <--
0.20       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.25       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.30       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.35       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.40       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.45       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.50       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.55       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.60       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.65       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.70       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.75       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
0.80       0.8297   0.7008   0.7877   0.9992   0.9976   0.5402  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8297, F1=0.7008, Normal Recall=0.7877, Normal Precision=0.9992, Attack Recall=0.9976, Attack Precision=0.5402

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
0.15       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673   <--
0.20       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.25       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.30       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.35       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.40       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.45       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.50       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.55       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.60       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.65       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.70       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.75       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
0.80       0.8501   0.7997   0.7868   0.9987   0.9976   0.6673  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8501, F1=0.7997, Normal Recall=0.7868, Normal Precision=0.9987, Attack Recall=0.9976, Attack Precision=0.6673

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
0.15       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570   <--
0.20       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.25       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.30       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.35       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.40       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.45       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.50       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.55       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.60       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.65       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.70       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.75       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
0.80       0.8709   0.8608   0.7865   0.9980   0.9976   0.7570  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8709, F1=0.8608, Normal Recall=0.7865, Normal Precision=0.9980, Attack Recall=0.9976, Attack Precision=0.7570

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
0.15       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222   <--
0.20       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.25       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.30       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.35       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.40       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.45       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.50       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.55       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.60       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.65       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.70       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.75       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
0.80       0.8909   0.9014   0.7843   0.9969   0.9976   0.8222  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8909, F1=0.9014, Normal Recall=0.7843, Normal Precision=0.9969, Attack Recall=0.9976, Attack Precision=0.8222

```

