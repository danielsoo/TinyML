# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e2_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-20 06:51:17 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5731 | 0.6019 | 0.6316 | 0.6625 | 0.6915 | 0.7245 | 0.7524 | 0.7835 | 0.8114 | 0.8419 | 0.8723 |
| QAT+Prune only | 0.8784 | 0.8854 | 0.8926 | 0.9007 | 0.9077 | 0.9139 | 0.9225 | 0.9301 | 0.9364 | 0.9448 | 0.9518 |
| QAT+PTQ | 0.8778 | 0.8853 | 0.8925 | 0.9008 | 0.9080 | 0.9144 | 0.9229 | 0.9309 | 0.9373 | 0.9458 | 0.9529 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8778 | 0.8853 | 0.8925 | 0.9008 | 0.9080 | 0.9144 | 0.9229 | 0.9309 | 0.9373 | 0.9458 | 0.9529 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3049 | 0.4864 | 0.6080 | 0.6934 | 0.7600 | 0.8087 | 0.8494 | 0.8809 | 0.9085 | 0.9318 |
| QAT+Prune only | 0.0000 | 0.6240 | 0.7800 | 0.8519 | 0.8919 | 0.9171 | 0.9364 | 0.9502 | 0.9599 | 0.9688 | 0.9753 |
| QAT+PTQ | 0.0000 | 0.6240 | 0.7801 | 0.8521 | 0.8923 | 0.9175 | 0.9368 | 0.9508 | 0.9605 | 0.9694 | 0.9759 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.6240 | 0.7801 | 0.8521 | 0.8923 | 0.9175 | 0.9368 | 0.9508 | 0.9605 | 0.9694 | 0.9759 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5731 | 0.5717 | 0.5714 | 0.5726 | 0.5709 | 0.5767 | 0.5726 | 0.5763 | 0.5677 | 0.5687 | 0.0000 |
| QAT+Prune only | 0.8784 | 0.8781 | 0.8778 | 0.8789 | 0.8784 | 0.8761 | 0.8785 | 0.8796 | 0.8749 | 0.8825 | 0.0000 |
| QAT+PTQ | 0.8778 | 0.8778 | 0.8775 | 0.8784 | 0.8780 | 0.8758 | 0.8778 | 0.8796 | 0.8748 | 0.8818 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8778 | 0.8778 | 0.8775 | 0.8784 | 0.8780 | 0.8758 | 0.8778 | 0.8796 | 0.8748 | 0.8818 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5731 | 0.0000 | 0.0000 | 0.0000 | 0.5731 | 1.0000 |
| 90 | 10 | 299,940 | 0.6019 | 0.1847 | 0.8733 | 0.3049 | 0.5717 | 0.9760 |
| 80 | 20 | 291,350 | 0.6316 | 0.3372 | 0.8723 | 0.4864 | 0.5714 | 0.9471 |
| 70 | 30 | 194,230 | 0.6625 | 0.4666 | 0.8723 | 0.6080 | 0.5726 | 0.9127 |
| 60 | 40 | 145,675 | 0.6915 | 0.5754 | 0.8723 | 0.6934 | 0.5709 | 0.8702 |
| 50 | 50 | 116,540 | 0.7245 | 0.6733 | 0.8723 | 0.7600 | 0.5767 | 0.8187 |
| 40 | 60 | 97,115 | 0.7524 | 0.7538 | 0.8723 | 0.8087 | 0.5726 | 0.7493 |
| 30 | 70 | 83,240 | 0.7835 | 0.8277 | 0.8723 | 0.8494 | 0.5763 | 0.6591 |
| 20 | 80 | 72,835 | 0.8114 | 0.8898 | 0.8723 | 0.8809 | 0.5677 | 0.5263 |
| 10 | 90 | 64,740 | 0.8419 | 0.9479 | 0.8723 | 0.9085 | 0.5687 | 0.3310 |
| 0 | 100 | 58,270 | 0.8723 | 1.0000 | 0.8723 | 0.9318 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8784 | 0.0000 | 0.0000 | 0.0000 | 0.8784 | 1.0000 |
| 90 | 10 | 299,940 | 0.8854 | 0.4644 | 0.9509 | 0.6240 | 0.8781 | 0.9938 |
| 80 | 20 | 291,350 | 0.8926 | 0.6607 | 0.9518 | 0.7800 | 0.8778 | 0.9864 |
| 70 | 30 | 194,230 | 0.9007 | 0.7710 | 0.9518 | 0.8519 | 0.8789 | 0.9770 |
| 60 | 40 | 145,675 | 0.9077 | 0.8391 | 0.9518 | 0.8919 | 0.8784 | 0.9647 |
| 50 | 50 | 116,540 | 0.9139 | 0.8848 | 0.9518 | 0.9171 | 0.8761 | 0.9478 |
| 40 | 60 | 97,115 | 0.9225 | 0.9216 | 0.9518 | 0.9364 | 0.8785 | 0.9239 |
| 30 | 70 | 83,240 | 0.9301 | 0.9486 | 0.9518 | 0.9502 | 0.8796 | 0.8865 |
| 20 | 80 | 72,835 | 0.9364 | 0.9682 | 0.9518 | 0.9599 | 0.8749 | 0.8193 |
| 10 | 90 | 64,740 | 0.9448 | 0.9865 | 0.9518 | 0.9688 | 0.8825 | 0.6703 |
| 0 | 100 | 58,270 | 0.9518 | 1.0000 | 0.9518 | 0.9753 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8778 | 0.0000 | 0.0000 | 0.0000 | 0.8778 | 1.0000 |
| 90 | 10 | 299,940 | 0.8853 | 0.4641 | 0.9521 | 0.6240 | 0.8778 | 0.9940 |
| 80 | 20 | 291,350 | 0.8925 | 0.6603 | 0.9529 | 0.7801 | 0.8775 | 0.9868 |
| 70 | 30 | 194,230 | 0.9008 | 0.7706 | 0.9529 | 0.8521 | 0.8784 | 0.9775 |
| 60 | 40 | 145,675 | 0.9080 | 0.8389 | 0.9529 | 0.8923 | 0.8780 | 0.9655 |
| 50 | 50 | 116,540 | 0.9144 | 0.8847 | 0.9529 | 0.9175 | 0.8758 | 0.9490 |
| 40 | 60 | 97,115 | 0.9229 | 0.9213 | 0.9529 | 0.9368 | 0.8778 | 0.9255 |
| 30 | 70 | 83,240 | 0.9309 | 0.9486 | 0.9529 | 0.9508 | 0.8796 | 0.8890 |
| 20 | 80 | 72,835 | 0.9373 | 0.9682 | 0.9529 | 0.9605 | 0.8748 | 0.8228 |
| 10 | 90 | 64,740 | 0.9458 | 0.9864 | 0.9529 | 0.9694 | 0.8818 | 0.6755 |
| 0 | 100 | 58,270 | 0.9529 | 1.0000 | 0.9529 | 0.9759 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8778 | 0.0000 | 0.0000 | 0.0000 | 0.8778 | 1.0000 |
| 90 | 10 | 299,940 | 0.8853 | 0.4641 | 0.9521 | 0.6240 | 0.8778 | 0.9940 |
| 80 | 20 | 291,350 | 0.8925 | 0.6603 | 0.9529 | 0.7801 | 0.8775 | 0.9868 |
| 70 | 30 | 194,230 | 0.9008 | 0.7706 | 0.9529 | 0.8521 | 0.8784 | 0.9775 |
| 60 | 40 | 145,675 | 0.9080 | 0.8389 | 0.9529 | 0.8923 | 0.8780 | 0.9655 |
| 50 | 50 | 116,540 | 0.9144 | 0.8847 | 0.9529 | 0.9175 | 0.8758 | 0.9490 |
| 40 | 60 | 97,115 | 0.9229 | 0.9213 | 0.9529 | 0.9368 | 0.8778 | 0.9255 |
| 30 | 70 | 83,240 | 0.9309 | 0.9486 | 0.9529 | 0.9508 | 0.8796 | 0.8890 |
| 20 | 80 | 72,835 | 0.9373 | 0.9682 | 0.9529 | 0.9605 | 0.8748 | 0.8228 |
| 10 | 90 | 64,740 | 0.9458 | 0.9864 | 0.9529 | 0.9694 | 0.8818 | 0.6755 |
| 0 | 100 | 58,270 | 0.9529 | 1.0000 | 0.9529 | 0.9759 | 0.0000 | 0.0000 |


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
0.15       0.6018   0.3048   0.5717   0.9759   0.8727   0.1846   <--
0.20       0.6018   0.3048   0.5717   0.9759   0.8727   0.1846  
0.25       0.6018   0.3048   0.5717   0.9759   0.8727   0.1846  
0.30       0.6018   0.3048   0.5717   0.9759   0.8727   0.1846  
0.35       0.6018   0.3048   0.5717   0.9759   0.8727   0.1846  
0.40       0.6018   0.3048   0.5717   0.9759   0.8727   0.1846  
0.45       0.6018   0.3048   0.5717   0.9759   0.8727   0.1846  
0.50       0.6018   0.3048   0.5717   0.9759   0.8727   0.1846  
0.55       0.6018   0.3048   0.5717   0.9759   0.8727   0.1846  
0.60       0.6018   0.3048   0.5717   0.9759   0.8727   0.1846  
0.65       0.6018   0.3048   0.5717   0.9759   0.8727   0.1846  
0.70       0.6018   0.3048   0.5717   0.9759   0.8727   0.1846  
0.75       0.6018   0.3048   0.5717   0.9759   0.8727   0.1846  
0.80       0.6018   0.3048   0.5717   0.9759   0.8727   0.1846  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6018, F1=0.3048, Normal Recall=0.5717, Normal Precision=0.9759, Attack Recall=0.8727, Attack Precision=0.1846

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
0.15       0.6319   0.4866   0.5718   0.9471   0.8723   0.3374   <--
0.20       0.6319   0.4866   0.5718   0.9471   0.8723   0.3374  
0.25       0.6319   0.4866   0.5718   0.9471   0.8723   0.3374  
0.30       0.6319   0.4866   0.5718   0.9471   0.8723   0.3374  
0.35       0.6319   0.4866   0.5718   0.9471   0.8723   0.3374  
0.40       0.6319   0.4866   0.5718   0.9471   0.8723   0.3374  
0.45       0.6319   0.4866   0.5718   0.9471   0.8723   0.3374  
0.50       0.6319   0.4866   0.5718   0.9471   0.8723   0.3374  
0.55       0.6319   0.4866   0.5718   0.9471   0.8723   0.3374  
0.60       0.6319   0.4866   0.5718   0.9471   0.8723   0.3374  
0.65       0.6319   0.4866   0.5718   0.9471   0.8723   0.3374  
0.70       0.6319   0.4866   0.5718   0.9471   0.8723   0.3374  
0.75       0.6319   0.4866   0.5718   0.9471   0.8723   0.3374  
0.80       0.6319   0.4866   0.5718   0.9471   0.8723   0.3374  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6319, F1=0.4866, Normal Recall=0.5718, Normal Precision=0.9471, Attack Recall=0.8723, Attack Precision=0.3374

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
0.15       0.6631   0.6084   0.5735   0.9129   0.8723   0.4671   <--
0.20       0.6631   0.6084   0.5735   0.9129   0.8723   0.4671  
0.25       0.6631   0.6084   0.5735   0.9129   0.8723   0.4671  
0.30       0.6631   0.6084   0.5735   0.9129   0.8723   0.4671  
0.35       0.6631   0.6084   0.5735   0.9129   0.8723   0.4671  
0.40       0.6631   0.6084   0.5735   0.9129   0.8723   0.4671  
0.45       0.6631   0.6084   0.5735   0.9129   0.8723   0.4671  
0.50       0.6631   0.6084   0.5735   0.9129   0.8723   0.4671  
0.55       0.6631   0.6084   0.5735   0.9129   0.8723   0.4671  
0.60       0.6631   0.6084   0.5735   0.9129   0.8723   0.4671  
0.65       0.6631   0.6084   0.5735   0.9129   0.8723   0.4671  
0.70       0.6631   0.6084   0.5735   0.9129   0.8723   0.4671  
0.75       0.6631   0.6084   0.5735   0.9129   0.8723   0.4671  
0.80       0.6631   0.6084   0.5735   0.9129   0.8723   0.4671  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6631, F1=0.6084, Normal Recall=0.5735, Normal Precision=0.9129, Attack Recall=0.8723, Attack Precision=0.4671

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
0.15       0.6925   0.6942   0.5727   0.8706   0.8723   0.5765   <--
0.20       0.6925   0.6942   0.5727   0.8706   0.8723   0.5765  
0.25       0.6925   0.6942   0.5727   0.8706   0.8723   0.5765  
0.30       0.6925   0.6942   0.5727   0.8706   0.8723   0.5765  
0.35       0.6925   0.6942   0.5727   0.8706   0.8723   0.5765  
0.40       0.6925   0.6942   0.5727   0.8706   0.8723   0.5765  
0.45       0.6925   0.6942   0.5727   0.8706   0.8723   0.5765  
0.50       0.6925   0.6942   0.5727   0.8706   0.8723   0.5765  
0.55       0.6925   0.6942   0.5727   0.8706   0.8723   0.5765  
0.60       0.6925   0.6942   0.5727   0.8706   0.8723   0.5765  
0.65       0.6925   0.6942   0.5727   0.8706   0.8723   0.5765  
0.70       0.6925   0.6942   0.5727   0.8706   0.8723   0.5765  
0.75       0.6925   0.6942   0.5727   0.8706   0.8723   0.5765  
0.80       0.6925   0.6942   0.5727   0.8706   0.8723   0.5765  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6925, F1=0.6942, Normal Recall=0.5727, Normal Precision=0.8706, Attack Recall=0.8723, Attack Precision=0.5765

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
0.15       0.7233   0.7592   0.5743   0.8181   0.8723   0.6720   <--
0.20       0.7233   0.7592   0.5743   0.8181   0.8723   0.6720  
0.25       0.7233   0.7592   0.5743   0.8181   0.8723   0.6720  
0.30       0.7233   0.7592   0.5743   0.8181   0.8723   0.6720  
0.35       0.7233   0.7592   0.5743   0.8181   0.8723   0.6720  
0.40       0.7233   0.7592   0.5743   0.8181   0.8723   0.6720  
0.45       0.7233   0.7592   0.5743   0.8181   0.8723   0.6720  
0.50       0.7233   0.7592   0.5743   0.8181   0.8723   0.6720  
0.55       0.7233   0.7592   0.5743   0.8181   0.8723   0.6720  
0.60       0.7233   0.7592   0.5743   0.8181   0.8723   0.6720  
0.65       0.7233   0.7592   0.5743   0.8181   0.8723   0.6720  
0.70       0.7233   0.7592   0.5743   0.8181   0.8723   0.6720  
0.75       0.7233   0.7592   0.5743   0.8181   0.8723   0.6720  
0.80       0.7233   0.7592   0.5743   0.8181   0.8723   0.6720  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7233, F1=0.7592, Normal Recall=0.5743, Normal Precision=0.8181, Attack Recall=0.8723, Attack Precision=0.6720

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
0.15       0.8855   0.6243   0.8782   0.9939   0.9514   0.4646   <--
0.20       0.8855   0.6243   0.8782   0.9939   0.9514   0.4646  
0.25       0.8855   0.6243   0.8782   0.9939   0.9514   0.4646  
0.30       0.8855   0.6243   0.8782   0.9939   0.9514   0.4646  
0.35       0.8855   0.6243   0.8782   0.9939   0.9514   0.4646  
0.40       0.8855   0.6243   0.8782   0.9939   0.9514   0.4646  
0.45       0.8855   0.6243   0.8782   0.9939   0.9514   0.4646  
0.50       0.8855   0.6243   0.8782   0.9939   0.9514   0.4646  
0.55       0.8855   0.6243   0.8782   0.9939   0.9514   0.4646  
0.60       0.8855   0.6243   0.8782   0.9939   0.9514   0.4646  
0.65       0.8855   0.6243   0.8782   0.9939   0.9514   0.4646  
0.70       0.8855   0.6243   0.8782   0.9939   0.9514   0.4646  
0.75       0.8855   0.6243   0.8782   0.9939   0.9514   0.4646  
0.80       0.8855   0.6243   0.8782   0.9939   0.9514   0.4646  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8855, F1=0.6243, Normal Recall=0.8782, Normal Precision=0.9939, Attack Recall=0.9514, Attack Precision=0.4646

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
0.15       0.8930   0.7806   0.8783   0.9865   0.9518   0.6616   <--
0.20       0.8930   0.7806   0.8783   0.9865   0.9518   0.6616  
0.25       0.8930   0.7806   0.8783   0.9865   0.9518   0.6616  
0.30       0.8930   0.7806   0.8783   0.9865   0.9518   0.6616  
0.35       0.8930   0.7806   0.8783   0.9865   0.9518   0.6616  
0.40       0.8930   0.7806   0.8783   0.9865   0.9518   0.6616  
0.45       0.8930   0.7806   0.8783   0.9865   0.9518   0.6616  
0.50       0.8930   0.7806   0.8783   0.9865   0.9518   0.6616  
0.55       0.8930   0.7806   0.8783   0.9865   0.9518   0.6616  
0.60       0.8930   0.7806   0.8783   0.9865   0.9518   0.6616  
0.65       0.8930   0.7806   0.8783   0.9865   0.9518   0.6616  
0.70       0.8930   0.7806   0.8783   0.9865   0.9518   0.6616  
0.75       0.8930   0.7806   0.8783   0.9865   0.9518   0.6616  
0.80       0.8930   0.7806   0.8783   0.9865   0.9518   0.6616  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8930, F1=0.7806, Normal Recall=0.8783, Normal Precision=0.9865, Attack Recall=0.9518, Attack Precision=0.6616

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
0.15       0.9003   0.8513   0.8782   0.9770   0.9518   0.7701   <--
0.20       0.9003   0.8513   0.8782   0.9770   0.9518   0.7701  
0.25       0.9003   0.8513   0.8782   0.9770   0.9518   0.7701  
0.30       0.9003   0.8513   0.8782   0.9770   0.9518   0.7701  
0.35       0.9003   0.8513   0.8782   0.9770   0.9518   0.7701  
0.40       0.9003   0.8513   0.8782   0.9770   0.9518   0.7701  
0.45       0.9003   0.8513   0.8782   0.9770   0.9518   0.7701  
0.50       0.9003   0.8513   0.8782   0.9770   0.9518   0.7701  
0.55       0.9003   0.8513   0.8782   0.9770   0.9518   0.7701  
0.60       0.9003   0.8513   0.8782   0.9770   0.9518   0.7701  
0.65       0.9003   0.8513   0.8782   0.9770   0.9518   0.7701  
0.70       0.9003   0.8513   0.8782   0.9770   0.9518   0.7701  
0.75       0.9003   0.8513   0.8782   0.9770   0.9518   0.7701  
0.80       0.9003   0.8513   0.8782   0.9770   0.9518   0.7701  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9003, F1=0.8513, Normal Recall=0.8782, Normal Precision=0.9770, Attack Recall=0.9518, Attack Precision=0.7701

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
0.15       0.9075   0.8917   0.8781   0.9647   0.9518   0.8388   <--
0.20       0.9075   0.8917   0.8781   0.9647   0.9518   0.8388  
0.25       0.9075   0.8917   0.8781   0.9647   0.9518   0.8388  
0.30       0.9075   0.8917   0.8781   0.9647   0.9518   0.8388  
0.35       0.9075   0.8917   0.8781   0.9647   0.9518   0.8388  
0.40       0.9075   0.8917   0.8781   0.9647   0.9518   0.8388  
0.45       0.9075   0.8917   0.8781   0.9647   0.9518   0.8388  
0.50       0.9075   0.8917   0.8781   0.9647   0.9518   0.8388  
0.55       0.9075   0.8917   0.8781   0.9647   0.9518   0.8388  
0.60       0.9075   0.8917   0.8781   0.9647   0.9518   0.8388  
0.65       0.9075   0.8917   0.8781   0.9647   0.9518   0.8388  
0.70       0.9075   0.8917   0.8781   0.9647   0.9518   0.8388  
0.75       0.9075   0.8917   0.8781   0.9647   0.9518   0.8388  
0.80       0.9075   0.8917   0.8781   0.9647   0.9518   0.8388  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9075, F1=0.8917, Normal Recall=0.8781, Normal Precision=0.9647, Attack Recall=0.9518, Attack Precision=0.8388

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
0.15       0.9141   0.9172   0.8764   0.9478   0.9518   0.8850   <--
0.20       0.9141   0.9172   0.8764   0.9478   0.9518   0.8850  
0.25       0.9141   0.9172   0.8764   0.9478   0.9518   0.8850  
0.30       0.9141   0.9172   0.8764   0.9478   0.9518   0.8850  
0.35       0.9141   0.9172   0.8764   0.9478   0.9518   0.8850  
0.40       0.9141   0.9172   0.8764   0.9478   0.9518   0.8850  
0.45       0.9141   0.9172   0.8764   0.9478   0.9518   0.8850  
0.50       0.9141   0.9172   0.8764   0.9478   0.9518   0.8850  
0.55       0.9141   0.9172   0.8764   0.9478   0.9518   0.8850  
0.60       0.9141   0.9172   0.8764   0.9478   0.9518   0.8850  
0.65       0.9141   0.9172   0.8764   0.9478   0.9518   0.8850  
0.70       0.9141   0.9172   0.8764   0.9478   0.9518   0.8850  
0.75       0.9141   0.9172   0.8764   0.9478   0.9518   0.8850  
0.80       0.9141   0.9172   0.8764   0.9478   0.9518   0.8850  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9141, F1=0.9172, Normal Recall=0.8764, Normal Precision=0.9478, Attack Recall=0.9518, Attack Precision=0.8850

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
0.15       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643   <--
0.20       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.25       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.30       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.35       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.40       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.45       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.50       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.55       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.60       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.65       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.70       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.75       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.80       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8853, F1=0.6243, Normal Recall=0.8778, Normal Precision=0.9941, Attack Recall=0.9528, Attack Precision=0.4643

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
0.15       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612   <--
0.20       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.25       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.30       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.35       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.40       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.45       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.50       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.55       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.60       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.65       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.70       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.75       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.80       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8929, F1=0.7807, Normal Recall=0.8779, Normal Precision=0.9868, Attack Recall=0.9529, Attack Precision=0.6612

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
0.15       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696   <--
0.20       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.25       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.30       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.35       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.40       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.45       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.50       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.55       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.60       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.65       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.70       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.75       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.80       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9003, F1=0.8515, Normal Recall=0.8778, Normal Precision=0.9775, Attack Recall=0.9529, Attack Precision=0.7696

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
0.15       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383   <--
0.20       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.25       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.30       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.35       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.40       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.45       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.50       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.55       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.60       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.65       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.70       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.75       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.80       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9076, F1=0.8919, Normal Recall=0.8774, Normal Precision=0.9655, Attack Recall=0.9529, Attack Precision=0.8383

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
0.15       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847   <--
0.20       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.25       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.30       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.35       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.40       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.45       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.50       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.55       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.60       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.65       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.70       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.75       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.80       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9143, F1=0.9175, Normal Recall=0.8758, Normal Precision=0.9490, Attack Recall=0.9529, Attack Precision=0.8847

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
0.15       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643   <--
0.20       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.25       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.30       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.35       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.40       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.45       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.50       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.55       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.60       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.65       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.70       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.75       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
0.80       0.8853   0.6243   0.8778   0.9941   0.9528   0.4643  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8853, F1=0.6243, Normal Recall=0.8778, Normal Precision=0.9941, Attack Recall=0.9528, Attack Precision=0.4643

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
0.15       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612   <--
0.20       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.25       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.30       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.35       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.40       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.45       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.50       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.55       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.60       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.65       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.70       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.75       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
0.80       0.8929   0.7807   0.8779   0.9868   0.9529   0.6612  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8929, F1=0.7807, Normal Recall=0.8779, Normal Precision=0.9868, Attack Recall=0.9529, Attack Precision=0.6612

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
0.15       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696   <--
0.20       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.25       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.30       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.35       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.40       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.45       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.50       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.55       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.60       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.65       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.70       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.75       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
0.80       0.9003   0.8515   0.8778   0.9775   0.9529   0.7696  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9003, F1=0.8515, Normal Recall=0.8778, Normal Precision=0.9775, Attack Recall=0.9529, Attack Precision=0.7696

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
0.15       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383   <--
0.20       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.25       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.30       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.35       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.40       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.45       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.50       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.55       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.60       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.65       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.70       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.75       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
0.80       0.9076   0.8919   0.8774   0.9655   0.9529   0.8383  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9076, F1=0.8919, Normal Recall=0.8774, Normal Precision=0.9655, Attack Recall=0.9529, Attack Precision=0.8383

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
0.15       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847   <--
0.20       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.25       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.30       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.35       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.40       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.45       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.50       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.55       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.60       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.65       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.70       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.75       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
0.80       0.9143   0.9175   0.8758   0.9490   0.9529   0.8847  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9143, F1=0.9175, Normal Recall=0.8758, Normal Precision=0.9490, Attack Recall=0.9529, Attack Precision=0.8847

```

