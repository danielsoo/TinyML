# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 6 models (same as compression_analysis) |
| **Config** | `config/federated.yaml` |
| **Generated** | 2026-02-23 22:57:53 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 2 |
| **Local epochs** | 2 |
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 6, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9286 | 0.9210 | 0.9135 | 0.9069 | 0.8998 | 0.8925 | 0.8859 | 0.8799 | 0.8717 | 0.8651 | 0.8581 |
| QAT+Prune only | 0.9846 | 0.9547 | 0.9252 | 0.8960 | 0.8660 | 0.8367 | 0.8070 | 0.7779 | 0.7486 | 0.7190 | 0.6898 |
| QAT+PTQ | 0.9847 | 0.9548 | 0.9253 | 0.8961 | 0.8663 | 0.8370 | 0.8074 | 0.7783 | 0.7490 | 0.7195 | 0.6903 |
| noQAT+PTQ | 0.0521 | 0.1469 | 0.2418 | 0.3362 | 0.4319 | 0.5260 | 0.6211 | 0.7160 | 0.8101 | 0.9056 | 1.0000 |
| Compressed (QAT) | 0.9791 | 0.9291 | 0.8795 | 0.8299 | 0.7800 | 0.7307 | 0.6814 | 0.6311 | 0.5820 | 0.5325 | 0.4828 |
| Compressed (PTQ) | 0.9847 | 0.9548 | 0.9253 | 0.8961 | 0.8663 | 0.8370 | 0.8074 | 0.7783 | 0.7490 | 0.7195 | 0.6903 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6851 | 0.7986 | 0.8468 | 0.8726 | 0.8887 | 0.9002 | 0.9091 | 0.9145 | 0.9197 | 0.9236 |
| QAT+Prune only | 0.0000 | 0.7531 | 0.7867 | 0.7991 | 0.8046 | 0.8086 | 0.8109 | 0.8130 | 0.8145 | 0.8154 | 0.8164 |
| QAT+PTQ | 0.0000 | 0.7538 | 0.7871 | 0.7994 | 0.8050 | 0.8090 | 0.8113 | 0.8134 | 0.8148 | 0.8158 | 0.8168 |
| noQAT+PTQ | 0.0000 | 0.1899 | 0.3454 | 0.4748 | 0.5847 | 0.6784 | 0.7600 | 0.8314 | 0.8939 | 0.9502 | 1.0000 |
| Compressed (QAT) | 0.0000 | 0.5775 | 0.6157 | 0.6300 | 0.6372 | 0.6419 | 0.6452 | 0.6470 | 0.6489 | 0.6502 | 0.6512 |
| Compressed (PTQ) | 0.0000 | 0.7538 | 0.7871 | 0.7994 | 0.8050 | 0.8090 | 0.8113 | 0.8134 | 0.8148 | 0.8158 | 0.8168 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9286 | 0.9279 | 0.9273 | 0.9278 | 0.9276 | 0.9270 | 0.9276 | 0.9309 | 0.9263 | 0.9282 | 0.0000 |
| QAT+Prune only | 0.9846 | 0.9840 | 0.9840 | 0.9843 | 0.9835 | 0.9837 | 0.9829 | 0.9835 | 0.9840 | 0.9821 | 0.0000 |
| QAT+PTQ | 0.9847 | 0.9840 | 0.9841 | 0.9843 | 0.9836 | 0.9837 | 0.9831 | 0.9836 | 0.9840 | 0.9822 | 0.0000 |
| noQAT+PTQ | 0.0521 | 0.0521 | 0.0522 | 0.0518 | 0.0531 | 0.0519 | 0.0527 | 0.0535 | 0.0505 | 0.0562 | 0.0000 |
| Compressed (QAT) | 0.9791 | 0.9786 | 0.9786 | 0.9786 | 0.9782 | 0.9785 | 0.9792 | 0.9772 | 0.9787 | 0.9791 | 0.0000 |
| Compressed (PTQ) | 0.9847 | 0.9840 | 0.9841 | 0.9843 | 0.9836 | 0.9837 | 0.9831 | 0.9836 | 0.9840 | 0.9822 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9286 | 0.0000 | 0.0000 | 0.0000 | 0.9286 | 1.0000 |
| 90 | 10 | 299,940 | 0.9210 | 0.5698 | 0.8590 | 0.6851 | 0.9279 | 0.9834 |
| 80 | 20 | 291,350 | 0.9135 | 0.7469 | 0.8581 | 0.7986 | 0.9273 | 0.9631 |
| 70 | 30 | 194,230 | 0.9069 | 0.8358 | 0.8581 | 0.8468 | 0.9278 | 0.9385 |
| 60 | 40 | 145,675 | 0.8998 | 0.8876 | 0.8581 | 0.8726 | 0.9276 | 0.9074 |
| 50 | 50 | 116,540 | 0.8925 | 0.9216 | 0.8581 | 0.8887 | 0.9270 | 0.8672 |
| 40 | 60 | 97,115 | 0.8859 | 0.9468 | 0.8581 | 0.9002 | 0.9276 | 0.8133 |
| 30 | 70 | 83,240 | 0.8799 | 0.9666 | 0.8581 | 0.9091 | 0.9309 | 0.7376 |
| 20 | 80 | 72,835 | 0.8717 | 0.9790 | 0.8581 | 0.9145 | 0.9263 | 0.6200 |
| 10 | 90 | 64,740 | 0.8651 | 0.9908 | 0.8581 | 0.9197 | 0.9282 | 0.4208 |
| 0 | 100 | 58,270 | 0.8581 | 1.0000 | 0.8581 | 0.9236 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9846 | 0.0000 | 0.0000 | 0.0000 | 0.9846 | 1.0000 |
| 90 | 10 | 299,940 | 0.9547 | 0.8275 | 0.6909 | 0.7531 | 0.9840 | 0.9663 |
| 80 | 20 | 291,350 | 0.9252 | 0.9153 | 0.6898 | 0.7867 | 0.9840 | 0.9269 |
| 70 | 30 | 194,230 | 0.8960 | 0.9496 | 0.6897 | 0.7991 | 0.9843 | 0.8810 |
| 60 | 40 | 145,675 | 0.8660 | 0.9653 | 0.6898 | 0.8046 | 0.9835 | 0.8262 |
| 50 | 50 | 116,540 | 0.8367 | 0.9770 | 0.6898 | 0.8086 | 0.9837 | 0.7602 |
| 40 | 60 | 97,115 | 0.8070 | 0.9838 | 0.6897 | 0.8109 | 0.9829 | 0.6787 |
| 30 | 70 | 83,240 | 0.7779 | 0.9899 | 0.6897 | 0.8130 | 0.9835 | 0.5760 |
| 20 | 80 | 72,835 | 0.7486 | 0.9942 | 0.6898 | 0.8145 | 0.9840 | 0.4423 |
| 10 | 90 | 64,740 | 0.7190 | 0.9971 | 0.6898 | 0.8154 | 0.9821 | 0.2602 |
| 0 | 100 | 58,270 | 0.6898 | 1.0000 | 0.6898 | 0.8164 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9847 | 0.0000 | 0.0000 | 0.0000 | 0.9847 | 1.0000 |
| 90 | 10 | 299,940 | 0.9548 | 0.8279 | 0.6919 | 0.7538 | 0.9840 | 0.9664 |
| 80 | 20 | 291,350 | 0.9253 | 0.9154 | 0.6903 | 0.7871 | 0.9841 | 0.9271 |
| 70 | 30 | 194,230 | 0.8961 | 0.9496 | 0.6903 | 0.7994 | 0.9843 | 0.8812 |
| 60 | 40 | 145,675 | 0.8663 | 0.9656 | 0.6903 | 0.8050 | 0.9836 | 0.8265 |
| 50 | 50 | 116,540 | 0.8370 | 0.9769 | 0.6903 | 0.8090 | 0.9837 | 0.7605 |
| 40 | 60 | 97,115 | 0.8074 | 0.9839 | 0.6903 | 0.8113 | 0.9831 | 0.6791 |
| 30 | 70 | 83,240 | 0.7783 | 0.9899 | 0.6903 | 0.8134 | 0.9836 | 0.5764 |
| 20 | 80 | 72,835 | 0.7490 | 0.9942 | 0.6903 | 0.8148 | 0.9840 | 0.4427 |
| 10 | 90 | 64,740 | 0.7195 | 0.9971 | 0.6903 | 0.8158 | 0.9822 | 0.2606 |
| 0 | 100 | 58,270 | 0.6903 | 1.0000 | 0.6903 | 0.8168 | 0.0000 | 0.0000 |

### noQAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0521 | 0.0000 | 0.0000 | 0.0000 | 0.0521 | 1.0000 |
| 90 | 10 | 299,940 | 0.1469 | 0.1049 | 1.0000 | 0.1899 | 0.0521 | 1.0000 |
| 80 | 20 | 291,350 | 0.2418 | 0.2087 | 1.0000 | 0.3454 | 0.0522 | 1.0000 |
| 70 | 30 | 194,230 | 0.3362 | 0.3113 | 1.0000 | 0.4748 | 0.0518 | 1.0000 |
| 60 | 40 | 145,675 | 0.4319 | 0.4132 | 1.0000 | 0.5847 | 0.0531 | 1.0000 |
| 50 | 50 | 116,540 | 0.5260 | 0.5133 | 1.0000 | 0.6784 | 0.0519 | 1.0000 |
| 40 | 60 | 97,115 | 0.6211 | 0.6129 | 1.0000 | 0.7600 | 0.0527 | 1.0000 |
| 30 | 70 | 83,240 | 0.7160 | 0.7114 | 1.0000 | 0.8314 | 0.0535 | 1.0000 |
| 20 | 80 | 72,835 | 0.8101 | 0.8082 | 1.0000 | 0.8939 | 0.0505 | 1.0000 |
| 10 | 90 | 64,740 | 0.9056 | 0.9051 | 1.0000 | 0.9502 | 0.0562 | 1.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### Compressed (QAT)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9791 | 0.0000 | 0.0000 | 0.0000 | 0.9791 | 1.0000 |
| 90 | 10 | 299,940 | 0.9291 | 0.7151 | 0.4843 | 0.5775 | 0.9786 | 0.9447 |
| 80 | 20 | 291,350 | 0.8795 | 0.8495 | 0.4828 | 0.6157 | 0.9786 | 0.8833 |
| 70 | 30 | 194,230 | 0.8299 | 0.9065 | 0.4828 | 0.6300 | 0.9786 | 0.8153 |
| 60 | 40 | 145,675 | 0.7800 | 0.9365 | 0.4828 | 0.6372 | 0.9782 | 0.7394 |
| 50 | 50 | 116,540 | 0.7307 | 0.9574 | 0.4828 | 0.6419 | 0.9785 | 0.6542 |
| 40 | 60 | 97,115 | 0.6814 | 0.9721 | 0.4828 | 0.6452 | 0.9792 | 0.5580 |
| 30 | 70 | 83,240 | 0.6311 | 0.9801 | 0.4828 | 0.6470 | 0.9772 | 0.4474 |
| 20 | 80 | 72,835 | 0.5820 | 0.9891 | 0.4828 | 0.6489 | 0.9787 | 0.3211 |
| 10 | 90 | 64,740 | 0.5325 | 0.9952 | 0.4828 | 0.6502 | 0.9791 | 0.1738 |
| 0 | 100 | 58,270 | 0.4828 | 1.0000 | 0.4828 | 0.6512 | 0.0000 | 0.0000 |

### Compressed (PTQ)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9847 | 0.0000 | 0.0000 | 0.0000 | 0.9847 | 1.0000 |
| 90 | 10 | 299,940 | 0.9548 | 0.8279 | 0.6919 | 0.7538 | 0.9840 | 0.9664 |
| 80 | 20 | 291,350 | 0.9253 | 0.9154 | 0.6903 | 0.7871 | 0.9841 | 0.9271 |
| 70 | 30 | 194,230 | 0.8961 | 0.9496 | 0.6903 | 0.7994 | 0.9843 | 0.8812 |
| 60 | 40 | 145,675 | 0.8663 | 0.9656 | 0.6903 | 0.8050 | 0.9836 | 0.8265 |
| 50 | 50 | 116,540 | 0.8370 | 0.9769 | 0.6903 | 0.8090 | 0.9837 | 0.7605 |
| 40 | 60 | 97,115 | 0.8074 | 0.9839 | 0.6903 | 0.8113 | 0.9831 | 0.6791 |
| 30 | 70 | 83,240 | 0.7783 | 0.9899 | 0.6903 | 0.8134 | 0.9836 | 0.5764 |
| 20 | 80 | 72,835 | 0.7490 | 0.9942 | 0.6903 | 0.8148 | 0.9840 | 0.4427 |
| 10 | 90 | 64,740 | 0.7195 | 0.9971 | 0.6903 | 0.8158 | 0.9822 | 0.2606 |
| 0 | 100 | 58,270 | 0.6903 | 1.0000 | 0.6903 | 0.8168 | 0.0000 | 0.0000 |


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
0.15       0.9211   0.6853   0.9279   0.9834   0.8593   0.5699   <--
0.20       0.9211   0.6853   0.9279   0.9834   0.8593   0.5699  
0.25       0.9211   0.6853   0.9279   0.9834   0.8593   0.5699  
0.30       0.9211   0.6853   0.9279   0.9834   0.8593   0.5699  
0.35       0.9211   0.6853   0.9279   0.9834   0.8593   0.5699  
0.40       0.9211   0.6853   0.9279   0.9834   0.8593   0.5699  
0.45       0.9211   0.6853   0.9279   0.9834   0.8593   0.5699  
0.50       0.9211   0.6853   0.9279   0.9834   0.8593   0.5699  
0.55       0.9211   0.6853   0.9279   0.9834   0.8593   0.5699  
0.60       0.9211   0.6853   0.9279   0.9834   0.8593   0.5699  
0.65       0.9211   0.6853   0.9279   0.9834   0.8593   0.5699  
0.70       0.9211   0.6853   0.9279   0.9834   0.8593   0.5699  
0.75       0.9211   0.6853   0.9279   0.9834   0.8593   0.5699  
0.80       0.9211   0.6853   0.9279   0.9834   0.8593   0.5699  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9211, F1=0.6853, Normal Recall=0.9279, Normal Precision=0.9834, Attack Recall=0.8593, Attack Precision=0.5699

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
0.15       0.9141   0.7998   0.9281   0.9632   0.8581   0.7489   <--
0.20       0.9141   0.7998   0.9281   0.9632   0.8581   0.7489  
0.25       0.9141   0.7998   0.9281   0.9632   0.8581   0.7489  
0.30       0.9141   0.7998   0.9281   0.9632   0.8581   0.7489  
0.35       0.9141   0.7998   0.9281   0.9632   0.8581   0.7489  
0.40       0.9141   0.7998   0.9281   0.9632   0.8581   0.7489  
0.45       0.9141   0.7998   0.9281   0.9632   0.8581   0.7489  
0.50       0.9141   0.7998   0.9281   0.9632   0.8581   0.7489  
0.55       0.9141   0.7998   0.9281   0.9632   0.8581   0.7489  
0.60       0.9141   0.7998   0.9281   0.9632   0.8581   0.7489  
0.65       0.9141   0.7998   0.9281   0.9632   0.8581   0.7489  
0.70       0.9141   0.7998   0.9281   0.9632   0.8581   0.7489  
0.75       0.9141   0.7998   0.9281   0.9632   0.8581   0.7489  
0.80       0.9141   0.7998   0.9281   0.9632   0.8581   0.7489  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9141, F1=0.7998, Normal Recall=0.9281, Normal Precision=0.9632, Attack Recall=0.8581, Attack Precision=0.7489

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
0.15       0.9077   0.8480   0.9290   0.9385   0.8581   0.8381   <--
0.20       0.9077   0.8480   0.9290   0.9385   0.8581   0.8381  
0.25       0.9077   0.8480   0.9290   0.9385   0.8581   0.8381  
0.30       0.9077   0.8480   0.9290   0.9385   0.8581   0.8381  
0.35       0.9077   0.8480   0.9290   0.9385   0.8581   0.8381  
0.40       0.9077   0.8480   0.9290   0.9385   0.8581   0.8381  
0.45       0.9077   0.8480   0.9290   0.9385   0.8581   0.8381  
0.50       0.9077   0.8480   0.9290   0.9385   0.8581   0.8381  
0.55       0.9077   0.8480   0.9290   0.9385   0.8581   0.8381  
0.60       0.9077   0.8480   0.9290   0.9385   0.8581   0.8381  
0.65       0.9077   0.8480   0.9290   0.9385   0.8581   0.8381  
0.70       0.9077   0.8480   0.9290   0.9385   0.8581   0.8381  
0.75       0.9077   0.8480   0.9290   0.9385   0.8581   0.8381  
0.80       0.9077   0.8480   0.9290   0.9385   0.8581   0.8381  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9077, F1=0.8480, Normal Recall=0.9290, Normal Precision=0.9385, Attack Recall=0.8581, Attack Precision=0.8381

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
0.15       0.9004   0.8732   0.9286   0.9075   0.8581   0.8890   <--
0.20       0.9004   0.8732   0.9286   0.9075   0.8581   0.8890  
0.25       0.9004   0.8732   0.9286   0.9075   0.8581   0.8890  
0.30       0.9004   0.8732   0.9286   0.9075   0.8581   0.8890  
0.35       0.9004   0.8732   0.9286   0.9075   0.8581   0.8890  
0.40       0.9004   0.8732   0.9286   0.9075   0.8581   0.8890  
0.45       0.9004   0.8732   0.9286   0.9075   0.8581   0.8890  
0.50       0.9004   0.8732   0.9286   0.9075   0.8581   0.8890  
0.55       0.9004   0.8732   0.9286   0.9075   0.8581   0.8890  
0.60       0.9004   0.8732   0.9286   0.9075   0.8581   0.8890  
0.65       0.9004   0.8732   0.9286   0.9075   0.8581   0.8890  
0.70       0.9004   0.8732   0.9286   0.9075   0.8581   0.8890  
0.75       0.9004   0.8732   0.9286   0.9075   0.8581   0.8890  
0.80       0.9004   0.8732   0.9286   0.9075   0.8581   0.8890  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9004, F1=0.8732, Normal Recall=0.9286, Normal Precision=0.9075, Attack Recall=0.8581, Attack Precision=0.8890

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
0.15       0.8931   0.8892   0.9281   0.8673   0.8581   0.9226   <--
0.20       0.8931   0.8892   0.9281   0.8673   0.8581   0.9226  
0.25       0.8931   0.8892   0.9281   0.8673   0.8581   0.9226  
0.30       0.8931   0.8892   0.9281   0.8673   0.8581   0.9226  
0.35       0.8931   0.8892   0.9281   0.8673   0.8581   0.9226  
0.40       0.8931   0.8892   0.9281   0.8673   0.8581   0.9226  
0.45       0.8931   0.8892   0.9281   0.8673   0.8581   0.9226  
0.50       0.8931   0.8892   0.9281   0.8673   0.8581   0.9226  
0.55       0.8931   0.8892   0.9281   0.8673   0.8581   0.9226  
0.60       0.8931   0.8892   0.9281   0.8673   0.8581   0.9226  
0.65       0.8931   0.8892   0.9281   0.8673   0.8581   0.9226  
0.70       0.8931   0.8892   0.9281   0.8673   0.8581   0.9226  
0.75       0.8931   0.8892   0.9281   0.8673   0.8581   0.9226  
0.80       0.8931   0.8892   0.9281   0.8673   0.8581   0.9226  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8931, F1=0.8892, Normal Recall=0.9281, Normal Precision=0.8673, Attack Recall=0.8581, Attack Precision=0.9226

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
0.15       0.9547   0.7534   0.9840   0.9663   0.6914   0.8277   <--
0.20       0.9547   0.7534   0.9840   0.9663   0.6914   0.8277  
0.25       0.9547   0.7534   0.9840   0.9663   0.6914   0.8277  
0.30       0.9547   0.7534   0.9840   0.9663   0.6914   0.8277  
0.35       0.9547   0.7534   0.9840   0.9663   0.6914   0.8277  
0.40       0.9547   0.7534   0.9840   0.9663   0.6914   0.8277  
0.45       0.9547   0.7534   0.9840   0.9663   0.6914   0.8277  
0.50       0.9547   0.7534   0.9840   0.9663   0.6914   0.8277  
0.55       0.9547   0.7534   0.9840   0.9663   0.6914   0.8277  
0.60       0.9547   0.7534   0.9840   0.9663   0.6914   0.8277  
0.65       0.9547   0.7534   0.9840   0.9663   0.6914   0.8277  
0.70       0.9547   0.7534   0.9840   0.9663   0.6914   0.8277  
0.75       0.9547   0.7534   0.9840   0.9663   0.6914   0.8277  
0.80       0.9547   0.7534   0.9840   0.9663   0.6914   0.8277  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9547, F1=0.7534, Normal Recall=0.9840, Normal Precision=0.9663, Attack Recall=0.6914, Attack Precision=0.8277

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
0.15       0.9251   0.7865   0.9840   0.9269   0.6898   0.9149   <--
0.20       0.9251   0.7865   0.9840   0.9269   0.6898   0.9149  
0.25       0.9251   0.7865   0.9840   0.9269   0.6898   0.9149  
0.30       0.9251   0.7865   0.9840   0.9269   0.6898   0.9149  
0.35       0.9251   0.7865   0.9840   0.9269   0.6898   0.9149  
0.40       0.9251   0.7865   0.9840   0.9269   0.6898   0.9149  
0.45       0.9251   0.7865   0.9840   0.9269   0.6898   0.9149  
0.50       0.9251   0.7865   0.9840   0.9269   0.6898   0.9149  
0.55       0.9251   0.7865   0.9840   0.9269   0.6898   0.9149  
0.60       0.9251   0.7865   0.9840   0.9269   0.6898   0.9149  
0.65       0.9251   0.7865   0.9840   0.9269   0.6898   0.9149  
0.70       0.9251   0.7865   0.9840   0.9269   0.6898   0.9149  
0.75       0.9251   0.7865   0.9840   0.9269   0.6898   0.9149  
0.80       0.9251   0.7865   0.9840   0.9269   0.6898   0.9149  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9251, F1=0.7865, Normal Recall=0.9840, Normal Precision=0.9269, Attack Recall=0.6898, Attack Precision=0.9149

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
0.15       0.8959   0.7991   0.9843   0.8810   0.6897   0.9496   <--
0.20       0.8959   0.7991   0.9843   0.8810   0.6897   0.9496  
0.25       0.8959   0.7991   0.9843   0.8810   0.6897   0.9496  
0.30       0.8959   0.7991   0.9843   0.8810   0.6897   0.9496  
0.35       0.8959   0.7991   0.9843   0.8810   0.6897   0.9496  
0.40       0.8959   0.7991   0.9843   0.8810   0.6897   0.9496  
0.45       0.8959   0.7991   0.9843   0.8810   0.6897   0.9496  
0.50       0.8959   0.7991   0.9843   0.8810   0.6897   0.9496  
0.55       0.8959   0.7991   0.9843   0.8810   0.6897   0.9496  
0.60       0.8959   0.7991   0.9843   0.8810   0.6897   0.9496  
0.65       0.8959   0.7991   0.9843   0.8810   0.6897   0.9496  
0.70       0.8959   0.7991   0.9843   0.8810   0.6897   0.9496  
0.75       0.8959   0.7991   0.9843   0.8810   0.6897   0.9496  
0.80       0.8959   0.7991   0.9843   0.8810   0.6897   0.9496  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8959, F1=0.7991, Normal Recall=0.9843, Normal Precision=0.8810, Attack Recall=0.6897, Attack Precision=0.9496

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
0.15       0.8666   0.8053   0.9845   0.8264   0.6898   0.9675   <--
0.20       0.8666   0.8053   0.9845   0.8264   0.6898   0.9675  
0.25       0.8666   0.8053   0.9845   0.8264   0.6898   0.9675  
0.30       0.8666   0.8053   0.9845   0.8264   0.6898   0.9675  
0.35       0.8666   0.8053   0.9845   0.8264   0.6898   0.9675  
0.40       0.8666   0.8053   0.9845   0.8264   0.6898   0.9675  
0.45       0.8666   0.8053   0.9845   0.8264   0.6898   0.9675  
0.50       0.8666   0.8053   0.9845   0.8264   0.6898   0.9675  
0.55       0.8666   0.8053   0.9845   0.8264   0.6898   0.9675  
0.60       0.8666   0.8053   0.9845   0.8264   0.6898   0.9675  
0.65       0.8666   0.8053   0.9845   0.8264   0.6898   0.9675  
0.70       0.8666   0.8053   0.9845   0.8264   0.6898   0.9675  
0.75       0.8666   0.8053   0.9845   0.8264   0.6898   0.9675  
0.80       0.8666   0.8053   0.9845   0.8264   0.6898   0.9675  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8666, F1=0.8053, Normal Recall=0.9845, Normal Precision=0.8264, Attack Recall=0.6898, Attack Precision=0.9675

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
0.15       0.8372   0.8090   0.9846   0.7604   0.6898   0.9781   <--
0.20       0.8372   0.8090   0.9846   0.7604   0.6898   0.9781  
0.25       0.8372   0.8090   0.9846   0.7604   0.6898   0.9781  
0.30       0.8372   0.8090   0.9846   0.7604   0.6898   0.9781  
0.35       0.8372   0.8090   0.9846   0.7604   0.6898   0.9781  
0.40       0.8372   0.8090   0.9846   0.7604   0.6898   0.9781  
0.45       0.8372   0.8090   0.9846   0.7604   0.6898   0.9781  
0.50       0.8372   0.8090   0.9846   0.7604   0.6898   0.9781  
0.55       0.8372   0.8090   0.9846   0.7604   0.6898   0.9781  
0.60       0.8372   0.8090   0.9846   0.7604   0.6898   0.9781  
0.65       0.8372   0.8090   0.9846   0.7604   0.6898   0.9781  
0.70       0.8372   0.8090   0.9846   0.7604   0.6898   0.9781  
0.75       0.8372   0.8090   0.9846   0.7604   0.6898   0.9781  
0.80       0.8372   0.8090   0.9846   0.7604   0.6898   0.9781  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8372, F1=0.8090, Normal Recall=0.9846, Normal Precision=0.7604, Attack Recall=0.6898, Attack Precision=0.9781

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
0.15       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278   <--
0.20       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.25       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.30       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.35       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.40       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.45       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.50       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.55       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.60       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.65       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.70       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.75       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.80       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9548, F1=0.7534, Normal Recall=0.9840, Normal Precision=0.9663, Attack Recall=0.6913, Attack Precision=0.8278

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
0.15       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152   <--
0.20       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.25       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.30       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.35       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.40       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.45       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.50       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.55       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.60       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.65       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.70       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.75       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.80       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9253, F1=0.7870, Normal Recall=0.9840, Normal Precision=0.9270, Attack Recall=0.6903, Attack Precision=0.9152

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
0.15       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499   <--
0.20       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.25       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.30       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.35       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.40       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.45       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.50       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.55       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.60       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.65       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.70       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.75       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.80       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8962, F1=0.7995, Normal Recall=0.9844, Normal Precision=0.8812, Attack Recall=0.6903, Attack Precision=0.9499

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
0.15       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677   <--
0.20       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.25       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.30       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.35       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.40       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.45       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.50       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.55       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.60       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.65       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.70       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.75       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.80       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8669, F1=0.8058, Normal Recall=0.9846, Normal Precision=0.8266, Attack Recall=0.6903, Attack Precision=0.9677

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
0.15       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783   <--
0.20       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.25       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.30       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.35       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.40       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.45       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.50       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.55       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.60       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.65       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.70       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.75       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.80       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8375, F1=0.8094, Normal Recall=0.9847, Normal Precision=0.7607, Attack Recall=0.6903, Attack Precision=0.9783

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
0.15       0.1129   0.1840   0.0143   1.0000   1.0000   0.1013  
0.20       0.1187   0.1850   0.0208   1.0000   1.0000   0.1019  
0.25       0.1314   0.1872   0.0348   1.0000   1.0000   0.1032  
0.30       0.1469   0.1899   0.0521   1.0000   1.0000   0.1049  
0.35       0.1772   0.1955   0.0858   1.0000   1.0000   0.1084  
0.40       0.2735   0.2159   0.1928   1.0000   1.0000   0.1210  
0.45       0.5748   0.3199   0.5276   1.0000   0.9998   0.1904  
0.50       0.7845   0.4806   0.7608   0.9996   0.9974   0.3166  
0.55       0.9485   0.7618   0.9624   0.9800   0.8235   0.7087   <--
0.60       0.9503   0.7017   0.9911   0.9554   0.5840   0.8789  
0.65       0.9471   0.6485   0.9982   0.9461   0.4878   0.9671  
0.70       0.9475   0.6466   0.9993   0.9454   0.4805   0.9879  
0.75       0.9465   0.6357   0.9997   0.9441   0.4671   0.9945  
0.80       0.9413   0.5858   0.9998   0.9389   0.4148   0.9967  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9485, F1=0.7618, Normal Recall=0.9624, Normal Precision=0.9800, Attack Recall=0.8235, Attack Precision=0.7087

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
0.15       0.2113   0.3365   0.0142   1.0000   1.0000   0.2023  
0.20       0.2166   0.3380   0.0207   1.0000   1.0000   0.2034  
0.25       0.2278   0.3412   0.0347   1.0000   1.0000   0.2057  
0.30       0.2415   0.3453   0.0518   1.0000   1.0000   0.2087  
0.35       0.2684   0.3535   0.0855   0.9999   1.0000   0.2147  
0.40       0.3541   0.3824   0.1927   1.0000   1.0000   0.2364  
0.45       0.6221   0.5141   0.5277   0.9999   0.9997   0.3460  
0.50       0.8083   0.6754   0.7610   0.9991   0.9972   0.5106  
0.55       0.9345   0.8342   0.9623   0.9561   0.8233   0.8453   <--
0.60       0.9099   0.7220   0.9912   0.9052   0.5849   0.9431  
0.65       0.8961   0.6524   0.9982   0.8863   0.4876   0.9852  
0.70       0.8955   0.6478   0.9993   0.8850   0.4803   0.9945  
0.75       0.8931   0.6359   0.9997   0.8823   0.4667   0.9975  
0.80       0.8826   0.5850   0.9998   0.8721   0.4137   0.9985  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9345, F1=0.8342, Normal Recall=0.9623, Normal Precision=0.9561, Attack Recall=0.8233, Attack Precision=0.8453

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
0.15       0.3100   0.4651   0.0143   1.0000   1.0000   0.3030  
0.20       0.3149   0.4669   0.0213   1.0000   1.0000   0.3045  
0.25       0.3248   0.4705   0.0354   1.0000   1.0000   0.3076  
0.30       0.3369   0.4750   0.0528   1.0000   1.0000   0.3115  
0.35       0.3605   0.4841   0.0865   0.9999   1.0000   0.3193  
0.40       0.4358   0.5154   0.1940   0.9999   1.0000   0.3471  
0.45       0.6702   0.6452   0.5290   0.9998   0.9997   0.4763  
0.50       0.8314   0.7802   0.7604   0.9984   0.9972   0.6408  
0.55       0.9203   0.8611   0.9619   0.9270   0.8233   0.9025   <--
0.60       0.8691   0.7283   0.9909   0.8478   0.5849   0.9648  
0.65       0.8450   0.6537   0.9981   0.8197   0.4876   0.9911  
0.70       0.8436   0.6483   0.9993   0.8178   0.4803   0.9967  
0.75       0.8398   0.6361   0.9997   0.8139   0.4667   0.9986  
0.80       0.8240   0.5851   0.9999   0.7992   0.4137   0.9993  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9203, F1=0.8611, Normal Recall=0.9619, Normal Precision=0.9270, Attack Recall=0.8233, Attack Precision=0.9025

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
0.15       0.4083   0.5748   0.0138   1.0000   1.0000   0.4033  
0.20       0.4125   0.5766   0.0208   1.0000   1.0000   0.4051  
0.25       0.4211   0.5802   0.0351   1.0000   1.0000   0.4086  
0.30       0.4314   0.5845   0.0523   1.0000   1.0000   0.4129  
0.35       0.4518   0.5934   0.0863   0.9999   1.0000   0.4218  
0.40       0.5165   0.6233   0.1942   0.9999   1.0000   0.4527  
0.45       0.7172   0.7388   0.5288   0.9996   0.9997   0.5858  
0.50       0.8550   0.8462   0.7603   0.9975   0.9972   0.7350  
0.55       0.9065   0.8757   0.9619   0.8909   0.8233   0.9351   <--
0.60       0.8287   0.7320   0.9912   0.7817   0.5849   0.9780  
0.65       0.7940   0.6544   0.9982   0.7451   0.4876   0.9946  
0.70       0.7918   0.6485   0.9994   0.7426   0.4803   0.9981  
0.75       0.7865   0.6362   0.9997   0.7376   0.4667   0.9990  
0.80       0.7654   0.5852   0.9998   0.7189   0.4137   0.9994  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9065, F1=0.8757, Normal Recall=0.9619, Normal Precision=0.8909, Attack Recall=0.8233, Attack Precision=0.9351

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
0.15       0.5069   0.6697   0.0137   1.0000   1.0000   0.5035  
0.20       0.5103   0.6713   0.0205   1.0000   1.0000   0.5052  
0.25       0.5173   0.6745   0.0346   1.0000   1.0000   0.5088  
0.30       0.5259   0.6784   0.0517   1.0000   1.0000   0.5133  
0.35       0.5430   0.6863   0.0860   0.9998   1.0000   0.5225  
0.40       0.5965   0.7125   0.1930   0.9998   1.0000   0.5534  
0.45       0.7643   0.8092   0.5289   0.9994   0.9997   0.6797  
0.50       0.8788   0.8917   0.7605   0.9963   0.9972   0.8063   <--
0.55       0.8926   0.8846   0.9618   0.8448   0.8233   0.9556  
0.60       0.7880   0.7340   0.9911   0.7048   0.5849   0.9851  
0.65       0.7429   0.6548   0.9982   0.6608   0.4876   0.9964  
0.70       0.7399   0.6487   0.9994   0.6579   0.4803   0.9987  
0.75       0.7332   0.6362   0.9997   0.6521   0.4667   0.9993  
0.80       0.7068   0.5852   0.9998   0.6304   0.4137   0.9996  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.8788, F1=0.8917, Normal Recall=0.7605, Normal Precision=0.9963, Attack Recall=0.9972, Attack Precision=0.8063

```


## Threshold Tuning (saved_model_pruned_qat)

Model: `models/tflite/saved_model_pruned_qat.tflite`

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
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 90% : Attack 10%  (n=299,940, N=269,946, A=29,994)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9286   0.5731   0.9786   0.9442   0.4791   0.7130   <--
0.20       0.9286   0.5731   0.9786   0.9442   0.4791   0.7130  
0.25       0.9286   0.5731   0.9786   0.9442   0.4791   0.7130  
0.30       0.9286   0.5731   0.9786   0.9442   0.4791   0.7130  
0.35       0.9286   0.5731   0.9786   0.9442   0.4791   0.7130  
0.40       0.9286   0.5731   0.9786   0.9442   0.4791   0.7130  
0.45       0.9286   0.5731   0.9786   0.9442   0.4791   0.7130  
0.50       0.9286   0.5731   0.9786   0.9442   0.4791   0.7130  
0.55       0.9286   0.5730   0.9786   0.9441   0.4790   0.7130  
0.60       0.9286   0.5730   0.9786   0.9441   0.4790   0.7130  
0.65       0.9286   0.5730   0.9786   0.9441   0.4790   0.7130  
0.70       0.9286   0.5730   0.9786   0.9441   0.4790   0.7130  
0.75       0.9286   0.5730   0.9786   0.9441   0.4790   0.7130  
0.80       0.9286   0.5730   0.9786   0.9441   0.4790   0.7130  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9286, F1=0.5731, Normal Recall=0.9786, Normal Precision=0.9442, Attack Recall=0.4791, Attack Precision=0.7130

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
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 80% : Attack 20%  (n=291,350, N=233,080, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.8794   0.6156   0.9785   0.8833   0.4828   0.8491   <--
0.20       0.8794   0.6156   0.9785   0.8833   0.4828   0.8491  
0.25       0.8794   0.6156   0.9785   0.8833   0.4828   0.8491  
0.30       0.8794   0.6156   0.9785   0.8833   0.4828   0.8491  
0.35       0.8794   0.6156   0.9785   0.8833   0.4828   0.8491  
0.40       0.8794   0.6156   0.9785   0.8833   0.4828   0.8491  
0.45       0.8794   0.6156   0.9785   0.8833   0.4828   0.8491  
0.50       0.8794   0.6156   0.9785   0.8833   0.4828   0.8491  
0.55       0.8794   0.6155   0.9786   0.8833   0.4827   0.8491  
0.60       0.8794   0.6155   0.9786   0.8833   0.4827   0.8491  
0.65       0.8794   0.6155   0.9786   0.8833   0.4827   0.8491  
0.70       0.8794   0.6155   0.9786   0.8833   0.4827   0.8491  
0.75       0.8794   0.6155   0.9786   0.8833   0.4827   0.8491  
0.80       0.8794   0.6155   0.9786   0.8833   0.4827   0.8491  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8794, F1=0.6156, Normal Recall=0.9785, Normal Precision=0.8833, Attack Recall=0.4828, Attack Precision=0.8491

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
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 70% : Attack 30%  (n=194,230, N=135,961, A=58,269)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.8301   0.6303   0.9789   0.8154   0.4828   0.9076   <--
0.20       0.8301   0.6303   0.9789   0.8154   0.4828   0.9076  
0.25       0.8301   0.6303   0.9789   0.8154   0.4828   0.9076  
0.30       0.8301   0.6303   0.9789   0.8154   0.4828   0.9076  
0.35       0.8301   0.6303   0.9789   0.8154   0.4828   0.9076  
0.40       0.8301   0.6303   0.9789   0.8154   0.4828   0.9076  
0.45       0.8301   0.6303   0.9789   0.8154   0.4828   0.9076  
0.50       0.8301   0.6303   0.9789   0.8154   0.4828   0.9076  
0.55       0.8301   0.6302   0.9789   0.8154   0.4827   0.9076  
0.60       0.8301   0.6302   0.9789   0.8154   0.4827   0.9076  
0.65       0.8301   0.6302   0.9789   0.8154   0.4827   0.9076  
0.70       0.8301   0.6302   0.9789   0.8154   0.4827   0.9076  
0.75       0.8301   0.6302   0.9789   0.8154   0.4827   0.9076  
0.80       0.8301   0.6302   0.9789   0.8154   0.4827   0.9076  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8301, F1=0.6303, Normal Recall=0.9789, Normal Precision=0.8154, Attack Recall=0.4828, Attack Precision=0.9076

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
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 60% : Attack 40%  (n=145,675, N=87,405, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.7806   0.6377   0.9791   0.7396   0.4828   0.9389   <--
0.20       0.7806   0.6377   0.9791   0.7396   0.4828   0.9389  
0.25       0.7806   0.6377   0.9791   0.7396   0.4828   0.9389  
0.30       0.7806   0.6377   0.9791   0.7396   0.4828   0.9389  
0.35       0.7806   0.6377   0.9791   0.7396   0.4828   0.9389  
0.40       0.7806   0.6377   0.9791   0.7396   0.4828   0.9389  
0.45       0.7806   0.6377   0.9791   0.7396   0.4828   0.9389  
0.50       0.7806   0.6377   0.9791   0.7396   0.4828   0.9389  
0.55       0.7805   0.6376   0.9791   0.7395   0.4827   0.9389  
0.60       0.7805   0.6376   0.9791   0.7395   0.4827   0.9389  
0.65       0.7805   0.6376   0.9791   0.7395   0.4827   0.9389  
0.70       0.7805   0.6376   0.9791   0.7395   0.4827   0.9389  
0.75       0.7805   0.6376   0.9791   0.7395   0.4827   0.9389  
0.80       0.7805   0.6376   0.9791   0.7395   0.4827   0.9389  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7806, F1=0.6377, Normal Recall=0.9791, Normal Precision=0.7396, Attack Recall=0.4828, Attack Precision=0.9389

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
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 50% : Attack 50%  (n=116,540, N=58,270, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.7310   0.6422   0.9791   0.6544   0.4828   0.9585   <--
0.20       0.7310   0.6422   0.9791   0.6544   0.4828   0.9585  
0.25       0.7310   0.6422   0.9791   0.6544   0.4828   0.9585  
0.30       0.7310   0.6422   0.9791   0.6544   0.4828   0.9585  
0.35       0.7310   0.6422   0.9791   0.6544   0.4828   0.9585  
0.40       0.7310   0.6422   0.9791   0.6544   0.4828   0.9585  
0.45       0.7310   0.6422   0.9791   0.6544   0.4828   0.9585  
0.50       0.7310   0.6422   0.9791   0.6544   0.4828   0.9585  
0.55       0.7309   0.6421   0.9791   0.6543   0.4827   0.9585  
0.60       0.7309   0.6421   0.9791   0.6543   0.4827   0.9585  
0.65       0.7309   0.6421   0.9791   0.6543   0.4827   0.9585  
0.70       0.7309   0.6421   0.9791   0.6543   0.4827   0.9585  
0.75       0.7309   0.6421   0.9791   0.6543   0.4827   0.9585  
0.80       0.7309   0.6421   0.9791   0.6543   0.4827   0.9585  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7310, F1=0.6422, Normal Recall=0.9791, Normal Precision=0.6544, Attack Recall=0.4828, Attack Precision=0.9585

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
0.15       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278   <--
0.20       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.25       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.30       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.35       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.40       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.45       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.50       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.55       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.60       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.65       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.70       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.75       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
0.80       0.9548   0.7534   0.9840   0.9663   0.6913   0.8278  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9548, F1=0.7534, Normal Recall=0.9840, Normal Precision=0.9663, Attack Recall=0.6913, Attack Precision=0.8278

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
0.15       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152   <--
0.20       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.25       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.30       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.35       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.40       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.45       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.50       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.55       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.60       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.65       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.70       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.75       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
0.80       0.9253   0.7870   0.9840   0.9270   0.6903   0.9152  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9253, F1=0.7870, Normal Recall=0.9840, Normal Precision=0.9270, Attack Recall=0.6903, Attack Precision=0.9152

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
0.15       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499   <--
0.20       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.25       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.30       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.35       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.40       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.45       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.50       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.55       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.60       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.65       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.70       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.75       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
0.80       0.8962   0.7995   0.9844   0.8812   0.6903   0.9499  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8962, F1=0.7995, Normal Recall=0.9844, Normal Precision=0.8812, Attack Recall=0.6903, Attack Precision=0.9499

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
0.15       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677   <--
0.20       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.25       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.30       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.35       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.40       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.45       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.50       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.55       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.60       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.65       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.70       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.75       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
0.80       0.8669   0.8058   0.9846   0.8266   0.6903   0.9677  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8669, F1=0.8058, Normal Recall=0.9846, Normal Precision=0.8266, Attack Recall=0.6903, Attack Precision=0.9677

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
0.15       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783   <--
0.20       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.25       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.30       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.35       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.40       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.45       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.50       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.55       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.60       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.65       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.70       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.75       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
0.80       0.8375   0.8094   0.9847   0.7607   0.6903   0.9783  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8375, F1=0.8094, Normal Recall=0.9847, Normal Precision=0.7607, Attack Recall=0.6903, Attack Precision=0.9783

```

