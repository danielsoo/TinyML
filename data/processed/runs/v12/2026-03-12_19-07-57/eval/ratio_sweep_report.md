# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/federated_local_sky.yaml` |
| **Generated** | 2026-03-12 19:19:36 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | None |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 70 |
| **Local epochs** | 3 |
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8898 | 0.8290 | 0.7677 | 0.7063 | 0.6453 | 0.5847 | 0.5229 | 0.4615 | 0.4003 | 0.3391 | 0.2778 |
| QAT+PTQ | 0.9788 | 0.9692 | 0.9589 | 0.9486 | 0.9387 | 0.9287 | 0.9184 | 0.9087 | 0.8981 | 0.8881 | 0.8781 |
| noQAT+PTQ | 0.9861 | 0.9460 | 0.9053 | 0.8648 | 0.8244 | 0.7835 | 0.7435 | 0.7029 | 0.6624 | 0.6218 | 0.5813 |
| Compressed (QAT) | 0.9782 | 0.9716 | 0.9649 | 0.9580 | 0.9516 | 0.9451 | 0.9382 | 0.9318 | 0.9250 | 0.9182 | 0.9117 |
| Compressed (PTQ) | 0.9788 | 0.9692 | 0.9589 | 0.9486 | 0.9387 | 0.9287 | 0.9184 | 0.9087 | 0.8981 | 0.8881 | 0.8781 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2460 | 0.3236 | 0.3620 | 0.3852 | 0.4008 | 0.4113 | 0.4194 | 0.4256 | 0.4307 | 0.4348 |
| QAT+PTQ | 0.0000 | 0.8513 | 0.8952 | 0.9112 | 0.9197 | 0.9249 | 0.9282 | 0.9309 | 0.9324 | 0.9339 | 0.9351 |
| noQAT+PTQ | 0.0000 | 0.6838 | 0.7106 | 0.7206 | 0.7259 | 0.7286 | 0.7311 | 0.7326 | 0.7337 | 0.7345 | 0.7352 |
| Compressed (QAT) | 0.0000 | 0.8653 | 0.9121 | 0.9287 | 0.9378 | 0.9432 | 0.9465 | 0.9492 | 0.9511 | 0.9525 | 0.9538 |
| Compressed (PTQ) | 0.0000 | 0.8513 | 0.8952 | 0.9112 | 0.9197 | 0.9249 | 0.9282 | 0.9309 | 0.9324 | 0.9339 | 0.9351 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8898 | 0.8901 | 0.8902 | 0.8899 | 0.8902 | 0.8917 | 0.8905 | 0.8903 | 0.8901 | 0.8904 | 0.0000 |
| QAT+PTQ | 0.9788 | 0.9792 | 0.9791 | 0.9789 | 0.9790 | 0.9793 | 0.9789 | 0.9800 | 0.9782 | 0.9783 | 0.0000 |
| noQAT+PTQ | 0.9861 | 0.9861 | 0.9863 | 0.9863 | 0.9865 | 0.9858 | 0.9869 | 0.9869 | 0.9871 | 0.9868 | 0.0000 |
| Compressed (QAT) | 0.9782 | 0.9782 | 0.9782 | 0.9779 | 0.9783 | 0.9785 | 0.9780 | 0.9786 | 0.9783 | 0.9765 | 0.0000 |
| Compressed (PTQ) | 0.9788 | 0.9792 | 0.9791 | 0.9789 | 0.9790 | 0.9793 | 0.9789 | 0.9800 | 0.9782 | 0.9783 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8898 | 0.0000 | 0.0000 | 0.0000 | 0.8898 | 1.0000 |
| 90 | 10 | 460,810 | 0.8290 | 0.2200 | 0.2789 | 0.2460 | 0.8901 | 0.9174 |
| 80 | 20 | 425,865 | 0.7677 | 0.3874 | 0.2778 | 0.3236 | 0.8902 | 0.8314 |
| 70 | 30 | 283,910 | 0.7063 | 0.5195 | 0.2778 | 0.3620 | 0.8899 | 0.7419 |
| 60 | 40 | 212,930 | 0.6453 | 0.6279 | 0.2778 | 0.3852 | 0.8902 | 0.6490 |
| 50 | 50 | 170,346 | 0.5847 | 0.7194 | 0.2778 | 0.4008 | 0.8917 | 0.5525 |
| 40 | 60 | 141,955 | 0.5229 | 0.7919 | 0.2778 | 0.4113 | 0.8905 | 0.4512 |
| 30 | 70 | 121,672 | 0.4615 | 0.8553 | 0.2778 | 0.4194 | 0.8903 | 0.3457 |
| 20 | 80 | 106,465 | 0.4003 | 0.9100 | 0.2778 | 0.4256 | 0.8901 | 0.2355 |
| 10 | 90 | 94,630 | 0.3391 | 0.9580 | 0.2778 | 0.4307 | 0.8904 | 0.1205 |
| 0 | 100 | 85,173 | 0.2778 | 1.0000 | 0.2778 | 0.4348 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9788 | 0.0000 | 0.0000 | 0.0000 | 0.9788 | 1.0000 |
| 90 | 10 | 460,810 | 0.9692 | 0.8244 | 0.8800 | 0.8513 | 0.9792 | 0.9866 |
| 80 | 20 | 425,865 | 0.9589 | 0.9130 | 0.8781 | 0.8952 | 0.9791 | 0.9698 |
| 70 | 30 | 283,910 | 0.9486 | 0.9468 | 0.8781 | 0.9112 | 0.9789 | 0.9493 |
| 60 | 40 | 212,930 | 0.9387 | 0.9654 | 0.8781 | 0.9197 | 0.9790 | 0.9234 |
| 50 | 50 | 170,346 | 0.9287 | 0.9770 | 0.8781 | 0.9249 | 0.9793 | 0.8893 |
| 40 | 60 | 141,955 | 0.9184 | 0.9842 | 0.8781 | 0.9282 | 0.9789 | 0.8426 |
| 30 | 70 | 121,672 | 0.9087 | 0.9903 | 0.8781 | 0.9309 | 0.9800 | 0.7751 |
| 20 | 80 | 106,465 | 0.8981 | 0.9938 | 0.8781 | 0.9324 | 0.9782 | 0.6674 |
| 10 | 90 | 94,630 | 0.8881 | 0.9973 | 0.8781 | 0.9339 | 0.9783 | 0.4714 |
| 0 | 100 | 85,173 | 0.8781 | 1.0000 | 0.8781 | 0.9351 | 0.0000 | 0.0000 |

### noQAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9861 | 0.0000 | 0.0000 | 0.0000 | 0.9861 | 1.0000 |
| 90 | 10 | 460,810 | 0.9460 | 0.8242 | 0.5843 | 0.6838 | 0.9861 | 0.9553 |
| 80 | 20 | 425,865 | 0.9053 | 0.9139 | 0.5813 | 0.7106 | 0.9863 | 0.9040 |
| 70 | 30 | 283,910 | 0.8648 | 0.9480 | 0.5813 | 0.7206 | 0.9863 | 0.8461 |
| 60 | 40 | 212,930 | 0.8244 | 0.9662 | 0.5813 | 0.7259 | 0.9865 | 0.7794 |
| 50 | 50 | 170,346 | 0.7835 | 0.9762 | 0.5813 | 0.7286 | 0.9858 | 0.7019 |
| 40 | 60 | 141,955 | 0.7435 | 0.9852 | 0.5813 | 0.7311 | 0.9869 | 0.6111 |
| 30 | 70 | 121,672 | 0.7029 | 0.9905 | 0.5812 | 0.7326 | 0.9869 | 0.5025 |
| 20 | 80 | 106,465 | 0.6624 | 0.9945 | 0.5812 | 0.7337 | 0.9871 | 0.3708 |
| 10 | 90 | 94,630 | 0.6218 | 0.9975 | 0.5812 | 0.7345 | 0.9868 | 0.2075 |
| 0 | 100 | 85,173 | 0.5813 | 1.0000 | 0.5813 | 0.7352 | 0.0000 | 0.0000 |

### Compressed (QAT)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9782 | 0.0000 | 0.0000 | 0.0000 | 0.9782 | 1.0000 |
| 90 | 10 | 460,810 | 0.9716 | 0.8228 | 0.9124 | 0.8653 | 0.9782 | 0.9901 |
| 80 | 20 | 425,865 | 0.9649 | 0.9126 | 0.9117 | 0.9121 | 0.9782 | 0.9779 |
| 70 | 30 | 283,910 | 0.9580 | 0.9464 | 0.9117 | 0.9287 | 0.9779 | 0.9627 |
| 60 | 40 | 212,930 | 0.9516 | 0.9655 | 0.9117 | 0.9378 | 0.9783 | 0.9432 |
| 50 | 50 | 170,346 | 0.9451 | 0.9770 | 0.9117 | 0.9432 | 0.9785 | 0.9172 |
| 40 | 60 | 141,955 | 0.9382 | 0.9842 | 0.9117 | 0.9465 | 0.9780 | 0.8807 |
| 30 | 70 | 121,672 | 0.9318 | 0.9900 | 0.9117 | 0.9492 | 0.9786 | 0.8260 |
| 20 | 80 | 106,465 | 0.9250 | 0.9941 | 0.9117 | 0.9511 | 0.9783 | 0.7347 |
| 10 | 90 | 94,630 | 0.9182 | 0.9971 | 0.9117 | 0.9525 | 0.9765 | 0.5512 |
| 0 | 100 | 85,173 | 0.9117 | 1.0000 | 0.9117 | 0.9538 | 0.0000 | 0.0000 |

### Compressed (PTQ)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9788 | 0.0000 | 0.0000 | 0.0000 | 0.9788 | 1.0000 |
| 90 | 10 | 460,810 | 0.9692 | 0.8244 | 0.8800 | 0.8513 | 0.9792 | 0.9866 |
| 80 | 20 | 425,865 | 0.9589 | 0.9130 | 0.8781 | 0.8952 | 0.9791 | 0.9698 |
| 70 | 30 | 283,910 | 0.9486 | 0.9468 | 0.8781 | 0.9112 | 0.9789 | 0.9493 |
| 60 | 40 | 212,930 | 0.9387 | 0.9654 | 0.8781 | 0.9197 | 0.9790 | 0.9234 |
| 50 | 50 | 170,346 | 0.9287 | 0.9770 | 0.8781 | 0.9249 | 0.9793 | 0.8893 |
| 40 | 60 | 141,955 | 0.9184 | 0.9842 | 0.8781 | 0.9282 | 0.9789 | 0.8426 |
| 30 | 70 | 121,672 | 0.9087 | 0.9903 | 0.8781 | 0.9309 | 0.9800 | 0.7751 |
| 20 | 80 | 106,465 | 0.8981 | 0.9938 | 0.8781 | 0.9324 | 0.9782 | 0.6674 |
| 10 | 90 | 94,630 | 0.8881 | 0.9973 | 0.8781 | 0.9339 | 0.9783 | 0.4714 |
| 0 | 100 | 85,173 | 0.8781 | 1.0000 | 0.8781 | 0.9351 | 0.0000 | 0.0000 |


## Threshold Tuning (Original)

Model: `models/tflite/saved_model_original.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 90% : Attack 10%  (n=460,810, N=414,729, A=46,081)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.1009   0.1819   0.0011   0.9274   0.9992   0.1000  
0.20       0.1011   0.1816   0.0015   0.8300   0.9973   0.0999  
0.25       0.1021   0.1817   0.0027   0.8846   0.9968   0.1000  
0.30       0.1038   0.1818   0.0047   0.9111   0.9959   0.1001  
0.35       0.1038   0.1780   0.0075   0.6933   0.9700   0.0980  
0.40       0.1036   0.1644   0.0171   0.5657   0.8820   0.0907  
0.45       0.1286   0.1378   0.0655   0.6600   0.6963   0.0765  
0.50       0.8290   0.2463   0.8901   0.9175   0.2793   0.2202   <--
0.55       0.8892   0.0597   0.9841   0.9018   0.0352   0.1969  
0.60       0.8950   0.0001   0.9945   0.8995   0.0001   0.0013  
0.65       0.8981   0.0000   0.9978   0.8998   0.0000   0.0011  
0.70       0.8996   0.0000   0.9996   0.9000   0.0000   0.0057  
0.75       0.9000   0.0000   1.0000   0.9000   0.0000   0.0500  
0.80       0.9000   0.0000   1.0000   0.9000   0.0000   0.1667  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.8290, F1=0.2463, Normal Recall=0.8901, Normal Precision=0.9175, Attack Recall=0.2793, Attack Precision=0.2202

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 80% : Attack 20%  (n=425,865, N=340,692, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.2008   0.3334   0.0011   0.8568   0.9992   0.2001   <--
0.20       0.2007   0.3329   0.0015   0.6943   0.9973   0.1998  
0.25       0.2016   0.3330   0.0028   0.7725   0.9968   0.1999  
0.30       0.2029   0.3332   0.0046   0.8139   0.9957   0.2001  
0.35       0.1999   0.3265   0.0074   0.4943   0.9695   0.1963  
0.40       0.1900   0.3036   0.0169   0.3653   0.8828   0.1833  
0.45       0.1913   0.2556   0.0655   0.4616   0.6944   0.1567  
0.50       0.7676   0.3235   0.8900   0.8314   0.2778   0.3871  
0.55       0.7942   0.0642   0.9840   0.8031   0.0353   0.3548  
0.60       0.7956   0.0002   0.9945   0.7991   0.0001   0.0037  
0.65       0.7983   0.0001   0.9978   0.7997   0.0000   0.0054  
0.70       0.7997   0.0001   0.9996   0.7999   0.0000   0.0276  
0.75       0.8000   0.0001   1.0000   0.8000   0.0000   0.2000  
0.80       0.8000   0.0001   1.0000   0.8000   0.0000   0.4286  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2008, F1=0.3334, Normal Recall=0.0011, Normal Precision=0.8568, Attack Recall=0.9992, Attack Precision=0.2001

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 70% : Attack 30%  (n=283,910, N=198,737, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.3006   0.4615   0.0011   0.7759   0.9992   0.3001   <--
0.20       0.3002   0.4610   0.0015   0.5630   0.9973   0.2997  
0.25       0.3009   0.4610   0.0026   0.6546   0.9968   0.2999  
0.30       0.3019   0.4611   0.0045   0.7116   0.9957   0.3001  
0.35       0.2959   0.4524   0.0072   0.3551   0.9695   0.2950  
0.40       0.2764   0.4226   0.0165   0.2477   0.8828   0.2778  
0.45       0.2536   0.3582   0.0647   0.3306   0.6944   0.2414  
0.50       0.7062   0.3620   0.8898   0.7419   0.2778   0.5193  
0.55       0.6994   0.0658   0.9840   0.7041   0.0353   0.4855  
0.60       0.6962   0.0002   0.9945   0.6989   0.0001   0.0063  
0.65       0.6985   0.0001   0.9979   0.6996   0.0000   0.0094  
0.70       0.6997   0.0001   0.9996   0.6999   0.0000   0.0465  
0.75       0.7000   0.0001   1.0000   0.7000   0.0000   0.3077  
0.80       0.7000   0.0001   1.0000   0.7000   0.0000   0.6000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3006, F1=0.4615, Normal Recall=0.0011, Normal Precision=0.7759, Attack Recall=0.9992, Attack Precision=0.3001

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 60% : Attack 40%  (n=212,930, N=127,758, A=85,172)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.4004   0.5714   0.0011   0.6919   0.9992   0.4001   <--
0.20       0.3998   0.5707   0.0015   0.4599   0.9973   0.3997  
0.25       0.4003   0.5708   0.0027   0.5527   0.9968   0.3999  
0.30       0.4010   0.5708   0.0046   0.6177   0.9957   0.4001  
0.35       0.3922   0.5607   0.0073   0.2651   0.9695   0.3944  
0.40       0.3630   0.5258   0.0166   0.1749   0.8828   0.3744  
0.45       0.3166   0.4484   0.0648   0.2413   0.6944   0.3311  
0.50       0.6449   0.3849   0.8896   0.6488   0.2778   0.6266  
0.55       0.6045   0.0666   0.9840   0.6048   0.0353   0.5960  
0.60       0.5968   0.0002   0.9945   0.5987   0.0001   0.0099  
0.65       0.5988   0.0001   0.9979   0.5995   0.0000   0.0147  
0.70       0.5998   0.0001   0.9996   0.5999   0.0000   0.0714  
0.75       0.6000   0.0001   1.0000   0.6000   0.0000   0.4444  
0.80       0.6000   0.0001   1.0000   0.6000   0.0000   0.7500  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4004, F1=0.5714, Normal Recall=0.0011, Normal Precision=0.6919, Attack Recall=0.9992, Attack Precision=0.4001

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 50% : Attack 50%  (n=170,346, N=85,173, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.5003   0.6666   0.0013   0.6264   0.9992   0.5001   <--
0.20       0.4995   0.6658   0.0017   0.3827   0.9973   0.4997  
0.25       0.4997   0.6658   0.0027   0.4567   0.9968   0.4999  
0.30       0.5003   0.6658   0.0048   0.5293   0.9957   0.5001  
0.35       0.4886   0.6547   0.0076   0.1996   0.9695   0.4942  
0.40       0.4500   0.6161   0.0172   0.1279   0.8828   0.4732  
0.45       0.3796   0.5281   0.0649   0.1751   0.6944   0.4261  
0.50       0.5841   0.4005   0.8904   0.5521   0.2778   0.7170  
0.55       0.5098   0.0672   0.9843   0.5050   0.0353   0.6920  
0.60       0.4974   0.0002   0.9946   0.4987   0.0001   0.0151  
0.65       0.4990   0.0001   0.9980   0.4995   0.0000   0.0225  
0.70       0.4998   0.0001   0.9996   0.4999   0.0000   0.1111  
0.75       0.5000   0.0001   1.0000   0.5000   0.0000   0.5714  
0.80       0.5000   0.0001   1.0000   0.5000   0.0000   0.7500  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5003, F1=0.6666, Normal Recall=0.0013, Normal Precision=0.6264, Attack Recall=0.9992, Attack Precision=0.5001

```


## Threshold Tuning (QAT+PTQ)

Model: `models/tflite/saved_model_qat_ptq.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 90% : Attack 10%  (n=460,810, N=414,729, A=46,081)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9548   0.8090   0.9547   0.9950   0.9565   0.7009  
0.20       0.9609   0.8294   0.9621   0.9943   0.9502   0.7359  
0.25       0.9655   0.8417   0.9709   0.9906   0.9169   0.7779  
0.30       0.9688   0.8516   0.9772   0.9881   0.8937   0.8132  
0.35       0.9692   0.8528   0.9776   0.9880   0.8929   0.8161  
0.40       0.9694   0.8536   0.9781   0.9878   0.8911   0.8191   <--
0.45       0.9690   0.8498   0.9792   0.9863   0.8773   0.8239  
0.50       0.9690   0.8498   0.9792   0.9863   0.8773   0.8239  
0.55       0.9695   0.8500   0.9812   0.9848   0.8639   0.8365  
0.60       0.9684   0.8418   0.9826   0.9823   0.8406   0.8429  
0.65       0.9694   0.8408   0.9872   0.9789   0.8087   0.8755  
0.70       0.9708   0.8424   0.9919   0.9760   0.7809   0.9144  
0.75       0.9733   0.8489   0.9979   0.9731   0.7513   0.9758  
0.80       0.9724   0.8418   0.9989   0.9713   0.7339   0.9869  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9694, F1=0.8536, Normal Recall=0.9781, Normal Precision=0.9878, Attack Recall=0.8911, Attack Precision=0.8191

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 80% : Attack 20%  (n=425,865, N=340,692, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9548   0.8943   0.9545   0.9886   0.9561   0.8401  
0.20       0.9596   0.9040   0.9621   0.9871   0.9499   0.8623   <--
0.25       0.9603   0.9025   0.9709   0.9794   0.9182   0.8873  
0.30       0.9607   0.9010   0.9771   0.9738   0.8948   0.9073  
0.35       0.9609   0.9014   0.9776   0.9736   0.8940   0.9088  
0.40       0.9609   0.9012   0.9781   0.9732   0.8921   0.9104  
0.45       0.9589   0.8952   0.9791   0.9698   0.8781   0.9130  
0.50       0.9589   0.8952   0.9791   0.9698   0.8781   0.9130  
0.55       0.9578   0.8912   0.9811   0.9666   0.8644   0.9198  
0.60       0.9542   0.8802   0.9825   0.9611   0.8411   0.9232  
0.65       0.9516   0.8699   0.9872   0.9539   0.8092   0.9405  
0.70       0.9499   0.8620   0.9919   0.9479   0.7821   0.9601  
0.75       0.9490   0.8553   0.9979   0.9418   0.7533   0.9892  
0.80       0.9461   0.8450   0.9989   0.9377   0.7347   0.9942  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9596, F1=0.9040, Normal Recall=0.9621, Normal Precision=0.9871, Attack Recall=0.9499, Attack Precision=0.8623

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 70% : Attack 30%  (n=283,910, N=198,737, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9547   0.9268   0.9541   0.9807   0.9561   0.8993  
0.20       0.9582   0.9317   0.9618   0.9782   0.9499   0.9142   <--
0.25       0.9550   0.9244   0.9707   0.9652   0.9182   0.9307  
0.30       0.9522   0.9183   0.9768   0.9559   0.8948   0.9430  
0.35       0.9523   0.9183   0.9772   0.9556   0.8940   0.9439  
0.40       0.9520   0.9178   0.9777   0.9548   0.8921   0.9449  
0.45       0.9486   0.9111   0.9788   0.9493   0.8781   0.9466  
0.50       0.9486   0.9111   0.9788   0.9493   0.8781   0.9466  
0.55       0.9460   0.9056   0.9809   0.9441   0.8644   0.9510  
0.60       0.9399   0.8936   0.9823   0.9352   0.8411   0.9531  
0.65       0.9336   0.8797   0.9869   0.9235   0.8092   0.9637  
0.70       0.9288   0.8683   0.9917   0.9139   0.7821   0.9759  
0.75       0.9246   0.8570   0.9980   0.9042   0.7533   0.9937  
0.80       0.9197   0.8459   0.9990   0.8978   0.7347   0.9968  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9582, F1=0.9317, Normal Recall=0.9618, Normal Precision=0.9782, Attack Recall=0.9499, Attack Precision=0.9142

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 60% : Attack 40%  (n=212,930, N=127,758, A=85,172)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9548   0.9442   0.9540   0.9702   0.9561   0.9326  
0.20       0.9571   0.9465   0.9618   0.9664   0.9499   0.9432   <--
0.25       0.9497   0.9359   0.9706   0.9468   0.9182   0.9542  
0.30       0.9439   0.9273   0.9766   0.9330   0.8948   0.9622  
0.35       0.9438   0.9272   0.9770   0.9326   0.8940   0.9629  
0.40       0.9434   0.9265   0.9775   0.9315   0.8921   0.9636  
0.45       0.9384   0.9194   0.9786   0.9233   0.8781   0.9648  
0.50       0.9384   0.9194   0.9786   0.9233   0.8781   0.9648  
0.55       0.9342   0.9131   0.9807   0.9156   0.8644   0.9676  
0.60       0.9257   0.9005   0.9821   0.9026   0.8411   0.9690  
0.65       0.9158   0.8849   0.9869   0.8858   0.8092   0.9763  
0.70       0.9079   0.8716   0.9917   0.8722   0.7821   0.9843  
0.75       0.9001   0.8578   0.9979   0.8585   0.7533   0.9959  
0.80       0.8933   0.8463   0.9990   0.8496   0.7347   0.9979  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9571, F1=0.9465, Normal Recall=0.9618, Normal Precision=0.9664, Attack Recall=0.9499, Attack Precision=0.9432

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 50% : Attack 50%  (n=170,346, N=85,173, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9549   0.9549   0.9537   0.9560   0.9561   0.9538  
0.20       0.9557   0.9555   0.9616   0.9505   0.9499   0.9611   <--
0.25       0.9445   0.9430   0.9708   0.9223   0.9182   0.9692  
0.30       0.9357   0.9330   0.9766   0.9028   0.8948   0.9745  
0.35       0.9355   0.9327   0.9770   0.9021   0.8940   0.9750  
0.40       0.9348   0.9319   0.9776   0.9006   0.8921   0.9755  
0.45       0.9284   0.9246   0.9786   0.8893   0.8781   0.9762  
0.50       0.9284   0.9246   0.9786   0.8893   0.8781   0.9762  
0.55       0.9225   0.9177   0.9806   0.8785   0.8644   0.9780  
0.60       0.9115   0.9048   0.9820   0.8607   0.8411   0.9790  
0.65       0.8980   0.8881   0.9868   0.8380   0.8092   0.9840  
0.70       0.8869   0.8737   0.9917   0.8198   0.7821   0.9895  
0.75       0.8756   0.8583   0.9979   0.8018   0.7533   0.9972  
0.80       0.8668   0.8466   0.9990   0.7902   0.7347   0.9986  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9557, F1=0.9555, Normal Recall=0.9616, Normal Precision=0.9505, Attack Recall=0.9499, Attack Precision=0.9611

```


## Threshold Tuning (noQAT+PTQ)

Model: `models/tflite/saved_model_no_qat_ptq.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 90% : Attack 10%  (n=460,810, N=414,729, A=46,081)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.1068   0.1830   0.0076   1.0000   1.0000   0.1007  
0.20       0.1096   0.1834   0.0107   0.9995   1.0000   0.1010  
0.25       0.1128   0.1840   0.0142   0.9997   1.0000   0.1013  
0.30       0.1180   0.1848   0.0200   0.9995   0.9999   0.1018  
0.35       0.1227   0.1856   0.0252   0.9996   0.9999   0.1023  
0.40       0.1432   0.1892   0.0481   0.9998   0.9999   0.1045  
0.45       0.1714   0.1944   0.0794   0.9998   0.9999   0.1077  
0.50       0.9456   0.6808   0.9861   0.9549   0.5804   0.8232  
0.55       0.9537   0.7136   0.9955   0.9549   0.5770   0.9348  
0.60       0.9544   0.7163   0.9966   0.9548   0.5751   0.9495  
0.65       0.9551   0.7183   0.9978   0.9545   0.5717   0.9658  
0.70       0.9557   0.7201   0.9985   0.9544   0.5703   0.9767  
0.75       0.9563   0.7219   0.9994   0.9541   0.5677   0.9911   <--
0.80       0.9563   0.7212   0.9998   0.9539   0.5652   0.9962  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.75
  At threshold 0.75: Accuracy=0.9563, F1=0.7219, Normal Recall=0.9994, Normal Precision=0.9541, Attack Recall=0.5677, Attack Precision=0.9911

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 80% : Attack 20%  (n=425,865, N=340,692, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.2061   0.3350   0.0076   0.9996   1.0000   0.2012  
0.20       0.2085   0.3357   0.0106   0.9986   0.9999   0.2017  
0.25       0.2113   0.3365   0.0142   0.9990   0.9999   0.2023  
0.30       0.2160   0.3378   0.0200   0.9988   0.9999   0.2032  
0.35       0.2200   0.3390   0.0251   0.9991   0.9999   0.2041  
0.40       0.2385   0.3444   0.0482   0.9995   0.9999   0.2080  
0.45       0.2635   0.3519   0.0795   0.9996   0.9999   0.2136  
0.50       0.9051   0.7102   0.9861   0.9040   0.5813   0.9125  
0.55       0.9119   0.7241   0.9955   0.9041   0.5778   0.9696  
0.60       0.9124   0.7244   0.9965   0.9038   0.5758   0.9765   <--
0.65       0.9127   0.7241   0.9977   0.9033   0.5727   0.9843  
0.70       0.9130   0.7243   0.9985   0.9031   0.5713   0.9895  
0.75       0.9133   0.7239   0.9994   0.9026   0.5686   0.9959  
0.80       0.9130   0.7225   0.9998   0.9021   0.5661   0.9983  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.6
  At threshold 0.6: Accuracy=0.9124, F1=0.7244, Normal Recall=0.9965, Normal Precision=0.9038, Attack Recall=0.5758, Attack Precision=0.9765

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 70% : Attack 30%  (n=283,910, N=198,737, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.3052   0.4634   0.0074   0.9993   1.0000   0.3016  
0.20       0.3073   0.4641   0.0105   0.9976   0.9999   0.3022  
0.25       0.3099   0.4651   0.0141   0.9982   0.9999   0.3030  
0.30       0.3138   0.4665   0.0198   0.9980   0.9999   0.3042  
0.35       0.3175   0.4678   0.0250   0.9984   0.9999   0.3053  
0.40       0.3337   0.4738   0.0482   0.9992   0.9999   0.3104  
0.45       0.3556   0.4821   0.0795   0.9994   0.9999   0.3176  
0.50       0.8649   0.7208   0.9865   0.8461   0.5813   0.9485  
0.55       0.8703   0.7277   0.9956   0.8462   0.5778   0.9826   <--
0.60       0.8704   0.7272   0.9967   0.8457   0.5758   0.9867  
0.65       0.8703   0.7260   0.9978   0.8449   0.5727   0.9911  
0.70       0.8703   0.7255   0.9985   0.8446   0.5713   0.9938  
0.75       0.8702   0.7244   0.9994   0.8439   0.5686   0.9977  
0.80       0.8697   0.7227   0.9998   0.8432   0.5661   0.9991  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.8703, F1=0.7277, Normal Recall=0.9956, Normal Precision=0.8462, Attack Recall=0.5778, Attack Precision=0.9826

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 60% : Attack 40%  (n=212,930, N=127,758, A=85,172)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.4045   0.5733   0.0075   0.9990   1.0000   0.4018  
0.20       0.4063   0.5740   0.0105   0.9963   0.9999   0.4025  
0.25       0.4085   0.5749   0.0141   0.9972   0.9999   0.4034  
0.30       0.4119   0.5763   0.0199   0.9969   0.9999   0.4048  
0.35       0.4152   0.5777   0.0254   0.9975   0.9999   0.4062  
0.40       0.4290   0.5835   0.0483   0.9987   0.9999   0.4119  
0.45       0.4477   0.5915   0.0795   0.9990   0.9999   0.4200  
0.50       0.8243   0.7258   0.9864   0.7794   0.5813   0.9660  
0.55       0.8285   0.7294   0.9956   0.7796   0.5778   0.9886   <--
0.60       0.8283   0.7284   0.9966   0.7790   0.5758   0.9912  
0.65       0.8277   0.7268   0.9977   0.7779   0.5727   0.9941  
0.70       0.8276   0.7261   0.9984   0.7774   0.5713   0.9959  
0.75       0.8271   0.7246   0.9994   0.7765   0.5686   0.9985  
0.80       0.8263   0.7228   0.9997   0.7756   0.5661   0.9993  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.8285, F1=0.7294, Normal Recall=0.9956, Normal Precision=0.7796, Attack Recall=0.5778, Attack Precision=0.9886

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 50% : Attack 50%  (n=170,346, N=85,173, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.5037   0.6683   0.0075   0.9984   1.0000   0.5019  
0.20       0.5053   0.6690   0.0107   0.9945   0.9999   0.5027  
0.25       0.5071   0.6698   0.0143   0.9959   0.9999   0.5036  
0.30       0.5099   0.6711   0.0199   0.9953   0.9999   0.5050  
0.35       0.5127   0.6723   0.0254   0.9963   0.9999   0.5064  
0.40       0.5242   0.6776   0.0484   0.9981   0.9999   0.5124  
0.45       0.5399   0.6849   0.0799   0.9985   0.9999   0.5208  
0.50       0.7836   0.7287   0.9859   0.7019   0.5813   0.9763  
0.55       0.7866   0.7303   0.9954   0.7022   0.5778   0.9920   <--
0.60       0.7861   0.7291   0.9964   0.7014   0.5758   0.9937  
0.65       0.7852   0.7272   0.9976   0.7001   0.5727   0.9958  
0.70       0.7848   0.7263   0.9983   0.6995   0.5713   0.9969  
0.75       0.7840   0.7247   0.9994   0.6985   0.5686   0.9989  
0.80       0.7829   0.7228   0.9997   0.6973   0.5661   0.9995  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.7866, F1=0.7303, Normal Recall=0.9954, Normal Precision=0.7022, Attack Recall=0.5778, Attack Precision=0.9920

```


## Threshold Tuning (saved_model_pruned_qat)

Model: `models/tflite/saved_model_pruned_qat.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 90% : Attack 10%  (n=460,810, N=414,729, A=46,081)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9400   0.7653   0.9357   0.9974   0.9783   0.6285  
0.20       0.9466   0.7853   0.9433   0.9972   0.9764   0.6567  
0.25       0.9485   0.7913   0.9454   0.9972   0.9762   0.6653  
0.30       0.9670   0.8519   0.9690   0.9942   0.9491   0.7728  
0.35       0.9672   0.8522   0.9698   0.9937   0.9445   0.7763  
0.40       0.9686   0.8568   0.9718   0.9931   0.9395   0.7874  
0.45       0.9691   0.8586   0.9724   0.9931   0.9392   0.7908  
0.50       0.9715   0.8647   0.9782   0.9900   0.9112   0.8226  
0.55       0.9706   0.8584   0.9796   0.9877   0.8903   0.8288  
0.60       0.9735   0.8692   0.9841   0.9865   0.8788   0.8598  
0.65       0.9726   0.8603   0.9870   0.9827   0.8432   0.8780  
0.70       0.9760   0.8711   0.9945   0.9792   0.8097   0.9425  
0.75       0.9764   0.8715   0.9961   0.9781   0.7994   0.9579   <--
0.80       0.9757   0.8648   0.9976   0.9759   0.7784   0.9729  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.75
  At threshold 0.75: Accuracy=0.9764, F1=0.8715, Normal Recall=0.9961, Normal Precision=0.9781, Attack Recall=0.7994, Attack Precision=0.9579

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 80% : Attack 20%  (n=425,865, N=340,692, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9442   0.8751   0.9357   0.9942   0.9782   0.7917  
0.20       0.9498   0.8862   0.9432   0.9938   0.9765   0.8111  
0.25       0.9515   0.8895   0.9453   0.9938   0.9763   0.8169  
0.30       0.9651   0.9159   0.9690   0.9872   0.9497   0.8844  
0.35       0.9649   0.9150   0.9697   0.9861   0.9454   0.8864  
0.40       0.9655   0.9159   0.9718   0.9849   0.9403   0.8928  
0.45       0.9659   0.9168   0.9724   0.9848   0.9400   0.8948   <--
0.50       0.9648   0.9121   0.9781   0.9779   0.9117   0.9125  
0.55       0.9617   0.9029   0.9796   0.9728   0.8902   0.9158  
0.60       0.9630   0.9048   0.9841   0.9701   0.8789   0.9324  
0.65       0.9583   0.8901   0.9870   0.9619   0.8438   0.9418  
0.70       0.9577   0.8847   0.9946   0.9545   0.8104   0.9739  
0.75       0.9569   0.8812   0.9961   0.9522   0.7998   0.9810  
0.80       0.9538   0.8708   0.9976   0.9474   0.7786   0.9878  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9659, F1=0.9168, Normal Recall=0.9724, Normal Precision=0.9848, Attack Recall=0.9400, Attack Precision=0.8948

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 70% : Attack 30%  (n=283,910, N=198,737, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9482   0.9189   0.9354   0.9901   0.9782   0.8664  
0.20       0.9529   0.9255   0.9427   0.9894   0.9765   0.8796  
0.25       0.9543   0.9276   0.9448   0.9894   0.9763   0.8835  
0.30       0.9630   0.9391   0.9688   0.9782   0.9497   0.9287   <--
0.35       0.9623   0.9376   0.9695   0.9764   0.9454   0.9300  
0.40       0.9621   0.9371   0.9715   0.9743   0.9403   0.9339  
0.45       0.9624   0.9375   0.9720   0.9742   0.9400   0.9351  
0.50       0.9580   0.9286   0.9778   0.9627   0.9117   0.9462  
0.55       0.9526   0.9185   0.9793   0.9542   0.8902   0.9486  
0.60       0.9523   0.9171   0.9838   0.9499   0.8789   0.9587  
0.65       0.9439   0.9002   0.9868   0.9365   0.8438   0.9647  
0.70       0.9392   0.8889   0.9945   0.9245   0.8104   0.9843  
0.75       0.9372   0.8843   0.9961   0.9207   0.7998   0.9888  
0.80       0.9319   0.8728   0.9976   0.9131   0.7786   0.9928  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9630, F1=0.9391, Normal Recall=0.9688, Normal Precision=0.9782, Attack Recall=0.9497, Attack Precision=0.9287

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 60% : Attack 40%  (n=212,930, N=127,758, A=85,172)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9520   0.9423   0.9346   0.9847   0.9782   0.9088  
0.20       0.9560   0.9466   0.9422   0.9837   0.9765   0.9185  
0.25       0.9572   0.9480   0.9445   0.9835   0.9763   0.9214  
0.30       0.9612   0.9514   0.9688   0.9665   0.9497   0.9531   <--
0.35       0.9599   0.9497   0.9696   0.9638   0.9454   0.9540  
0.40       0.9591   0.9484   0.9716   0.9607   0.9403   0.9567  
0.45       0.9593   0.9487   0.9722   0.9605   0.9400   0.9575  
0.50       0.9514   0.9376   0.9779   0.9432   0.9117   0.9649  
0.55       0.9437   0.9267   0.9793   0.9305   0.8902   0.9663  
0.60       0.9417   0.9235   0.9837   0.9241   0.8789   0.9729  
0.65       0.9295   0.9054   0.9866   0.9045   0.8438   0.9768  
0.70       0.9208   0.8911   0.9944   0.8872   0.8104   0.9897  
0.75       0.9175   0.8858   0.9960   0.8818   0.7998   0.9926  
0.80       0.9099   0.8737   0.9975   0.8711   0.7786   0.9952  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9612, F1=0.9514, Normal Recall=0.9688, Normal Precision=0.9665, Attack Recall=0.9497, Attack Precision=0.9531

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 50% : Attack 50%  (n=170,346, N=85,173, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9563   0.9573   0.9344   0.9772   0.9782   0.9372  
0.20       0.9593   0.9600   0.9421   0.9757   0.9765   0.9441  
0.25       0.9604   0.9610   0.9445   0.9755   0.9763   0.9462   <--
0.30       0.9594   0.9590   0.9690   0.9506   0.9497   0.9684  
0.35       0.9576   0.9571   0.9697   0.9467   0.9454   0.9690  
0.40       0.9560   0.9553   0.9717   0.9421   0.9403   0.9708  
0.45       0.9561   0.9554   0.9723   0.9419   0.9400   0.9713  
0.50       0.9449   0.9430   0.9781   0.9172   0.9117   0.9765  
0.55       0.9349   0.9318   0.9795   0.8992   0.8902   0.9775  
0.60       0.9313   0.9275   0.9836   0.8903   0.8789   0.9817  
0.65       0.9152   0.9087   0.9866   0.8633   0.8438   0.9844  
0.70       0.9024   0.8925   0.9944   0.8399   0.8104   0.9932  
0.75       0.8979   0.8868   0.9961   0.8326   0.7998   0.9951  
0.80       0.8881   0.8743   0.9975   0.8184   0.7786   0.9968  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.9604, F1=0.9610, Normal Recall=0.9445, Normal Precision=0.9755, Attack Recall=0.9763, Attack Precision=0.9462

```


## Threshold Tuning (Compressed (PTQ))

Model: `models/tflite/saved_model_pruned_quantized.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 90% : Attack 10%  (n=460,810, N=414,729, A=46,081)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9548   0.8090   0.9547   0.9950   0.9565   0.7009  
0.20       0.9609   0.8294   0.9621   0.9943   0.9502   0.7359  
0.25       0.9655   0.8417   0.9709   0.9906   0.9169   0.7779  
0.30       0.9688   0.8516   0.9772   0.9881   0.8937   0.8132  
0.35       0.9692   0.8528   0.9776   0.9880   0.8929   0.8161  
0.40       0.9694   0.8536   0.9781   0.9878   0.8911   0.8191   <--
0.45       0.9690   0.8498   0.9792   0.9863   0.8773   0.8239  
0.50       0.9690   0.8498   0.9792   0.9863   0.8773   0.8239  
0.55       0.9695   0.8500   0.9812   0.9848   0.8639   0.8365  
0.60       0.9684   0.8418   0.9826   0.9823   0.8406   0.8429  
0.65       0.9694   0.8408   0.9872   0.9789   0.8087   0.8755  
0.70       0.9708   0.8424   0.9919   0.9760   0.7809   0.9144  
0.75       0.9733   0.8489   0.9979   0.9731   0.7513   0.9758  
0.80       0.9724   0.8418   0.9989   0.9713   0.7339   0.9869  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9694, F1=0.8536, Normal Recall=0.9781, Normal Precision=0.9878, Attack Recall=0.8911, Attack Precision=0.8191

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 80% : Attack 20%  (n=425,865, N=340,692, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9548   0.8943   0.9545   0.9886   0.9561   0.8401  
0.20       0.9596   0.9040   0.9621   0.9871   0.9499   0.8623   <--
0.25       0.9603   0.9025   0.9709   0.9794   0.9182   0.8873  
0.30       0.9607   0.9010   0.9771   0.9738   0.8948   0.9073  
0.35       0.9609   0.9014   0.9776   0.9736   0.8940   0.9088  
0.40       0.9609   0.9012   0.9781   0.9732   0.8921   0.9104  
0.45       0.9589   0.8952   0.9791   0.9698   0.8781   0.9130  
0.50       0.9589   0.8952   0.9791   0.9698   0.8781   0.9130  
0.55       0.9578   0.8912   0.9811   0.9666   0.8644   0.9198  
0.60       0.9542   0.8802   0.9825   0.9611   0.8411   0.9232  
0.65       0.9516   0.8699   0.9872   0.9539   0.8092   0.9405  
0.70       0.9499   0.8620   0.9919   0.9479   0.7821   0.9601  
0.75       0.9490   0.8553   0.9979   0.9418   0.7533   0.9892  
0.80       0.9461   0.8450   0.9989   0.9377   0.7347   0.9942  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9596, F1=0.9040, Normal Recall=0.9621, Normal Precision=0.9871, Attack Recall=0.9499, Attack Precision=0.8623

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 70% : Attack 30%  (n=283,910, N=198,737, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9547   0.9268   0.9541   0.9807   0.9561   0.8993  
0.20       0.9582   0.9317   0.9618   0.9782   0.9499   0.9142   <--
0.25       0.9550   0.9244   0.9707   0.9652   0.9182   0.9307  
0.30       0.9522   0.9183   0.9768   0.9559   0.8948   0.9430  
0.35       0.9523   0.9183   0.9772   0.9556   0.8940   0.9439  
0.40       0.9520   0.9178   0.9777   0.9548   0.8921   0.9449  
0.45       0.9486   0.9111   0.9788   0.9493   0.8781   0.9466  
0.50       0.9486   0.9111   0.9788   0.9493   0.8781   0.9466  
0.55       0.9460   0.9056   0.9809   0.9441   0.8644   0.9510  
0.60       0.9399   0.8936   0.9823   0.9352   0.8411   0.9531  
0.65       0.9336   0.8797   0.9869   0.9235   0.8092   0.9637  
0.70       0.9288   0.8683   0.9917   0.9139   0.7821   0.9759  
0.75       0.9246   0.8570   0.9980   0.9042   0.7533   0.9937  
0.80       0.9197   0.8459   0.9990   0.8978   0.7347   0.9968  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9582, F1=0.9317, Normal Recall=0.9618, Normal Precision=0.9782, Attack Recall=0.9499, Attack Precision=0.9142

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 60% : Attack 40%  (n=212,930, N=127,758, A=85,172)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9548   0.9442   0.9540   0.9702   0.9561   0.9326  
0.20       0.9571   0.9465   0.9618   0.9664   0.9499   0.9432   <--
0.25       0.9497   0.9359   0.9706   0.9468   0.9182   0.9542  
0.30       0.9439   0.9273   0.9766   0.9330   0.8948   0.9622  
0.35       0.9438   0.9272   0.9770   0.9326   0.8940   0.9629  
0.40       0.9434   0.9265   0.9775   0.9315   0.8921   0.9636  
0.45       0.9384   0.9194   0.9786   0.9233   0.8781   0.9648  
0.50       0.9384   0.9194   0.9786   0.9233   0.8781   0.9648  
0.55       0.9342   0.9131   0.9807   0.9156   0.8644   0.9676  
0.60       0.9257   0.9005   0.9821   0.9026   0.8411   0.9690  
0.65       0.9158   0.8849   0.9869   0.8858   0.8092   0.9763  
0.70       0.9079   0.8716   0.9917   0.8722   0.7821   0.9843  
0.75       0.9001   0.8578   0.9979   0.8585   0.7533   0.9959  
0.80       0.8933   0.8463   0.9990   0.8496   0.7347   0.9979  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9571, F1=0.9465, Normal Recall=0.9618, Normal Precision=0.9664, Attack Recall=0.9499, Attack Precision=0.9432

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 50% : Attack 50%  (n=170,346, N=85,173, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9549   0.9549   0.9537   0.9560   0.9561   0.9538  
0.20       0.9557   0.9555   0.9616   0.9505   0.9499   0.9611   <--
0.25       0.9445   0.9430   0.9708   0.9223   0.9182   0.9692  
0.30       0.9357   0.9330   0.9766   0.9028   0.8948   0.9745  
0.35       0.9355   0.9327   0.9770   0.9021   0.8940   0.9750  
0.40       0.9348   0.9319   0.9776   0.9006   0.8921   0.9755  
0.45       0.9284   0.9246   0.9786   0.8893   0.8781   0.9762  
0.50       0.9284   0.9246   0.9786   0.8893   0.8781   0.9762  
0.55       0.9225   0.9177   0.9806   0.8785   0.8644   0.9780  
0.60       0.9115   0.9048   0.9820   0.8607   0.8411   0.9790  
0.65       0.8980   0.8881   0.9868   0.8380   0.8092   0.9840  
0.70       0.8869   0.8737   0.9917   0.8198   0.7821   0.9895  
0.75       0.8756   0.8583   0.9979   0.8018   0.7533   0.9972  
0.80       0.8668   0.8466   0.9990   0.7902   0.7347   0.9986  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9557, F1=0.9555, Normal Recall=0.9616, Normal Precision=0.9505, Attack Recall=0.9499, Attack Precision=0.9611

```

