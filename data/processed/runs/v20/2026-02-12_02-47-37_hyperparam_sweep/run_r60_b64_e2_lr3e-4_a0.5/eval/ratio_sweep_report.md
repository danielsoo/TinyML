# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e2_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-12 16:07:08 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3341 | 0.3985 | 0.4639 | 0.5292 | 0.5941 | 0.6596 | 0.7256 | 0.7925 | 0.8567 | 0.9207 | 0.9860 |
| QAT+Prune only | 0.9409 | 0.9423 | 0.9432 | 0.9450 | 0.9462 | 0.9470 | 0.9489 | 0.9500 | 0.9510 | 0.9522 | 0.9535 |
| QAT+PTQ | 0.9404 | 0.9419 | 0.9428 | 0.9446 | 0.9460 | 0.9466 | 0.9487 | 0.9498 | 0.9509 | 0.9522 | 0.9535 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9404 | 0.9419 | 0.9428 | 0.9446 | 0.9460 | 0.9466 | 0.9487 | 0.9498 | 0.9509 | 0.9522 | 0.9535 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2467 | 0.4239 | 0.5568 | 0.6602 | 0.7433 | 0.8118 | 0.8693 | 0.9167 | 0.9572 | 0.9930 |
| QAT+Prune only | 0.0000 | 0.7677 | 0.8703 | 0.9122 | 0.9342 | 0.9473 | 0.9572 | 0.9639 | 0.9689 | 0.9729 | 0.9762 |
| QAT+PTQ | 0.0000 | 0.7664 | 0.8696 | 0.9117 | 0.9338 | 0.9470 | 0.9571 | 0.9637 | 0.9688 | 0.9729 | 0.9762 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.7664 | 0.8696 | 0.9117 | 0.9338 | 0.9470 | 0.9571 | 0.9637 | 0.9688 | 0.9729 | 0.9762 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3341 | 0.3333 | 0.3334 | 0.3334 | 0.3328 | 0.3331 | 0.3350 | 0.3410 | 0.3393 | 0.3324 | 0.0000 |
| QAT+Prune only | 0.9409 | 0.9410 | 0.9406 | 0.9413 | 0.9414 | 0.9404 | 0.9418 | 0.9417 | 0.9408 | 0.9399 | 0.0000 |
| QAT+PTQ | 0.9404 | 0.9406 | 0.9401 | 0.9408 | 0.9409 | 0.9398 | 0.9415 | 0.9411 | 0.9403 | 0.9401 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9404 | 0.9406 | 0.9401 | 0.9408 | 0.9409 | 0.9398 | 0.9415 | 0.9411 | 0.9403 | 0.9401 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3341 | 0.0000 | 0.0000 | 0.0000 | 0.3341 | 1.0000 |
| 90 | 10 | 299,940 | 0.3985 | 0.1410 | 0.9851 | 0.2467 | 0.3333 | 0.9951 |
| 80 | 20 | 291,350 | 0.4639 | 0.2699 | 0.9860 | 0.4239 | 0.3334 | 0.9896 |
| 70 | 30 | 194,230 | 0.5292 | 0.3880 | 0.9860 | 0.5568 | 0.3334 | 0.9823 |
| 60 | 40 | 145,675 | 0.5941 | 0.4963 | 0.9860 | 0.6602 | 0.3328 | 0.9727 |
| 50 | 50 | 116,540 | 0.6596 | 0.5965 | 0.9860 | 0.7433 | 0.3331 | 0.9597 |
| 40 | 60 | 97,115 | 0.7256 | 0.6898 | 0.9860 | 0.8118 | 0.3350 | 0.9411 |
| 30 | 70 | 83,240 | 0.7925 | 0.7773 | 0.9860 | 0.8693 | 0.3410 | 0.9126 |
| 20 | 80 | 72,835 | 0.8567 | 0.8565 | 0.9860 | 0.9167 | 0.3393 | 0.8584 |
| 10 | 90 | 64,740 | 0.9207 | 0.9300 | 0.9860 | 0.9572 | 0.3324 | 0.7253 |
| 0 | 100 | 58,270 | 0.9860 | 1.0000 | 0.9860 | 0.9930 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9409 | 0.0000 | 0.0000 | 0.0000 | 0.9409 | 1.0000 |
| 90 | 10 | 299,940 | 0.9423 | 0.6425 | 0.9535 | 0.7677 | 0.9410 | 0.9945 |
| 80 | 20 | 291,350 | 0.9432 | 0.8005 | 0.9535 | 0.8703 | 0.9406 | 0.9878 |
| 70 | 30 | 194,230 | 0.9450 | 0.8744 | 0.9535 | 0.9122 | 0.9413 | 0.9793 |
| 60 | 40 | 145,675 | 0.9462 | 0.9156 | 0.9535 | 0.9342 | 0.9414 | 0.9681 |
| 50 | 50 | 116,540 | 0.9470 | 0.9412 | 0.9535 | 0.9473 | 0.9404 | 0.9529 |
| 40 | 60 | 97,115 | 0.9489 | 0.9609 | 0.9535 | 0.9572 | 0.9418 | 0.9311 |
| 30 | 70 | 83,240 | 0.9500 | 0.9745 | 0.9535 | 0.9639 | 0.9417 | 0.8967 |
| 20 | 80 | 72,835 | 0.9510 | 0.9847 | 0.9535 | 0.9689 | 0.9408 | 0.8350 |
| 10 | 90 | 64,740 | 0.9522 | 0.9930 | 0.9535 | 0.9729 | 0.9399 | 0.6921 |
| 0 | 100 | 58,270 | 0.9535 | 1.0000 | 0.9535 | 0.9762 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9404 | 0.0000 | 0.0000 | 0.0000 | 0.9404 | 1.0000 |
| 90 | 10 | 299,940 | 0.9419 | 0.6407 | 0.9534 | 0.7664 | 0.9406 | 0.9945 |
| 80 | 20 | 291,350 | 0.9428 | 0.7993 | 0.9535 | 0.8696 | 0.9401 | 0.9878 |
| 70 | 30 | 194,230 | 0.9446 | 0.8735 | 0.9535 | 0.9117 | 0.9408 | 0.9793 |
| 60 | 40 | 145,675 | 0.9460 | 0.9150 | 0.9535 | 0.9338 | 0.9409 | 0.9681 |
| 50 | 50 | 116,540 | 0.9466 | 0.9406 | 0.9535 | 0.9470 | 0.9398 | 0.9528 |
| 40 | 60 | 97,115 | 0.9487 | 0.9607 | 0.9535 | 0.9571 | 0.9415 | 0.9310 |
| 30 | 70 | 83,240 | 0.9498 | 0.9742 | 0.9535 | 0.9637 | 0.9411 | 0.8966 |
| 20 | 80 | 72,835 | 0.9509 | 0.9846 | 0.9535 | 0.9688 | 0.9403 | 0.8348 |
| 10 | 90 | 64,740 | 0.9522 | 0.9931 | 0.9535 | 0.9729 | 0.9401 | 0.6920 |
| 0 | 100 | 58,270 | 0.9535 | 1.0000 | 0.9535 | 0.9762 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9404 | 0.0000 | 0.0000 | 0.0000 | 0.9404 | 1.0000 |
| 90 | 10 | 299,940 | 0.9419 | 0.6407 | 0.9534 | 0.7664 | 0.9406 | 0.9945 |
| 80 | 20 | 291,350 | 0.9428 | 0.7993 | 0.9535 | 0.8696 | 0.9401 | 0.9878 |
| 70 | 30 | 194,230 | 0.9446 | 0.8735 | 0.9535 | 0.9117 | 0.9408 | 0.9793 |
| 60 | 40 | 145,675 | 0.9460 | 0.9150 | 0.9535 | 0.9338 | 0.9409 | 0.9681 |
| 50 | 50 | 116,540 | 0.9466 | 0.9406 | 0.9535 | 0.9470 | 0.9398 | 0.9528 |
| 40 | 60 | 97,115 | 0.9487 | 0.9607 | 0.9535 | 0.9571 | 0.9415 | 0.9310 |
| 30 | 70 | 83,240 | 0.9498 | 0.9742 | 0.9535 | 0.9637 | 0.9411 | 0.8966 |
| 20 | 80 | 72,835 | 0.9509 | 0.9846 | 0.9535 | 0.9688 | 0.9403 | 0.8348 |
| 10 | 90 | 64,740 | 0.9522 | 0.9931 | 0.9535 | 0.9729 | 0.9401 | 0.6920 |
| 0 | 100 | 58,270 | 0.9535 | 1.0000 | 0.9535 | 0.9762 | 0.0000 | 0.0000 |


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
0.15       0.3986   0.2470   0.3333   0.9954   0.9862   0.1412   <--
0.20       0.3986   0.2470   0.3333   0.9954   0.9862   0.1412  
0.25       0.3986   0.2470   0.3333   0.9954   0.9862   0.1412  
0.30       0.3986   0.2470   0.3333   0.9954   0.9862   0.1412  
0.35       0.3986   0.2470   0.3333   0.9954   0.9862   0.1412  
0.40       0.3986   0.2470   0.3333   0.9954   0.9862   0.1412  
0.45       0.3986   0.2470   0.3333   0.9954   0.9862   0.1412  
0.50       0.3986   0.2470   0.3333   0.9954   0.9862   0.1412  
0.55       0.3986   0.2470   0.3333   0.9954   0.9862   0.1412  
0.60       0.3986   0.2470   0.3333   0.9954   0.9862   0.1412  
0.65       0.3986   0.2470   0.3333   0.9954   0.9862   0.1412  
0.70       0.3986   0.2470   0.3333   0.9954   0.9862   0.1412  
0.75       0.3986   0.2470   0.3333   0.9954   0.9862   0.1412  
0.80       0.3986   0.2470   0.3333   0.9954   0.9862   0.1412  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3986, F1=0.2470, Normal Recall=0.3333, Normal Precision=0.9954, Attack Recall=0.9862, Attack Precision=0.1412

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
0.15       0.4636   0.4237   0.3330   0.9896   0.9860   0.2698   <--
0.20       0.4636   0.4237   0.3330   0.9896   0.9860   0.2698  
0.25       0.4636   0.4237   0.3330   0.9896   0.9860   0.2698  
0.30       0.4636   0.4237   0.3330   0.9896   0.9860   0.2698  
0.35       0.4636   0.4237   0.3330   0.9896   0.9860   0.2698  
0.40       0.4636   0.4237   0.3330   0.9896   0.9860   0.2698  
0.45       0.4636   0.4237   0.3330   0.9896   0.9860   0.2698  
0.50       0.4636   0.4237   0.3330   0.9896   0.9860   0.2698  
0.55       0.4636   0.4237   0.3330   0.9896   0.9860   0.2698  
0.60       0.4636   0.4237   0.3330   0.9896   0.9860   0.2698  
0.65       0.4636   0.4237   0.3330   0.9896   0.9860   0.2698  
0.70       0.4636   0.4237   0.3330   0.9896   0.9860   0.2698  
0.75       0.4636   0.4237   0.3330   0.9896   0.9860   0.2698  
0.80       0.4636   0.4237   0.3330   0.9896   0.9860   0.2698  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4636, F1=0.4237, Normal Recall=0.3330, Normal Precision=0.9896, Attack Recall=0.9860, Attack Precision=0.2698

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
0.15       0.5298   0.5572   0.3343   0.9824   0.9860   0.3883   <--
0.20       0.5298   0.5572   0.3343   0.9824   0.9860   0.3883  
0.25       0.5298   0.5572   0.3343   0.9824   0.9860   0.3883  
0.30       0.5298   0.5572   0.3343   0.9824   0.9860   0.3883  
0.35       0.5298   0.5572   0.3343   0.9824   0.9860   0.3883  
0.40       0.5298   0.5572   0.3343   0.9824   0.9860   0.3883  
0.45       0.5298   0.5572   0.3343   0.9824   0.9860   0.3883  
0.50       0.5298   0.5572   0.3343   0.9824   0.9860   0.3883  
0.55       0.5298   0.5572   0.3343   0.9824   0.9860   0.3883  
0.60       0.5298   0.5572   0.3343   0.9824   0.9860   0.3883  
0.65       0.5298   0.5572   0.3343   0.9824   0.9860   0.3883  
0.70       0.5298   0.5572   0.3343   0.9824   0.9860   0.3883  
0.75       0.5298   0.5572   0.3343   0.9824   0.9860   0.3883  
0.80       0.5298   0.5572   0.3343   0.9824   0.9860   0.3883  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5298, F1=0.5572, Normal Recall=0.3343, Normal Precision=0.9824, Attack Recall=0.9860, Attack Precision=0.3883

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
0.15       0.5951   0.6608   0.3345   0.9729   0.9860   0.4969   <--
0.20       0.5951   0.6608   0.3345   0.9729   0.9860   0.4969  
0.25       0.5951   0.6608   0.3345   0.9729   0.9860   0.4969  
0.30       0.5951   0.6608   0.3345   0.9729   0.9860   0.4969  
0.35       0.5951   0.6608   0.3345   0.9729   0.9860   0.4969  
0.40       0.5951   0.6608   0.3345   0.9729   0.9860   0.4969  
0.45       0.5951   0.6608   0.3345   0.9729   0.9860   0.4969  
0.50       0.5951   0.6608   0.3345   0.9729   0.9860   0.4969  
0.55       0.5951   0.6608   0.3345   0.9729   0.9860   0.4969  
0.60       0.5951   0.6608   0.3345   0.9729   0.9860   0.4969  
0.65       0.5951   0.6608   0.3345   0.9729   0.9860   0.4969  
0.70       0.5951   0.6608   0.3345   0.9729   0.9860   0.4969  
0.75       0.5951   0.6608   0.3345   0.9729   0.9860   0.4969  
0.80       0.5951   0.6608   0.3345   0.9729   0.9860   0.4969  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5951, F1=0.6608, Normal Recall=0.3345, Normal Precision=0.9729, Attack Recall=0.9860, Attack Precision=0.4969

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
0.15       0.6600   0.7436   0.3341   0.9598   0.9860   0.5969   <--
0.20       0.6600   0.7436   0.3341   0.9598   0.9860   0.5969  
0.25       0.6600   0.7436   0.3341   0.9598   0.9860   0.5969  
0.30       0.6600   0.7436   0.3341   0.9598   0.9860   0.5969  
0.35       0.6600   0.7436   0.3341   0.9598   0.9860   0.5969  
0.40       0.6600   0.7436   0.3341   0.9598   0.9860   0.5969  
0.45       0.6600   0.7436   0.3341   0.9598   0.9860   0.5969  
0.50       0.6600   0.7436   0.3341   0.9598   0.9860   0.5969  
0.55       0.6600   0.7436   0.3341   0.9598   0.9860   0.5969  
0.60       0.6600   0.7436   0.3341   0.9598   0.9860   0.5969  
0.65       0.6600   0.7436   0.3341   0.9598   0.9860   0.5969  
0.70       0.6600   0.7436   0.3341   0.9598   0.9860   0.5969  
0.75       0.6600   0.7436   0.3341   0.9598   0.9860   0.5969  
0.80       0.6600   0.7436   0.3341   0.9598   0.9860   0.5969  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6600, F1=0.7436, Normal Recall=0.3341, Normal Precision=0.9598, Attack Recall=0.9860, Attack Precision=0.5969

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
0.15       0.9425   0.7685   0.9410   0.9947   0.9551   0.6429   <--
0.20       0.9425   0.7685   0.9410   0.9947   0.9551   0.6429  
0.25       0.9425   0.7685   0.9410   0.9947   0.9551   0.6429  
0.30       0.9425   0.7685   0.9410   0.9947   0.9551   0.6429  
0.35       0.9425   0.7685   0.9410   0.9947   0.9551   0.6429  
0.40       0.9425   0.7685   0.9410   0.9947   0.9551   0.6429  
0.45       0.9425   0.7685   0.9410   0.9947   0.9551   0.6429  
0.50       0.9425   0.7685   0.9410   0.9947   0.9551   0.6429  
0.55       0.9425   0.7685   0.9410   0.9947   0.9551   0.6429  
0.60       0.9425   0.7685   0.9410   0.9947   0.9551   0.6429  
0.65       0.9425   0.7685   0.9410   0.9947   0.9551   0.6429  
0.70       0.9425   0.7685   0.9410   0.9947   0.9551   0.6429  
0.75       0.9425   0.7685   0.9410   0.9947   0.9551   0.6429  
0.80       0.9425   0.7685   0.9410   0.9947   0.9551   0.6429  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9425, F1=0.7685, Normal Recall=0.9410, Normal Precision=0.9947, Attack Recall=0.9551, Attack Precision=0.6429

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
0.15       0.9438   0.8716   0.9414   0.9878   0.9535   0.8026   <--
0.20       0.9438   0.8716   0.9414   0.9878   0.9535   0.8026  
0.25       0.9438   0.8716   0.9414   0.9878   0.9535   0.8026  
0.30       0.9438   0.8716   0.9414   0.9878   0.9535   0.8026  
0.35       0.9438   0.8716   0.9414   0.9878   0.9535   0.8026  
0.40       0.9438   0.8716   0.9414   0.9878   0.9535   0.8026  
0.45       0.9438   0.8716   0.9414   0.9878   0.9535   0.8026  
0.50       0.9438   0.8716   0.9414   0.9878   0.9535   0.8026  
0.55       0.9438   0.8716   0.9414   0.9878   0.9535   0.8026  
0.60       0.9438   0.8716   0.9414   0.9878   0.9535   0.8026  
0.65       0.9438   0.8716   0.9414   0.9878   0.9535   0.8026  
0.70       0.9438   0.8716   0.9414   0.9878   0.9535   0.8026  
0.75       0.9438   0.8716   0.9414   0.9878   0.9535   0.8026  
0.80       0.9438   0.8716   0.9414   0.9878   0.9535   0.8026  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9438, F1=0.8716, Normal Recall=0.9414, Normal Precision=0.9878, Attack Recall=0.9535, Attack Precision=0.8026

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
0.15       0.9451   0.9125   0.9415   0.9793   0.9535   0.8748   <--
0.20       0.9451   0.9125   0.9415   0.9793   0.9535   0.8748  
0.25       0.9451   0.9125   0.9415   0.9793   0.9535   0.8748  
0.30       0.9451   0.9125   0.9415   0.9793   0.9535   0.8748  
0.35       0.9451   0.9125   0.9415   0.9793   0.9535   0.8748  
0.40       0.9451   0.9125   0.9415   0.9793   0.9535   0.8748  
0.45       0.9451   0.9125   0.9415   0.9793   0.9535   0.8748  
0.50       0.9451   0.9125   0.9415   0.9793   0.9535   0.8748  
0.55       0.9451   0.9125   0.9415   0.9793   0.9535   0.8748  
0.60       0.9451   0.9125   0.9415   0.9793   0.9535   0.8748  
0.65       0.9451   0.9125   0.9415   0.9793   0.9535   0.8748  
0.70       0.9451   0.9125   0.9415   0.9793   0.9535   0.8748  
0.75       0.9451   0.9125   0.9415   0.9793   0.9535   0.8748  
0.80       0.9451   0.9125   0.9415   0.9793   0.9535   0.8748  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9451, F1=0.9125, Normal Recall=0.9415, Normal Precision=0.9793, Attack Recall=0.9535, Attack Precision=0.8748

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
0.15       0.9462   0.9341   0.9413   0.9681   0.9535   0.9154   <--
0.20       0.9462   0.9341   0.9413   0.9681   0.9535   0.9154  
0.25       0.9462   0.9341   0.9413   0.9681   0.9535   0.9154  
0.30       0.9462   0.9341   0.9413   0.9681   0.9535   0.9154  
0.35       0.9462   0.9341   0.9413   0.9681   0.9535   0.9154  
0.40       0.9462   0.9341   0.9413   0.9681   0.9535   0.9154  
0.45       0.9462   0.9341   0.9413   0.9681   0.9535   0.9154  
0.50       0.9462   0.9341   0.9413   0.9681   0.9535   0.9154  
0.55       0.9462   0.9341   0.9413   0.9681   0.9535   0.9154  
0.60       0.9462   0.9341   0.9413   0.9681   0.9535   0.9154  
0.65       0.9462   0.9341   0.9413   0.9681   0.9535   0.9154  
0.70       0.9462   0.9341   0.9413   0.9681   0.9535   0.9154  
0.75       0.9462   0.9341   0.9413   0.9681   0.9535   0.9154  
0.80       0.9462   0.9341   0.9413   0.9681   0.9535   0.9154  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9462, F1=0.9341, Normal Recall=0.9413, Normal Precision=0.9681, Attack Recall=0.9535, Attack Precision=0.9154

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
0.15       0.9474   0.9477   0.9413   0.9530   0.9535   0.9420   <--
0.20       0.9474   0.9477   0.9413   0.9530   0.9535   0.9420  
0.25       0.9474   0.9477   0.9413   0.9530   0.9535   0.9420  
0.30       0.9474   0.9477   0.9413   0.9530   0.9535   0.9420  
0.35       0.9474   0.9477   0.9413   0.9530   0.9535   0.9420  
0.40       0.9474   0.9477   0.9413   0.9530   0.9535   0.9420  
0.45       0.9474   0.9477   0.9413   0.9530   0.9535   0.9420  
0.50       0.9474   0.9477   0.9413   0.9530   0.9535   0.9420  
0.55       0.9474   0.9477   0.9413   0.9530   0.9535   0.9420  
0.60       0.9474   0.9477   0.9413   0.9530   0.9535   0.9420  
0.65       0.9474   0.9477   0.9413   0.9530   0.9535   0.9420  
0.70       0.9474   0.9477   0.9413   0.9530   0.9535   0.9420  
0.75       0.9474   0.9477   0.9413   0.9530   0.9535   0.9420  
0.80       0.9474   0.9477   0.9413   0.9530   0.9535   0.9420  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9474, F1=0.9477, Normal Recall=0.9413, Normal Precision=0.9530, Attack Recall=0.9535, Attack Precision=0.9420

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
0.15       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411   <--
0.20       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.25       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.30       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.35       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.40       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.45       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.50       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.55       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.60       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.65       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.70       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.75       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.80       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9420, F1=0.7672, Normal Recall=0.9406, Normal Precision=0.9947, Attack Recall=0.9551, Attack Precision=0.6411

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
0.15       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013   <--
0.20       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.25       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.30       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.35       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.40       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.45       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.50       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.55       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.60       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.65       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.70       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.75       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.80       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9434, F1=0.8708, Normal Recall=0.9409, Normal Precision=0.9878, Attack Recall=0.9535, Attack Precision=0.8013

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
0.15       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738   <--
0.20       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.25       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.30       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.35       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.40       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.45       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.50       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.55       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.60       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.65       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.70       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.75       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.80       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9447, F1=0.9119, Normal Recall=0.9410, Normal Precision=0.9793, Attack Recall=0.9535, Attack Precision=0.8738

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
0.15       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147   <--
0.20       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.25       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.30       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.35       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.40       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.45       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.50       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.55       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.60       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.65       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.70       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.75       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.80       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9458, F1=0.9337, Normal Recall=0.9407, Normal Precision=0.9681, Attack Recall=0.9535, Attack Precision=0.9147

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
0.15       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415   <--
0.20       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.25       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.30       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.35       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.40       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.45       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.50       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.55       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.60       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.65       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.70       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.75       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.80       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9471, F1=0.9475, Normal Recall=0.9407, Normal Precision=0.9529, Attack Recall=0.9535, Attack Precision=0.9415

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
0.15       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411   <--
0.20       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.25       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.30       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.35       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.40       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.45       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.50       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.55       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.60       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.65       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.70       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.75       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
0.80       0.9420   0.7672   0.9406   0.9947   0.9551   0.6411  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9420, F1=0.7672, Normal Recall=0.9406, Normal Precision=0.9947, Attack Recall=0.9551, Attack Precision=0.6411

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
0.15       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013   <--
0.20       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.25       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.30       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.35       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.40       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.45       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.50       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.55       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.60       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.65       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.70       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.75       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
0.80       0.9434   0.8708   0.9409   0.9878   0.9535   0.8013  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9434, F1=0.8708, Normal Recall=0.9409, Normal Precision=0.9878, Attack Recall=0.9535, Attack Precision=0.8013

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
0.15       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738   <--
0.20       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.25       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.30       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.35       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.40       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.45       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.50       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.55       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.60       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.65       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.70       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.75       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
0.80       0.9447   0.9119   0.9410   0.9793   0.9535   0.8738  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9447, F1=0.9119, Normal Recall=0.9410, Normal Precision=0.9793, Attack Recall=0.9535, Attack Precision=0.8738

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
0.15       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147   <--
0.20       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.25       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.30       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.35       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.40       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.45       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.50       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.55       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.60       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.65       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.70       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.75       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
0.80       0.9458   0.9337   0.9407   0.9681   0.9535   0.9147  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9458, F1=0.9337, Normal Recall=0.9407, Normal Precision=0.9681, Attack Recall=0.9535, Attack Precision=0.9147

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
0.15       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415   <--
0.20       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.25       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.30       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.35       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.40       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.45       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.50       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.55       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.60       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.65       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.70       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.75       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
0.80       0.9471   0.9475   0.9407   0.9529   0.9535   0.9415  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9471, F1=0.9475, Normal Recall=0.9407, Normal Precision=0.9529, Attack Recall=0.9535, Attack Precision=0.9415

```

