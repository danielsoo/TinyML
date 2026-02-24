# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e1_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-16 04:55:56 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9677 | 0.8724 | 0.7769 | 0.6812 | 0.5856 | 0.4905 | 0.3951 | 0.2991 | 0.2039 | 0.1083 | 0.0129 |
| QAT+Prune only | 0.8251 | 0.8432 | 0.8596 | 0.8771 | 0.8940 | 0.9095 | 0.9281 | 0.9443 | 0.9606 | 0.9777 | 0.9948 |
| QAT+PTQ | 0.8246 | 0.8428 | 0.8592 | 0.8768 | 0.8937 | 0.9091 | 0.9278 | 0.9439 | 0.9604 | 0.9776 | 0.9946 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8246 | 0.8428 | 0.8592 | 0.8768 | 0.8937 | 0.9091 | 0.9278 | 0.9439 | 0.9604 | 0.9776 | 0.9946 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.0199 | 0.0227 | 0.0238 | 0.0244 | 0.0248 | 0.0250 | 0.0252 | 0.0253 | 0.0255 | 0.0255 |
| QAT+Prune only | 0.0000 | 0.5593 | 0.7392 | 0.8293 | 0.8825 | 0.9166 | 0.9432 | 0.9615 | 0.9758 | 0.9877 | 0.9974 |
| QAT+PTQ | 0.0000 | 0.5585 | 0.7386 | 0.8289 | 0.8822 | 0.9163 | 0.9430 | 0.9613 | 0.9757 | 0.9876 | 0.9973 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5585 | 0.7386 | 0.8289 | 0.8822 | 0.9163 | 0.9430 | 0.9613 | 0.9757 | 0.9876 | 0.9973 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9677 | 0.9679 | 0.9679 | 0.9675 | 0.9673 | 0.9681 | 0.9683 | 0.9668 | 0.9679 | 0.9663 | 0.0000 |
| QAT+Prune only | 0.8251 | 0.8264 | 0.8258 | 0.8267 | 0.8269 | 0.8242 | 0.8282 | 0.8264 | 0.8239 | 0.8242 | 0.0000 |
| QAT+PTQ | 0.8246 | 0.8259 | 0.8253 | 0.8263 | 0.8265 | 0.8236 | 0.8276 | 0.8256 | 0.8232 | 0.8238 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8246 | 0.8259 | 0.8253 | 0.8263 | 0.8265 | 0.8236 | 0.8276 | 0.8256 | 0.8232 | 0.8238 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9677 | 0.0000 | 0.0000 | 0.0000 | 0.9677 | 1.0000 |
| 90 | 10 | 299,940 | 0.8724 | 0.0429 | 0.0130 | 0.0199 | 0.9679 | 0.8982 |
| 80 | 20 | 291,350 | 0.7769 | 0.0914 | 0.0129 | 0.0227 | 0.9679 | 0.7968 |
| 70 | 30 | 194,230 | 0.6812 | 0.1459 | 0.0129 | 0.0238 | 0.9675 | 0.6958 |
| 60 | 40 | 145,675 | 0.5856 | 0.2089 | 0.0129 | 0.0244 | 0.9673 | 0.5951 |
| 50 | 50 | 116,540 | 0.4905 | 0.2884 | 0.0129 | 0.0248 | 0.9681 | 0.4951 |
| 40 | 60 | 97,115 | 0.3951 | 0.3795 | 0.0129 | 0.0250 | 0.9683 | 0.3954 |
| 30 | 70 | 83,240 | 0.2991 | 0.4763 | 0.0129 | 0.0252 | 0.9668 | 0.2957 |
| 20 | 80 | 72,835 | 0.2039 | 0.6175 | 0.0129 | 0.0253 | 0.9679 | 0.1969 |
| 10 | 90 | 64,740 | 0.1083 | 0.7757 | 0.0129 | 0.0255 | 0.9663 | 0.0981 |
| 0 | 100 | 58,270 | 0.0129 | 1.0000 | 0.0129 | 0.0255 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8251 | 0.0000 | 0.0000 | 0.0000 | 0.8251 | 1.0000 |
| 90 | 10 | 299,940 | 0.8432 | 0.3890 | 0.9948 | 0.5593 | 0.8264 | 0.9993 |
| 80 | 20 | 291,350 | 0.8596 | 0.5881 | 0.9948 | 0.7392 | 0.8258 | 0.9984 |
| 70 | 30 | 194,230 | 0.8771 | 0.7110 | 0.9948 | 0.8293 | 0.8267 | 0.9973 |
| 60 | 40 | 145,675 | 0.8940 | 0.7930 | 0.9948 | 0.8825 | 0.8269 | 0.9958 |
| 50 | 50 | 116,540 | 0.9095 | 0.8498 | 0.9948 | 0.9166 | 0.8242 | 0.9937 |
| 40 | 60 | 97,115 | 0.9281 | 0.8967 | 0.9948 | 0.9432 | 0.8282 | 0.9906 |
| 30 | 70 | 83,240 | 0.9443 | 0.9304 | 0.9948 | 0.9615 | 0.8264 | 0.9855 |
| 20 | 80 | 72,835 | 0.9606 | 0.9576 | 0.9948 | 0.9758 | 0.8239 | 0.9753 |
| 10 | 90 | 64,740 | 0.9777 | 0.9807 | 0.9948 | 0.9877 | 0.8242 | 0.9461 |
| 0 | 100 | 58,270 | 0.9948 | 1.0000 | 0.9948 | 0.9974 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8246 | 0.0000 | 0.0000 | 0.0000 | 0.8246 | 1.0000 |
| 90 | 10 | 299,940 | 0.8428 | 0.3883 | 0.9946 | 0.5585 | 0.8259 | 0.9993 |
| 80 | 20 | 291,350 | 0.8592 | 0.5874 | 0.9946 | 0.7386 | 0.8253 | 0.9984 |
| 70 | 30 | 194,230 | 0.8768 | 0.7104 | 0.9946 | 0.8289 | 0.8263 | 0.9972 |
| 60 | 40 | 145,675 | 0.8937 | 0.7926 | 0.9946 | 0.8822 | 0.8265 | 0.9957 |
| 50 | 50 | 116,540 | 0.9091 | 0.8494 | 0.9946 | 0.9163 | 0.8236 | 0.9935 |
| 40 | 60 | 97,115 | 0.9278 | 0.8964 | 0.9946 | 0.9430 | 0.8276 | 0.9904 |
| 30 | 70 | 83,240 | 0.9439 | 0.9301 | 0.9946 | 0.9613 | 0.8256 | 0.9851 |
| 20 | 80 | 72,835 | 0.9604 | 0.9575 | 0.9946 | 0.9757 | 0.8232 | 0.9746 |
| 10 | 90 | 64,740 | 0.9776 | 0.9807 | 0.9946 | 0.9876 | 0.8238 | 0.9447 |
| 0 | 100 | 58,270 | 0.9946 | 1.0000 | 0.9946 | 0.9973 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8246 | 0.0000 | 0.0000 | 0.0000 | 0.8246 | 1.0000 |
| 90 | 10 | 299,940 | 0.8428 | 0.3883 | 0.9946 | 0.5585 | 0.8259 | 0.9993 |
| 80 | 20 | 291,350 | 0.8592 | 0.5874 | 0.9946 | 0.7386 | 0.8253 | 0.9984 |
| 70 | 30 | 194,230 | 0.8768 | 0.7104 | 0.9946 | 0.8289 | 0.8263 | 0.9972 |
| 60 | 40 | 145,675 | 0.8937 | 0.7926 | 0.9946 | 0.8822 | 0.8265 | 0.9957 |
| 50 | 50 | 116,540 | 0.9091 | 0.8494 | 0.9946 | 0.9163 | 0.8236 | 0.9935 |
| 40 | 60 | 97,115 | 0.9278 | 0.8964 | 0.9946 | 0.9430 | 0.8276 | 0.9904 |
| 30 | 70 | 83,240 | 0.9439 | 0.9301 | 0.9946 | 0.9613 | 0.8256 | 0.9851 |
| 20 | 80 | 72,835 | 0.9604 | 0.9575 | 0.9946 | 0.9757 | 0.8232 | 0.9746 |
| 10 | 90 | 64,740 | 0.9776 | 0.9807 | 0.9946 | 0.9876 | 0.8238 | 0.9447 |
| 0 | 100 | 58,270 | 0.9946 | 1.0000 | 0.9946 | 0.9973 | 0.0000 | 0.0000 |


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
0.15       0.8724   0.0207   0.9679   0.8983   0.0135   0.0445   <--
0.20       0.8724   0.0207   0.9679   0.8983   0.0135   0.0445  
0.25       0.8724   0.0207   0.9679   0.8983   0.0135   0.0445  
0.30       0.8724   0.0207   0.9679   0.8983   0.0135   0.0445  
0.35       0.8724   0.0207   0.9679   0.8983   0.0135   0.0445  
0.40       0.8724   0.0207   0.9679   0.8983   0.0135   0.0445  
0.45       0.8724   0.0207   0.9679   0.8983   0.0135   0.0445  
0.50       0.8724   0.0207   0.9679   0.8983   0.0135   0.0445  
0.55       0.8724   0.0207   0.9679   0.8983   0.0135   0.0445  
0.60       0.8724   0.0207   0.9679   0.8983   0.0135   0.0445  
0.65       0.8724   0.0207   0.9679   0.8983   0.0135   0.0445  
0.70       0.8724   0.0207   0.9679   0.8983   0.0135   0.0445  
0.75       0.8724   0.0207   0.9679   0.8983   0.0135   0.0445  
0.80       0.8724   0.0207   0.9679   0.8983   0.0135   0.0445  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8724, F1=0.0207, Normal Recall=0.9679, Normal Precision=0.8983, Attack Recall=0.0135, Attack Precision=0.0445

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
0.15       0.7767   0.0227   0.9676   0.7968   0.0129   0.0909   <--
0.20       0.7767   0.0227   0.9676   0.7968   0.0129   0.0909  
0.25       0.7767   0.0227   0.9676   0.7968   0.0129   0.0909  
0.30       0.7767   0.0227   0.9676   0.7968   0.0129   0.0909  
0.35       0.7767   0.0227   0.9676   0.7968   0.0129   0.0909  
0.40       0.7767   0.0227   0.9676   0.7968   0.0129   0.0909  
0.45       0.7767   0.0227   0.9676   0.7968   0.0129   0.0909  
0.50       0.7767   0.0227   0.9676   0.7968   0.0129   0.0909  
0.55       0.7767   0.0227   0.9676   0.7968   0.0129   0.0909  
0.60       0.7767   0.0227   0.9676   0.7968   0.0129   0.0909  
0.65       0.7767   0.0227   0.9676   0.7968   0.0129   0.0909  
0.70       0.7767   0.0227   0.9676   0.7968   0.0129   0.0909  
0.75       0.7767   0.0227   0.9676   0.7968   0.0129   0.0909  
0.80       0.7767   0.0227   0.9676   0.7968   0.0129   0.0909  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7767, F1=0.0227, Normal Recall=0.9676, Normal Precision=0.7968, Attack Recall=0.0129, Attack Precision=0.0909

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
0.15       0.6813   0.0238   0.9677   0.6958   0.0129   0.1466   <--
0.20       0.6813   0.0238   0.9677   0.6958   0.0129   0.1466  
0.25       0.6813   0.0238   0.9677   0.6958   0.0129   0.1466  
0.30       0.6813   0.0238   0.9677   0.6958   0.0129   0.1466  
0.35       0.6813   0.0238   0.9677   0.6958   0.0129   0.1466  
0.40       0.6813   0.0238   0.9677   0.6958   0.0129   0.1466  
0.45       0.6813   0.0238   0.9677   0.6958   0.0129   0.1466  
0.50       0.6813   0.0238   0.9677   0.6958   0.0129   0.1466  
0.55       0.6813   0.0238   0.9677   0.6958   0.0129   0.1466  
0.60       0.6813   0.0238   0.9677   0.6958   0.0129   0.1466  
0.65       0.6813   0.0238   0.9677   0.6958   0.0129   0.1466  
0.70       0.6813   0.0238   0.9677   0.6958   0.0129   0.1466  
0.75       0.6813   0.0238   0.9677   0.6958   0.0129   0.1466  
0.80       0.6813   0.0238   0.9677   0.6958   0.0129   0.1466  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6813, F1=0.0238, Normal Recall=0.9677, Normal Precision=0.6958, Attack Recall=0.0129, Attack Precision=0.1466

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
0.15       0.5857   0.0244   0.9676   0.5952   0.0129   0.2104   <--
0.20       0.5857   0.0244   0.9676   0.5952   0.0129   0.2104  
0.25       0.5857   0.0244   0.9676   0.5952   0.0129   0.2104  
0.30       0.5857   0.0244   0.9676   0.5952   0.0129   0.2104  
0.35       0.5857   0.0244   0.9676   0.5952   0.0129   0.2104  
0.40       0.5857   0.0244   0.9676   0.5952   0.0129   0.2104  
0.45       0.5857   0.0244   0.9676   0.5952   0.0129   0.2104  
0.50       0.5857   0.0244   0.9676   0.5952   0.0129   0.2104  
0.55       0.5857   0.0244   0.9676   0.5952   0.0129   0.2104  
0.60       0.5857   0.0244   0.9676   0.5952   0.0129   0.2104  
0.65       0.5857   0.0244   0.9676   0.5952   0.0129   0.2104  
0.70       0.5857   0.0244   0.9676   0.5952   0.0129   0.2104  
0.75       0.5857   0.0244   0.9676   0.5952   0.0129   0.2104  
0.80       0.5857   0.0244   0.9676   0.5952   0.0129   0.2104  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5857, F1=0.0244, Normal Recall=0.9676, Normal Precision=0.5952, Attack Recall=0.0129, Attack Precision=0.2104

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
0.15       0.4902   0.0248   0.9674   0.4950   0.0129   0.2841   <--
0.20       0.4902   0.0248   0.9674   0.4950   0.0129   0.2841  
0.25       0.4902   0.0248   0.9674   0.4950   0.0129   0.2841  
0.30       0.4902   0.0248   0.9674   0.4950   0.0129   0.2841  
0.35       0.4902   0.0248   0.9674   0.4950   0.0129   0.2841  
0.40       0.4902   0.0248   0.9674   0.4950   0.0129   0.2841  
0.45       0.4902   0.0248   0.9674   0.4950   0.0129   0.2841  
0.50       0.4902   0.0248   0.9674   0.4950   0.0129   0.2841  
0.55       0.4902   0.0248   0.9674   0.4950   0.0129   0.2841  
0.60       0.4902   0.0248   0.9674   0.4950   0.0129   0.2841  
0.65       0.4902   0.0248   0.9674   0.4950   0.0129   0.2841  
0.70       0.4902   0.0248   0.9674   0.4950   0.0129   0.2841  
0.75       0.4902   0.0248   0.9674   0.4950   0.0129   0.2841  
0.80       0.4902   0.0248   0.9674   0.4950   0.0129   0.2841  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4902, F1=0.0248, Normal Recall=0.9674, Normal Precision=0.4950, Attack Recall=0.0129, Attack Precision=0.2841

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
0.15       0.8433   0.5595   0.8264   0.9994   0.9953   0.3891   <--
0.20       0.8433   0.5595   0.8264   0.9994   0.9953   0.3891  
0.25       0.8433   0.5595   0.8264   0.9994   0.9953   0.3891  
0.30       0.8433   0.5595   0.8264   0.9994   0.9953   0.3891  
0.35       0.8433   0.5595   0.8264   0.9994   0.9953   0.3891  
0.40       0.8433   0.5595   0.8264   0.9994   0.9953   0.3891  
0.45       0.8433   0.5595   0.8264   0.9994   0.9953   0.3891  
0.50       0.8433   0.5595   0.8264   0.9994   0.9953   0.3891  
0.55       0.8433   0.5595   0.8264   0.9994   0.9953   0.3891  
0.60       0.8433   0.5595   0.8264   0.9994   0.9953   0.3891  
0.65       0.8433   0.5595   0.8264   0.9994   0.9953   0.3891  
0.70       0.8433   0.5595   0.8264   0.9994   0.9953   0.3891  
0.75       0.8433   0.5595   0.8264   0.9994   0.9953   0.3891  
0.80       0.8433   0.5595   0.8264   0.9994   0.9953   0.3891  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8433, F1=0.5595, Normal Recall=0.8264, Normal Precision=0.9994, Attack Recall=0.9953, Attack Precision=0.3891

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
0.15       0.8604   0.7403   0.8268   0.9984   0.9948   0.5895   <--
0.20       0.8604   0.7403   0.8268   0.9984   0.9948   0.5895  
0.25       0.8604   0.7403   0.8268   0.9984   0.9948   0.5895  
0.30       0.8604   0.7403   0.8268   0.9984   0.9948   0.5895  
0.35       0.8604   0.7403   0.8268   0.9984   0.9948   0.5895  
0.40       0.8604   0.7403   0.8268   0.9984   0.9948   0.5895  
0.45       0.8604   0.7403   0.8268   0.9984   0.9948   0.5895  
0.50       0.8604   0.7403   0.8268   0.9984   0.9948   0.5895  
0.55       0.8604   0.7403   0.8268   0.9984   0.9948   0.5895  
0.60       0.8604   0.7403   0.8268   0.9984   0.9948   0.5895  
0.65       0.8604   0.7403   0.8268   0.9984   0.9948   0.5895  
0.70       0.8604   0.7403   0.8268   0.9984   0.9948   0.5895  
0.75       0.8604   0.7403   0.8268   0.9984   0.9948   0.5895  
0.80       0.8604   0.7403   0.8268   0.9984   0.9948   0.5895  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8604, F1=0.7403, Normal Recall=0.8268, Normal Precision=0.9984, Attack Recall=0.9948, Attack Precision=0.5895

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
0.15       0.8768   0.8289   0.8262   0.9973   0.9948   0.7104   <--
0.20       0.8768   0.8289   0.8262   0.9973   0.9948   0.7104  
0.25       0.8768   0.8289   0.8262   0.9973   0.9948   0.7104  
0.30       0.8768   0.8289   0.8262   0.9973   0.9948   0.7104  
0.35       0.8768   0.8289   0.8262   0.9973   0.9948   0.7104  
0.40       0.8768   0.8289   0.8262   0.9973   0.9948   0.7104  
0.45       0.8768   0.8289   0.8262   0.9973   0.9948   0.7104  
0.50       0.8768   0.8289   0.8262   0.9973   0.9948   0.7104  
0.55       0.8768   0.8289   0.8262   0.9973   0.9948   0.7104  
0.60       0.8768   0.8289   0.8262   0.9973   0.9948   0.7104  
0.65       0.8768   0.8289   0.8262   0.9973   0.9948   0.7104  
0.70       0.8768   0.8289   0.8262   0.9973   0.9948   0.7104  
0.75       0.8768   0.8289   0.8262   0.9973   0.9948   0.7104  
0.80       0.8768   0.8289   0.8262   0.9973   0.9948   0.7104  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8768, F1=0.8289, Normal Recall=0.8262, Normal Precision=0.9973, Attack Recall=0.9948, Attack Precision=0.7104

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
0.15       0.8931   0.8816   0.8253   0.9958   0.9948   0.7915   <--
0.20       0.8931   0.8816   0.8253   0.9958   0.9948   0.7915  
0.25       0.8931   0.8816   0.8253   0.9958   0.9948   0.7915  
0.30       0.8931   0.8816   0.8253   0.9958   0.9948   0.7915  
0.35       0.8931   0.8816   0.8253   0.9958   0.9948   0.7915  
0.40       0.8931   0.8816   0.8253   0.9958   0.9948   0.7915  
0.45       0.8931   0.8816   0.8253   0.9958   0.9948   0.7915  
0.50       0.8931   0.8816   0.8253   0.9958   0.9948   0.7915  
0.55       0.8931   0.8816   0.8253   0.9958   0.9948   0.7915  
0.60       0.8931   0.8816   0.8253   0.9958   0.9948   0.7915  
0.65       0.8931   0.8816   0.8253   0.9958   0.9948   0.7915  
0.70       0.8931   0.8816   0.8253   0.9958   0.9948   0.7915  
0.75       0.8931   0.8816   0.8253   0.9958   0.9948   0.7915  
0.80       0.8931   0.8816   0.8253   0.9958   0.9948   0.7915  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8931, F1=0.8816, Normal Recall=0.8253, Normal Precision=0.9958, Attack Recall=0.9948, Attack Precision=0.7915

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
0.15       0.9092   0.9163   0.8236   0.9937   0.9948   0.8494   <--
0.20       0.9092   0.9163   0.8236   0.9937   0.9948   0.8494  
0.25       0.9092   0.9163   0.8236   0.9937   0.9948   0.8494  
0.30       0.9092   0.9163   0.8236   0.9937   0.9948   0.8494  
0.35       0.9092   0.9163   0.8236   0.9937   0.9948   0.8494  
0.40       0.9092   0.9163   0.8236   0.9937   0.9948   0.8494  
0.45       0.9092   0.9163   0.8236   0.9937   0.9948   0.8494  
0.50       0.9092   0.9163   0.8236   0.9937   0.9948   0.8494  
0.55       0.9092   0.9163   0.8236   0.9937   0.9948   0.8494  
0.60       0.9092   0.9163   0.8236   0.9937   0.9948   0.8494  
0.65       0.9092   0.9163   0.8236   0.9937   0.9948   0.8494  
0.70       0.9092   0.9163   0.8236   0.9937   0.9948   0.8494  
0.75       0.9092   0.9163   0.8236   0.9937   0.9948   0.8494  
0.80       0.9092   0.9163   0.8236   0.9937   0.9948   0.8494  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9092, F1=0.9163, Normal Recall=0.8236, Normal Precision=0.9937, Attack Recall=0.9948, Attack Precision=0.8494

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
0.15       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884   <--
0.20       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.25       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.30       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.35       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.40       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.45       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.50       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.55       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.60       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.65       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.70       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.75       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.80       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8428, F1=0.5587, Normal Recall=0.8259, Normal Precision=0.9993, Attack Recall=0.9951, Attack Precision=0.3884

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
0.15       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888   <--
0.20       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.25       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.30       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.35       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.40       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.45       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.50       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.55       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.60       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.65       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.70       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.75       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.80       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8600, F1=0.7397, Normal Recall=0.8263, Normal Precision=0.9984, Attack Recall=0.9946, Attack Precision=0.5888

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
0.15       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099   <--
0.20       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.25       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.30       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.35       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.40       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.45       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.50       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.55       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.60       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.65       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.70       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.75       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.80       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8765, F1=0.8285, Normal Recall=0.8258, Normal Precision=0.9972, Attack Recall=0.9946, Attack Precision=0.7099

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
0.15       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911   <--
0.20       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.25       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.30       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.35       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.40       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.45       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.50       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.55       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.60       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.65       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.70       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.75       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.80       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8928, F1=0.8812, Normal Recall=0.8249, Normal Precision=0.9957, Attack Recall=0.9946, Attack Precision=0.7911

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
0.15       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491   <--
0.20       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.25       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.30       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.35       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.40       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.45       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.50       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.55       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.60       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.65       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.70       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.75       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.80       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9089, F1=0.9161, Normal Recall=0.8232, Normal Precision=0.9935, Attack Recall=0.9946, Attack Precision=0.8491

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
0.15       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884   <--
0.20       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.25       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.30       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.35       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.40       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.45       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.50       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.55       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.60       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.65       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.70       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.75       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
0.80       0.8428   0.5587   0.8259   0.9993   0.9951   0.3884  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8428, F1=0.5587, Normal Recall=0.8259, Normal Precision=0.9993, Attack Recall=0.9951, Attack Precision=0.3884

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
0.15       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888   <--
0.20       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.25       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.30       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.35       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.40       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.45       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.50       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.55       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.60       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.65       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.70       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.75       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
0.80       0.8600   0.7397   0.8263   0.9984   0.9946   0.5888  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8600, F1=0.7397, Normal Recall=0.8263, Normal Precision=0.9984, Attack Recall=0.9946, Attack Precision=0.5888

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
0.15       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099   <--
0.20       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.25       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.30       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.35       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.40       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.45       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.50       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.55       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.60       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.65       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.70       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.75       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
0.80       0.8765   0.8285   0.8258   0.9972   0.9946   0.7099  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8765, F1=0.8285, Normal Recall=0.8258, Normal Precision=0.9972, Attack Recall=0.9946, Attack Precision=0.7099

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
0.15       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911   <--
0.20       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.25       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.30       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.35       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.40       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.45       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.50       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.55       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.60       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.65       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.70       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.75       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
0.80       0.8928   0.8812   0.8249   0.9957   0.9946   0.7911  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8928, F1=0.8812, Normal Recall=0.8249, Normal Precision=0.9957, Attack Recall=0.9946, Attack Precision=0.7911

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
0.15       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491   <--
0.20       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.25       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.30       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.35       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.40       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.45       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.50       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.55       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.60       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.65       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.70       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.75       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
0.80       0.9089   0.9161   0.8232   0.9935   0.9946   0.8491  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9089, F1=0.9161, Normal Recall=0.8232, Normal Precision=0.9935, Attack Recall=0.9946, Attack Precision=0.8491

```

