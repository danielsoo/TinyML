# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e1_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-14 16:08:28 |

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
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4190 | 0.4751 | 0.5332 | 0.5911 | 0.6479 | 0.7064 | 0.7640 | 0.8227 | 0.8802 | 0.9378 | 0.9957 |
| QAT+Prune only | 0.8113 | 0.8235 | 0.8348 | 0.8463 | 0.8587 | 0.8684 | 0.8822 | 0.8917 | 0.9044 | 0.9148 | 0.9275 |
| QAT+PTQ | 0.8119 | 0.8236 | 0.8349 | 0.8461 | 0.8587 | 0.8683 | 0.8818 | 0.8911 | 0.9041 | 0.9142 | 0.9268 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8119 | 0.8236 | 0.8349 | 0.8461 | 0.8587 | 0.8683 | 0.8818 | 0.8911 | 0.9041 | 0.9142 | 0.9268 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2751 | 0.4604 | 0.5937 | 0.6935 | 0.7723 | 0.8351 | 0.8872 | 0.9301 | 0.9665 | 0.9978 |
| QAT+Prune only | 0.0000 | 0.5125 | 0.6919 | 0.7836 | 0.8400 | 0.8757 | 0.9043 | 0.9230 | 0.9395 | 0.9515 | 0.9624 |
| QAT+PTQ | 0.0000 | 0.5125 | 0.6918 | 0.7832 | 0.8399 | 0.8756 | 0.9039 | 0.9226 | 0.9393 | 0.9511 | 0.9620 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5125 | 0.6918 | 0.7832 | 0.8399 | 0.8756 | 0.9039 | 0.9226 | 0.9393 | 0.9511 | 0.9620 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4190 | 0.4173 | 0.4175 | 0.4177 | 0.4160 | 0.4170 | 0.4165 | 0.4190 | 0.4181 | 0.4167 | 0.0000 |
| QAT+Prune only | 0.8113 | 0.8119 | 0.8117 | 0.8115 | 0.8128 | 0.8093 | 0.8142 | 0.8082 | 0.8120 | 0.8012 | 0.0000 |
| QAT+PTQ | 0.8119 | 0.8121 | 0.8119 | 0.8116 | 0.8133 | 0.8099 | 0.8144 | 0.8081 | 0.8136 | 0.8015 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8119 | 0.8121 | 0.8119 | 0.8116 | 0.8133 | 0.8099 | 0.8144 | 0.8081 | 0.8136 | 0.8015 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4190 | 0.0000 | 0.0000 | 0.0000 | 0.4190 | 1.0000 |
| 90 | 10 | 299,940 | 0.4751 | 0.1596 | 0.9961 | 0.2751 | 0.4173 | 0.9990 |
| 80 | 20 | 291,350 | 0.5332 | 0.2994 | 0.9957 | 0.4604 | 0.4175 | 0.9974 |
| 70 | 30 | 194,230 | 0.5911 | 0.4229 | 0.9957 | 0.5937 | 0.4177 | 0.9956 |
| 60 | 40 | 145,675 | 0.6479 | 0.5320 | 0.9957 | 0.6935 | 0.4160 | 0.9931 |
| 50 | 50 | 116,540 | 0.7064 | 0.6307 | 0.9957 | 0.7723 | 0.4170 | 0.9898 |
| 40 | 60 | 97,115 | 0.7640 | 0.7191 | 0.9957 | 0.8351 | 0.4165 | 0.9847 |
| 30 | 70 | 83,240 | 0.8227 | 0.7999 | 0.9957 | 0.8872 | 0.4190 | 0.9766 |
| 20 | 80 | 72,835 | 0.8802 | 0.8725 | 0.9957 | 0.9301 | 0.4181 | 0.9604 |
| 10 | 90 | 64,740 | 0.9378 | 0.9389 | 0.9957 | 0.9665 | 0.4167 | 0.9149 |
| 0 | 100 | 58,270 | 0.9957 | 1.0000 | 0.9957 | 0.9978 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8113 | 0.0000 | 0.0000 | 0.0000 | 0.8113 | 1.0000 |
| 90 | 10 | 299,940 | 0.8235 | 0.3540 | 0.9278 | 0.5125 | 0.8119 | 0.9902 |
| 80 | 20 | 291,350 | 0.8348 | 0.5518 | 0.9275 | 0.6919 | 0.8117 | 0.9781 |
| 70 | 30 | 194,230 | 0.8463 | 0.6783 | 0.9275 | 0.7836 | 0.8115 | 0.9631 |
| 60 | 40 | 145,675 | 0.8587 | 0.7676 | 0.9275 | 0.8400 | 0.8128 | 0.9438 |
| 50 | 50 | 116,540 | 0.8684 | 0.8294 | 0.9275 | 0.8757 | 0.8093 | 0.9177 |
| 40 | 60 | 97,115 | 0.8822 | 0.8822 | 0.9275 | 0.9043 | 0.8142 | 0.8821 |
| 30 | 70 | 83,240 | 0.8917 | 0.9186 | 0.9275 | 0.9230 | 0.8082 | 0.8268 |
| 20 | 80 | 72,835 | 0.9044 | 0.9518 | 0.9275 | 0.9395 | 0.8120 | 0.7368 |
| 10 | 90 | 64,740 | 0.9148 | 0.9767 | 0.9275 | 0.9515 | 0.8012 | 0.5510 |
| 0 | 100 | 58,270 | 0.9275 | 1.0000 | 0.9275 | 0.9624 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8119 | 0.0000 | 0.0000 | 0.0000 | 0.8119 | 1.0000 |
| 90 | 10 | 299,940 | 0.8236 | 0.3541 | 0.9272 | 0.5125 | 0.8121 | 0.9901 |
| 80 | 20 | 291,350 | 0.8349 | 0.5519 | 0.9268 | 0.6918 | 0.8119 | 0.9779 |
| 70 | 30 | 194,230 | 0.8461 | 0.6782 | 0.9268 | 0.7832 | 0.8116 | 0.9628 |
| 60 | 40 | 145,675 | 0.8587 | 0.7680 | 0.9268 | 0.8399 | 0.8133 | 0.9434 |
| 50 | 50 | 116,540 | 0.8683 | 0.8298 | 0.9268 | 0.8756 | 0.8099 | 0.9171 |
| 40 | 60 | 97,115 | 0.8818 | 0.8822 | 0.9268 | 0.9039 | 0.8144 | 0.8811 |
| 30 | 70 | 83,240 | 0.8911 | 0.9185 | 0.9268 | 0.9226 | 0.8081 | 0.8254 |
| 20 | 80 | 72,835 | 0.9041 | 0.9521 | 0.9268 | 0.9393 | 0.8136 | 0.7353 |
| 10 | 90 | 64,740 | 0.9142 | 0.9768 | 0.9268 | 0.9511 | 0.8015 | 0.5488 |
| 0 | 100 | 58,270 | 0.9268 | 1.0000 | 0.9268 | 0.9620 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8119 | 0.0000 | 0.0000 | 0.0000 | 0.8119 | 1.0000 |
| 90 | 10 | 299,940 | 0.8236 | 0.3541 | 0.9272 | 0.5125 | 0.8121 | 0.9901 |
| 80 | 20 | 291,350 | 0.8349 | 0.5519 | 0.9268 | 0.6918 | 0.8119 | 0.9779 |
| 70 | 30 | 194,230 | 0.8461 | 0.6782 | 0.9268 | 0.7832 | 0.8116 | 0.9628 |
| 60 | 40 | 145,675 | 0.8587 | 0.7680 | 0.9268 | 0.8399 | 0.8133 | 0.9434 |
| 50 | 50 | 116,540 | 0.8683 | 0.8298 | 0.9268 | 0.8756 | 0.8099 | 0.9171 |
| 40 | 60 | 97,115 | 0.8818 | 0.8822 | 0.9268 | 0.9039 | 0.8144 | 0.8811 |
| 30 | 70 | 83,240 | 0.8911 | 0.9185 | 0.9268 | 0.9226 | 0.8081 | 0.8254 |
| 20 | 80 | 72,835 | 0.9041 | 0.9521 | 0.9268 | 0.9393 | 0.8136 | 0.7353 |
| 10 | 90 | 64,740 | 0.9142 | 0.9768 | 0.9268 | 0.9511 | 0.8015 | 0.5488 |
| 0 | 100 | 58,270 | 0.9268 | 1.0000 | 0.9268 | 0.9620 | 0.0000 | 0.0000 |


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
0.15       0.4751   0.2751   0.4173   0.9989   0.9959   0.1596   <--
0.20       0.4751   0.2751   0.4173   0.9989   0.9959   0.1596  
0.25       0.4751   0.2751   0.4173   0.9989   0.9959   0.1596  
0.30       0.4751   0.2751   0.4173   0.9989   0.9959   0.1596  
0.35       0.4751   0.2751   0.4173   0.9989   0.9959   0.1596  
0.40       0.4751   0.2751   0.4173   0.9989   0.9959   0.1596  
0.45       0.4751   0.2751   0.4173   0.9989   0.9959   0.1596  
0.50       0.4751   0.2751   0.4173   0.9989   0.9959   0.1596  
0.55       0.4751   0.2751   0.4173   0.9989   0.9959   0.1596  
0.60       0.4751   0.2751   0.4173   0.9989   0.9959   0.1596  
0.65       0.4751   0.2751   0.4173   0.9989   0.9959   0.1596  
0.70       0.4751   0.2751   0.4173   0.9989   0.9959   0.1596  
0.75       0.4751   0.2751   0.4173   0.9989   0.9959   0.1596  
0.80       0.4751   0.2751   0.4173   0.9989   0.9959   0.1596  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4751, F1=0.2751, Normal Recall=0.4173, Normal Precision=0.9989, Attack Recall=0.9959, Attack Precision=0.1596

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
0.15       0.5330   0.4603   0.4173   0.9974   0.9957   0.2993   <--
0.20       0.5330   0.4603   0.4173   0.9974   0.9957   0.2993  
0.25       0.5330   0.4603   0.4173   0.9974   0.9957   0.2993  
0.30       0.5330   0.4603   0.4173   0.9974   0.9957   0.2993  
0.35       0.5330   0.4603   0.4173   0.9974   0.9957   0.2993  
0.40       0.5330   0.4603   0.4173   0.9974   0.9957   0.2993  
0.45       0.5330   0.4603   0.4173   0.9974   0.9957   0.2993  
0.50       0.5330   0.4603   0.4173   0.9974   0.9957   0.2993  
0.55       0.5330   0.4603   0.4173   0.9974   0.9957   0.2993  
0.60       0.5330   0.4603   0.4173   0.9974   0.9957   0.2993  
0.65       0.5330   0.4603   0.4173   0.9974   0.9957   0.2993  
0.70       0.5330   0.4603   0.4173   0.9974   0.9957   0.2993  
0.75       0.5330   0.4603   0.4173   0.9974   0.9957   0.2993  
0.80       0.5330   0.4603   0.4173   0.9974   0.9957   0.2993  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5330, F1=0.4603, Normal Recall=0.4173, Normal Precision=0.9974, Attack Recall=0.9957, Attack Precision=0.2993

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
0.15       0.5924   0.5944   0.4195   0.9956   0.9957   0.4237   <--
0.20       0.5924   0.5944   0.4195   0.9956   0.9957   0.4237  
0.25       0.5924   0.5944   0.4195   0.9956   0.9957   0.4237  
0.30       0.5924   0.5944   0.4195   0.9956   0.9957   0.4237  
0.35       0.5924   0.5944   0.4195   0.9956   0.9957   0.4237  
0.40       0.5924   0.5944   0.4195   0.9956   0.9957   0.4237  
0.45       0.5924   0.5944   0.4195   0.9956   0.9957   0.4237  
0.50       0.5924   0.5944   0.4195   0.9956   0.9957   0.4237  
0.55       0.5924   0.5944   0.4195   0.9956   0.9957   0.4237  
0.60       0.5924   0.5944   0.4195   0.9956   0.9957   0.4237  
0.65       0.5924   0.5944   0.4195   0.9956   0.9957   0.4237  
0.70       0.5924   0.5944   0.4195   0.9956   0.9957   0.4237  
0.75       0.5924   0.5944   0.4195   0.9956   0.9957   0.4237  
0.80       0.5924   0.5944   0.4195   0.9956   0.9957   0.4237  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5924, F1=0.5944, Normal Recall=0.4195, Normal Precision=0.9956, Attack Recall=0.9957, Attack Precision=0.4237

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
0.15       0.6497   0.6945   0.4190   0.9932   0.9957   0.5333   <--
0.20       0.6497   0.6945   0.4190   0.9932   0.9957   0.5333  
0.25       0.6497   0.6945   0.4190   0.9932   0.9957   0.5333  
0.30       0.6497   0.6945   0.4190   0.9932   0.9957   0.5333  
0.35       0.6497   0.6945   0.4190   0.9932   0.9957   0.5333  
0.40       0.6497   0.6945   0.4190   0.9932   0.9957   0.5333  
0.45       0.6497   0.6945   0.4190   0.9932   0.9957   0.5333  
0.50       0.6497   0.6945   0.4190   0.9932   0.9957   0.5333  
0.55       0.6497   0.6945   0.4190   0.9932   0.9957   0.5333  
0.60       0.6497   0.6945   0.4190   0.9932   0.9957   0.5333  
0.65       0.6497   0.6945   0.4190   0.9932   0.9957   0.5333  
0.70       0.6497   0.6945   0.4190   0.9932   0.9957   0.5333  
0.75       0.6497   0.6945   0.4190   0.9932   0.9957   0.5333  
0.80       0.6497   0.6945   0.4190   0.9932   0.9957   0.5333  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6497, F1=0.6945, Normal Recall=0.4190, Normal Precision=0.9932, Attack Recall=0.9957, Attack Precision=0.5333

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
0.15       0.7071   0.7727   0.4186   0.9898   0.9957   0.6313   <--
0.20       0.7071   0.7727   0.4186   0.9898   0.9957   0.6313  
0.25       0.7071   0.7727   0.4186   0.9898   0.9957   0.6313  
0.30       0.7071   0.7727   0.4186   0.9898   0.9957   0.6313  
0.35       0.7071   0.7727   0.4186   0.9898   0.9957   0.6313  
0.40       0.7071   0.7727   0.4186   0.9898   0.9957   0.6313  
0.45       0.7071   0.7727   0.4186   0.9898   0.9957   0.6313  
0.50       0.7071   0.7727   0.4186   0.9898   0.9957   0.6313  
0.55       0.7071   0.7727   0.4186   0.9898   0.9957   0.6313  
0.60       0.7071   0.7727   0.4186   0.9898   0.9957   0.6313  
0.65       0.7071   0.7727   0.4186   0.9898   0.9957   0.6313  
0.70       0.7071   0.7727   0.4186   0.9898   0.9957   0.6313  
0.75       0.7071   0.7727   0.4186   0.9898   0.9957   0.6313  
0.80       0.7071   0.7727   0.4186   0.9898   0.9957   0.6313  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7071, F1=0.7727, Normal Recall=0.4186, Normal Precision=0.9898, Attack Recall=0.9957, Attack Precision=0.6313

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
0.15       0.8234   0.5121   0.8119   0.9901   0.9269   0.3538   <--
0.20       0.8234   0.5121   0.8119   0.9901   0.9269   0.3538  
0.25       0.8234   0.5121   0.8119   0.9901   0.9269   0.3538  
0.30       0.8234   0.5121   0.8119   0.9901   0.9269   0.3538  
0.35       0.8234   0.5121   0.8119   0.9901   0.9269   0.3538  
0.40       0.8234   0.5121   0.8119   0.9901   0.9269   0.3538  
0.45       0.8234   0.5121   0.8119   0.9901   0.9269   0.3538  
0.50       0.8234   0.5121   0.8119   0.9901   0.9269   0.3538  
0.55       0.8234   0.5121   0.8119   0.9901   0.9269   0.3538  
0.60       0.8234   0.5121   0.8119   0.9901   0.9269   0.3538  
0.65       0.8234   0.5121   0.8119   0.9901   0.9269   0.3538  
0.70       0.8234   0.5121   0.8119   0.9901   0.9269   0.3538  
0.75       0.8234   0.5121   0.8119   0.9901   0.9269   0.3538  
0.80       0.8234   0.5121   0.8119   0.9901   0.9269   0.3538  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8234, F1=0.5121, Normal Recall=0.8119, Normal Precision=0.9901, Attack Recall=0.9269, Attack Precision=0.3538

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
0.15       0.8351   0.6923   0.8120   0.9782   0.9275   0.5523   <--
0.20       0.8351   0.6923   0.8120   0.9782   0.9275   0.5523  
0.25       0.8351   0.6923   0.8120   0.9782   0.9275   0.5523  
0.30       0.8351   0.6923   0.8120   0.9782   0.9275   0.5523  
0.35       0.8351   0.6923   0.8120   0.9782   0.9275   0.5523  
0.40       0.8351   0.6923   0.8120   0.9782   0.9275   0.5523  
0.45       0.8351   0.6923   0.8120   0.9782   0.9275   0.5523  
0.50       0.8351   0.6923   0.8120   0.9782   0.9275   0.5523  
0.55       0.8351   0.6923   0.8120   0.9782   0.9275   0.5523  
0.60       0.8351   0.6923   0.8120   0.9782   0.9275   0.5523  
0.65       0.8351   0.6923   0.8120   0.9782   0.9275   0.5523  
0.70       0.8351   0.6923   0.8120   0.9782   0.9275   0.5523  
0.75       0.8351   0.6923   0.8120   0.9782   0.9275   0.5523  
0.80       0.8351   0.6923   0.8120   0.9782   0.9275   0.5523  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8351, F1=0.6923, Normal Recall=0.8120, Normal Precision=0.9782, Attack Recall=0.9275, Attack Precision=0.5523

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
0.15       0.8466   0.7839   0.8120   0.9631   0.9275   0.6789   <--
0.20       0.8466   0.7839   0.8120   0.9631   0.9275   0.6789  
0.25       0.8466   0.7839   0.8120   0.9631   0.9275   0.6789  
0.30       0.8466   0.7839   0.8120   0.9631   0.9275   0.6789  
0.35       0.8466   0.7839   0.8120   0.9631   0.9275   0.6789  
0.40       0.8466   0.7839   0.8120   0.9631   0.9275   0.6789  
0.45       0.8466   0.7839   0.8120   0.9631   0.9275   0.6789  
0.50       0.8466   0.7839   0.8120   0.9631   0.9275   0.6789  
0.55       0.8466   0.7839   0.8120   0.9631   0.9275   0.6789  
0.60       0.8466   0.7839   0.8120   0.9631   0.9275   0.6789  
0.65       0.8466   0.7839   0.8120   0.9631   0.9275   0.6789  
0.70       0.8466   0.7839   0.8120   0.9631   0.9275   0.6789  
0.75       0.8466   0.7839   0.8120   0.9631   0.9275   0.6789  
0.80       0.8466   0.7839   0.8120   0.9631   0.9275   0.6789  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8466, F1=0.7839, Normal Recall=0.8120, Normal Precision=0.9631, Attack Recall=0.9275, Attack Precision=0.6789

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
0.15       0.8580   0.8393   0.8117   0.9438   0.9275   0.7665   <--
0.20       0.8580   0.8393   0.8117   0.9438   0.9275   0.7665  
0.25       0.8580   0.8393   0.8117   0.9438   0.9275   0.7665  
0.30       0.8580   0.8393   0.8117   0.9438   0.9275   0.7665  
0.35       0.8580   0.8393   0.8117   0.9438   0.9275   0.7665  
0.40       0.8580   0.8393   0.8117   0.9438   0.9275   0.7665  
0.45       0.8580   0.8393   0.8117   0.9438   0.9275   0.7665  
0.50       0.8580   0.8393   0.8117   0.9438   0.9275   0.7665  
0.55       0.8580   0.8393   0.8117   0.9438   0.9275   0.7665  
0.60       0.8580   0.8393   0.8117   0.9438   0.9275   0.7665  
0.65       0.8580   0.8393   0.8117   0.9438   0.9275   0.7665  
0.70       0.8580   0.8393   0.8117   0.9438   0.9275   0.7665  
0.75       0.8580   0.8393   0.8117   0.9438   0.9275   0.7665  
0.80       0.8580   0.8393   0.8117   0.9438   0.9275   0.7665  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8580, F1=0.8393, Normal Recall=0.8117, Normal Precision=0.9438, Attack Recall=0.9275, Attack Precision=0.7665

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
0.15       0.8695   0.8766   0.8115   0.9179   0.9275   0.8311   <--
0.20       0.8695   0.8766   0.8115   0.9179   0.9275   0.8311  
0.25       0.8695   0.8766   0.8115   0.9179   0.9275   0.8311  
0.30       0.8695   0.8766   0.8115   0.9179   0.9275   0.8311  
0.35       0.8695   0.8766   0.8115   0.9179   0.9275   0.8311  
0.40       0.8695   0.8766   0.8115   0.9179   0.9275   0.8311  
0.45       0.8695   0.8766   0.8115   0.9179   0.9275   0.8311  
0.50       0.8695   0.8766   0.8115   0.9179   0.9275   0.8311  
0.55       0.8695   0.8766   0.8115   0.9179   0.9275   0.8311  
0.60       0.8695   0.8766   0.8115   0.9179   0.9275   0.8311  
0.65       0.8695   0.8766   0.8115   0.9179   0.9275   0.8311  
0.70       0.8695   0.8766   0.8115   0.9179   0.9275   0.8311  
0.75       0.8695   0.8766   0.8115   0.9179   0.9275   0.8311  
0.80       0.8695   0.8766   0.8115   0.9179   0.9275   0.8311  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8695, F1=0.8766, Normal Recall=0.8115, Normal Precision=0.9179, Attack Recall=0.9275, Attack Precision=0.8311

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
0.15       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539   <--
0.20       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.25       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.30       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.35       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.40       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.45       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.50       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.55       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.60       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.65       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.70       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.75       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.80       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8235, F1=0.5121, Normal Recall=0.8121, Normal Precision=0.9900, Attack Recall=0.9262, Attack Precision=0.3539

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
0.15       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525   <--
0.20       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.25       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.30       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.35       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.40       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.45       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.50       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.55       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.60       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.65       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.70       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.75       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.80       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8352, F1=0.6923, Normal Recall=0.8123, Normal Precision=0.9780, Attack Recall=0.9268, Attack Precision=0.5525

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
0.15       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791   <--
0.20       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.25       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.30       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.35       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.40       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.45       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.50       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.55       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.60       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.65       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.70       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.75       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.80       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8467, F1=0.7839, Normal Recall=0.8123, Normal Precision=0.9628, Attack Recall=0.9268, Attack Precision=0.6791

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
0.15       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669   <--
0.20       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.25       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.30       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.35       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.40       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.45       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.50       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.55       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.60       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.65       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.70       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.75       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.80       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8580, F1=0.8393, Normal Recall=0.8122, Normal Precision=0.9433, Attack Recall=0.9268, Attack Precision=0.7669

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
0.15       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316   <--
0.20       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.25       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.30       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.35       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.40       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.45       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.50       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.55       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.60       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.65       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.70       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.75       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.80       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8696, F1=0.8766, Normal Recall=0.8124, Normal Precision=0.9173, Attack Recall=0.9268, Attack Precision=0.8316

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
0.15       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539   <--
0.20       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.25       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.30       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.35       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.40       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.45       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.50       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.55       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.60       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.65       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.70       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.75       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
0.80       0.8235   0.5121   0.8121   0.9900   0.9262   0.3539  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8235, F1=0.5121, Normal Recall=0.8121, Normal Precision=0.9900, Attack Recall=0.9262, Attack Precision=0.3539

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
0.15       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525   <--
0.20       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.25       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.30       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.35       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.40       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.45       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.50       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.55       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.60       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.65       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.70       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.75       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
0.80       0.8352   0.6923   0.8123   0.9780   0.9268   0.5525  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8352, F1=0.6923, Normal Recall=0.8123, Normal Precision=0.9780, Attack Recall=0.9268, Attack Precision=0.5525

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
0.15       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791   <--
0.20       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.25       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.30       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.35       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.40       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.45       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.50       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.55       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.60       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.65       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.70       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.75       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
0.80       0.8467   0.7839   0.8123   0.9628   0.9268   0.6791  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8467, F1=0.7839, Normal Recall=0.8123, Normal Precision=0.9628, Attack Recall=0.9268, Attack Precision=0.6791

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
0.15       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669   <--
0.20       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.25       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.30       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.35       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.40       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.45       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.50       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.55       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.60       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.65       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.70       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.75       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
0.80       0.8580   0.8393   0.8122   0.9433   0.9268   0.7669  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8580, F1=0.8393, Normal Recall=0.8122, Normal Precision=0.9433, Attack Recall=0.9268, Attack Precision=0.7669

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
0.15       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316   <--
0.20       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.25       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.30       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.35       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.40       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.45       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.50       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.55       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.60       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.65       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.70       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.75       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
0.80       0.8696   0.8766   0.8124   0.9173   0.9268   0.8316  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8696, F1=0.8766, Normal Recall=0.8124, Normal Precision=0.9173, Attack Recall=0.9268, Attack Precision=0.8316

```

