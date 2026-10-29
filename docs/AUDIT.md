# Audit record (Dr. Saha's experimental guide, section 2)

Status per check: **PASS** or **PROBLEM (confirmed)**. Measurements: job `2026-10-28_y1_audit`
(`data/processed/revision/2026-10-28_y1_audit/audit/audit.json`).
"Old pipeline" = `src/data/loader.py` + `partition_data`, used for every result up to 2026-10-26.
"v13" = `src/data/fed_pipeline.py`, the corrected pipeline; "v13.1" = v13 plus the two rules adopted
after the audit (data-error records dropped, old test records excluded from the new test split).
All final results use v13.1 (configs `config/v13/`, jobs `2026-10-29_z*`).

## Data separation and provenance
| # | Check | Status | Evidence / action |
|---|---|---|---|
| D1 | Record ids, cleaning rules, split indices recoverable | PROBLEM (old) → fixed in v13 | Old loaders kept no record ids. v13 keeps `file:row` ids and source metadata and writes a manifest per run (ids per partition, SHA-256 hashes, counts, config). The old test split is re-derived by `audit_pipeline.old_split_ids` (verified to reproduce the old loader's test rows exactly). |
| D2 | Threshold-validation records excluded from teacher training | **PROBLEM (confirmed)** | `scripts/recall_priority.py` draws its validation slice from client 0's partition, which is part of client 0's federated training data (and of the fine-tuning pool). v13: each client holds out a real validation slice before training; thresholds use only those. |
| D3 | Validation uses real records | **PROBLEM (confirmed)** | The old partitions were built after SMOTE, so the recall-priority validation slice was 37.8% synthetic on CIC-IDS2017 (**74.6% of its attack rows**) and 35.1% on TON_IoT (**71.0% of its benign rows**; benign is the minority class there). Thresholds were therefore tuned largely on synthetic records of the minority class. v13 has no SMOTE. |
| D4 | No record or synthetic descendant crosses a partition boundary | **PROBLEM (confirmed)** | SMOTE ran on the pooled training set before `partition_data`; synthetic rows interpolate records that end up on other clients. Each client received ~255k synthetic rows on CIC-IDS2017 (37.5% of training) and ~11.4k on TON_IoT (35.1%). v13 assigns clients before any fitted step and drops SMOTE. Exact duplicates were removed before the split in both pipelines (PASS for exact duplicates). |
| D5 | Scaler / encoders fit on training information only | PASS (old, with caveat) / PASS (v13) | Old: StandardScaler fit on the pooled, undersampled training split (no test). Label encoding (`factorize`) uses category sets of the whole file (no label information). v13: mean / variance of the union of client training slices, the aggregate a federation can compute from per-client sums; validation and test are not used. |
| D6 | Final test set untouched by method development | **PROBLEM (confirmed)** → fixed in v13.1 | Recipe and deployment choices (e.g. the 112 KB CIC model, distillation / clipping variants) were compared on the test split. A fresh random split alone still shared 20% of its records with the old test split (100,169 / 499,621 on CIC-IDS2017, 4,175 / 21,032 on TON_IoT). v13.1 re-derives the old test ids (`fed_pipeline.old_test_ids`, verified to reproduce the old loader) and draws the new test split only from records that were never in it (overlap 0). The TON_IoT grouped split (E6) is an additional confirmation set. |
| D7 | Duplicates and conflicting labels | PROBLEM (minor) → fixed in v13 | CIC-IDS2017: 2,830,743 records, 331,200 exact duplicates removed, 719 feature vectors occur with both labels (7,144 records). TON_IoT: 211,043 records, **105,867 duplicates (50.2%)**, 8 conflicting vectors (314 records). The old CIC dedup was keyed on features + label, so the conflicting vectors were kept with both labels. v13 removes duplicates before the split and drops conflicting-label groups (`conflict_policy: drop`). |

## Client ownership and access
| # | Check | Status | Evidence / action |
|---|---|---|---|
| C1 | Client assignment precedes SMOTE / fitted preprocessing | **PROBLEM (confirmed)** | See D4. Undersampling was also pooled. v13: per-client undersampling (4:1) on each client's training slice. |
| C2 | Which participant holds each calibration / fine-tuning sample | PASS (documented) | Fine-tuning and calibration draws come from client 0's partition (or all clients for federated fine-tuning). Under v13 they come from the client's training slice; validation slices are used only for threshold and estimator selection. |
| C3 | SMOTE on label-encoded categorical fields | **PROBLEM (confirmed)** | TON_IoT label-encodes 19 text columns and SMOTE interpolates them (and integer counts in both datasets). Share of synthetic rows with a value in a discrete column that never occurs in real data: **4.69% on TON_IoT** (worst: `dns_query_len` 3.1%, other text-derived counts up to 0.8%) and 0.04% on CIC-IDS2017. v13 removes SMOTE (class weights + focal loss remain). |

## Numerical and model checks
| # | Check | Status | Evidence / action |
|---|---|---|---|
| N1 | Extreme records behind failed calibration | **PROBLEM (confirmed, CIC-IDS2017)** → fixed in v13.1 | The most extreme CIC-IDS2017 training records are data errors, not rare traffic: negative header lengths from integer overflow (`Fwd Header Length` = −3.2·10¹⁰, z = −1,149; `Bwd Header Length` = −1.07·10⁹, z = −609, several records) and negative durations, rates and inter-arrival times (`Flow Duration` 115, `Flow IAT Min` 2,891 records). They also inflate the scaler's SD for those columns. `Init_Win_bytes_*` = −1 (1.0M / 1.4M records) is a defined sentinel (no window observed) and is kept. v13.1 drops records with any other negative value (`drop_negative`, counts per column and family in each manifest). TON_IoT has no negative values; its extremes (e.g. `dst_ip_bytes` 8.6·10⁷ in a DoS flow, z = 272) are plausible traffic and are kept. Share of training records beyond 100 SD: 7·10⁻⁵ (CIC), 2.8·10⁻⁴ (TON). E3 pairs calibration draws with and without the remaining extremes. |
| N2 | Input conversion: rounding then saturation to INT8 | PASS (deployed models) | Both deployment models take float input and start with TFLite QUANTIZE (scale 0.0392 / zp 0 for CIC, i.e. [-5.02, 5.0]; 0.0314 / zp -32 for TON, i.e. [-3.0, 5.0]); rounding and saturation follow the LiteRT int8 spec. Device parity (8/8 decisions, ESP32 and ESP32-S3) used the same float inputs. E7 extends this to thousands of inputs including out-of-range values. |
| N3 | Weight transfer and BatchNorm folding | PASS | Folding into the next layer is exact (≤ 2e-5); QAT stripping copies by name and refuses empty copies (Section "silent failures"). Re-checked for every v13 checkpoint by the evaluation script. |
| N4 | No test data used during training | PASS | FL clients evaluate on their test shard only for logging (`use_callbacks` is off in every config, so no early stopping / LR-on-plateau on test). The centralized baseline passes the test set as `validation_data` for logging only (no callbacks use it). v13 logs on validation slices instead. |
| N5 | Seeds / independent runs | PROBLEM (minor) → fixed in v13 | Initial weights were not seeded and recorded. v13 configs set `federated.seed` and record it. |

## Grouped / temporal split (E6)
`train_test_network.csv` has no timestamp column, so a temporal split is not possible on this file;
E6 uses a grouped split by host (`src_ip` / `dst_ip` are kept as metadata, never as model inputs).

## Results that must be re-run
Everything that depends on the old partitions or on test-guided choices: federated teachers,
compression / recall-priority tables, calibration studies. They remain exploratory evidence (they
motivated the hypotheses) but are not used for the final claims. The ESP32 / ESP32-S3 harness and
timings remain valid as measurements of those specific models and will be repeated for the final ones.
The first v13 TON_IoT checkpoints (job `2026-10-28_y2_train_ton`, 9 models, 27 min) were trained before
the two v13.1 rules and are superseded by job `2026-10-29_z1_train_ton`; they are kept as a pipeline check.
