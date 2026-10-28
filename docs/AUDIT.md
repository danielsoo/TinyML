# Audit record (Dr. Saha's experimental guide, section 2)

Status per check: **PASS**, **PROBLEM (confirmed)**, or **OPEN** (needs data; job `2026-10-28_y1_audit`).
"Old pipeline" = `src/data/loader.py` + `partition_data`, used for every result up to 2026-10-26.
"v13" = `src/data/fed_pipeline.py`, the corrected pipeline used for the re-run.

## Data separation and provenance
| # | Check | Status | Evidence / action |
|---|---|---|---|
| D1 | Record ids, cleaning rules, split indices recoverable | PROBLEM (old) → fixed in v13 | Old loaders kept no record ids. v13 keeps `file:row` ids and source metadata and writes a manifest per run (ids per partition, SHA-256 hashes, counts, config). The old test split is re-derived by `audit_pipeline.old_split_ids` (verified to reproduce the old loader's test rows exactly). |
| D2 | Threshold-validation records excluded from teacher training | **PROBLEM (confirmed)** | `scripts/recall_priority.py` draws its validation slice from client 0's partition, which is part of client 0's federated training data (and of the fine-tuning pool). v13: each client holds out a real validation slice before training; thresholds use only those. |
| D3 | Validation uses real records | **PROBLEM (confirmed)** | The old partitions were built after SMOTE, so the validation slice contains synthetic rows (share measured by the audit job). v13 has no SMOTE. |
| D4 | No record or synthetic descendant crosses a partition boundary | **PROBLEM (confirmed)** | SMOTE ran on the pooled training set before `partition_data`; synthetic rows interpolate records that end up on other clients. v13 assigns clients before any fitted step and drops SMOTE. Exact duplicates were removed before the split in both pipelines (PASS for exact duplicates). |
| D5 | Scaler / encoders fit on training information only | PASS (old, with caveat) / PASS (v13) | Old: StandardScaler fit on the pooled, undersampled training split (no test). Label encoding (`factorize`) uses category sets of the whole file (no label information). v13: mean / variance of the union of client training slices, the aggregate a federation can compute from per-client sums; validation and test are not used. |
| D6 | Final test set untouched by method development | **PROBLEM (confirmed)** | Recipe and deployment choices (e.g. the 112 KB CIC model, distillation / clipping variants) were compared on the test split. v13 uses a fresh split (`split_seed` 2027); the audit reports how many new-test records were in the old test split. The TON_IoT grouped / temporal split (E6) is an additional confirmation set. |
| D7 | Duplicates and conflicting labels | OPEN | Old CIC dedup was keyed on features + label, so identical vectors with different labels were kept. v13 drops conflicting-label groups and counts them (`conflict_policy: drop`). |

## Client ownership and access
| # | Check | Status | Evidence / action |
|---|---|---|---|
| C1 | Client assignment precedes SMOTE / fitted preprocessing | **PROBLEM (confirmed)** | See D4. Undersampling was also pooled. v13: per-client undersampling (4:1) on each client's training slice. |
| C2 | Which participant holds each calibration / fine-tuning sample | PASS (documented) | Fine-tuning and calibration draws come from client 0's partition (or all clients for federated fine-tuning). Under v13 they come from the client's training slice; validation slices are used only for threshold and estimator selection. |
| C3 | SMOTE on label-encoded categorical fields | **PROBLEM (confirmed by code)**, size OPEN | TON_IoT label-encodes 17 text columns and SMOTE interpolates them (and integer counts in both datasets). The audit job measures the share of synthetic rows with a value that never occurs in real data. v13 removes SMOTE (class weights + focal loss remain). |

## Numerical and model checks
| # | Check | Status | Evidence / action |
|---|---|---|---|
| N1 | Extreme records behind failed calibration | OPEN | Audit lists the most extreme training records (|z|, raw value, feature, family, id) and negative / inf / NaN counts per column, to separate genuine extremes from data errors. E3 then pairs calibration draws with and without them. |
| N2 | Input conversion: rounding then saturation to INT8 | PASS (deployed models) | Both deployment models take float input and start with TFLite QUANTIZE (scale 0.0392 / zp 0 for CIC, i.e. [-5.02, 5.0]; 0.0314 / zp -32 for TON, i.e. [-3.0, 5.0]); rounding and saturation follow the LiteRT int8 spec. Device parity (8/8 decisions, ESP32 and ESP32-S3) used the same float inputs. E7 extends this to thousands of inputs including out-of-range values. |
| N3 | Weight transfer and BatchNorm folding | PASS | Folding into the next layer is exact (≤ 2e-5); QAT stripping copies by name and refuses empty copies (Section "silent failures"). Re-checked for every v13 checkpoint by the evaluation script. |
| N4 | No test data used during training | PASS | FL clients evaluate on their test shard only for logging (`use_callbacks` is off in every config, so no early stopping / LR-on-plateau on test). The centralized baseline passes the test set as `validation_data` for logging only (no callbacks use it). v13 logs on validation slices instead. |
| N5 | Seeds / independent runs | PROBLEM (minor) → fixed in v13 | Initial weights were not seeded and recorded. v13 configs set `federated.seed` and record it. |

## Results that must be re-run
Everything that depends on the old partitions or on test-guided choices: federated teachers,
compression / recall-priority tables, calibration studies. They remain exploratory evidence (they
motivated the hypotheses) but are not used for the final claims. The ESP32 / ESP32-S3 harness and
timings remain valid as measurements of those specific models and will be repeated for the final ones.
