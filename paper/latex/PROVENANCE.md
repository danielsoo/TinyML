# Provenance of every number in main.tex

All numbers come from our own runs on the user's PC (WSL2, CPU simulation of Flower) and on two
boards (ESP32-S3 N16R8, ESP-WROOM-32). Each run directory under `data/processed/revision/<job>/`
holds `job.yaml`, `git_commit.txt`, the configs, step logs (`*.log`) and result files. The numbers
are collected in `docs/REVISION_RESULTS.md` (section "v13.1 re-run"); the audit is `docs/AUDIT.md`.
Pipeline: `src/data/fed_pipeline.py` with `config/v13/*.yaml`. Results of the earlier, exploratory
pipeline (jobs before 2026-10-28, manuscript `archive/main_v12_exploratory.tex`) are not used.

## Jobs (all under `data/processed/revision/`)
| Job | What it produced |
|---|---|
| 2026-10-28_y1_audit | audit of the old pipeline (Appendix A, §3.1 numbers on SMOTE, duplicates, data errors) |
| 2026-10-29_z1_train_ton | TON_IoT teachers F and compact S80 / S128, seeds 1-3, with manifests |
| 2026-10-29_z2_train_cic | CIC-IDS2017 teachers F and compact S80 / S128, seeds 1-3 |
| 2026-10-29_z3_train_ton_group | TON_IoT host-grouped split (E6), F / S80 / S128, seeds 1-3 |
| 2026-10-29_z1d_compare_ton_fed, z2b_compare_cic_fed | E1 / E2, federated fine-tuning (F, F8, P, Q, R, S, SQ) |
| 2026-10-29_z1e_compare_ton_client, z2c_compare_cic_client | E1, one-client fine-tuning |
| 2026-10-29_z1g_compare_ton_qc, z2e_compare_cic_qc | QC, SQC (QAT on clipped inputs) |
| 2026-10-29_z3b_compare_ton_group | E6, all arms on the host-grouped split |
| 2026-10-29_z1f_range_ton, z2d_range_cic | E3, E4 (first rows) |
| 2026-10-29_z1h_range_ton2, z2f_range_cic2 | E3, E4 with clipped-FP32 and 16-bit rows |
| 2026-10-29_z4_device_bundle, 2026-11-05_z7_device_bundle | E7 input bundles (v1, v2 with per-tensor re-exports) |
| 2026-11-01_z6_device_e7 | E7 first ESP32-S3 run (per-channel finding) |
| 2026-11-05_z8_device_e7, z8c_device_e7_s3_fill | E7 on the ESP32-S3 |
| 2026-11-05_z8b_device_e7 | E7 on the ESP32 |

Superseded (kept as pipeline checks): 2026-10-28_y2_train_ton and 2026-10-28_y2b_compare_ton_fed
(v13 before the two audit rules), 2026-10-29_z1b / z1c (threshold tie bug), 2026-11-01_z5 (step-name
and pip-path bugs), 2026-11-06_z10b / z10c / z10e (ESP-IDF build set-up: pkg_resources, component-manager
version, TFLM header path, git-ignored sdkconfig file; no measurements).

## Paper items
| Paper item | Source |
|---|---|
| Table 1 | `*_train_*/train_fl/models/F_s1_manifest.json` (dedup, invalid_records, test, clients) |
| Table 3 (deployable models), Table 11 (findings) | `paper/supplement/S8_deployable_models.csv`, `S1`, `S2` from `scripts/build_supplement.py` (reported per-tensor exports; latency from z8 / z8b) |
| Table 2 | `compare_arms/compare_arms_summary.csv` of z1d, z2b (fed), z1e, z2c (1 client), z1g, z2e (QC, SQC); F8 / P / S80 at 80 KB from the per-tensor exports of z9 (`saha_checks.json` -> exports); per run in `paper/supplement/S1_operating_points.csv` |
| Table 4 (E6) | z3b `compare_arms_summary.csv` |
| Table 5, Figure 1 | `compare_arms_transitions.csv` of z2b, z1d (FAR 0.01, budget 80, summed over seeds) |
| Table 6, Figure 2 | z2d, z1f `range_mechanism/e3_summary.csv` (model PQ_80) |
| Table 7 (E4) | z2f, z1h `range_mechanism/e4_summary.csv` |
| Table 8 (remedies) | Tables 2, 3, 6, 7, 9 and 10 |
| Table 9 (E7) | `device_e7/<board>/device_agreement.json` and `device_*.npy` of z8, z8c (S3) and z8b (ESP32), compared with the host outputs of the z7 bundle |
| Table 10, §7 current runtime | `docs/runtime_compare/runtime_compare.{md,csv}` from `scripts/runtime_compare.py` over z10 (TFLM 1.0.0), z10d (esp-tflite-micro 1.4.2, esp-nn), z10f (reference kernels); build provenance in each `device_e7/<board>/build_meta/` |
| §3.4 tie-group pitfall | z1b vs z1d (same models, float64 vs float32 thresholds) |

## Supplement
`scripts/build_supplement.py` writes `paper/supplement/S1-S8*.csv` and `supplement.tex/pdf` from existing records only: S1 per-run operating points (both budgets, export file and SHA-256), S2 run-paired differences and the z9 McNemar tests (with the export each used), S3 device disagreement classes, S4 memory/latency, S5 stream completeness, S6 runtime comparison, S7 split provenance and freeze history, S8 deployable models.
