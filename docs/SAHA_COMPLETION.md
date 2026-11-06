# Completion record for Dr. Saha's list (email of 2026-11-05)

Numbers: `docs/REVISION_RESULTS.md` (v13.1), `docs/model_hashes.csv`, and the job folders under
`data/processed/revision/`. Jobs run on 2026-11-06: **z9** (`saha_checks`, done: §1b, §2b, §4b) and
**z10** (`runtime_compare`, classic ESP32; pending). Nothing was retrained; thresholds stayed frozen.

## Completion table
| # | Item | Existing evidence | Unresolved | Additional execution? |
|---|---|---|---|---|
| 1 | E1/E5 operating points | Per-run recall **and achieved test FAR** for every arm, both budgets, random split and unseen hosts; run-paired and record-level paired differences with exact McNemar tests (§1, §1b) | None | Done (z9) |
| 2 | Exact deployed models | SHA-256, size, weight-scale type and conversion of every export (`docs/model_hashes.csv`, §2); per-tensor strong-PTQ replacements for all 27 F8/P/S runs and the two flashed files, with recall, FAR, size, hash (§2b) | None (whether the new runtime runs the per-channel exports: z10) | Done (z9) |
| 3 | E7 disagreements, resets, stream completeness | Every disagreement classified (§3): per-tensor INT8 models add **0** missed attacks (one CIC record for S80 at 1%) and only false alarms on inputs chosen within two steps of the threshold; 0 disagreements on 1,500 random records except one. Resets: accounted (§3); final streams complete, no missing or duplicated records; S3 and ESP32 outputs bit-identical | Nothing | No |
| 4 | Split provenance and freeze history | Manifests with SHA-256 per partition; overlap counts with the old split (§4b); timeline with freeze points and post-hoc additions (§4) | None; no validity problem found | Done (z9) |
| 5 | Resources and reproducibility | Static RAM / flash from the linker, arena, latency definition (§5); reproducibility map (§5) | Run-time heap (free / minimum free) not logged so far | z10 logs heap for both runtimes |
| - | Range experiments: simulation vs export | E3/E4 INT8 cells are **exported** TFLite INT8 models run in the TFLite interpreter; only the "QAT float" stage (Table 4, E1 `qat` rows) is fake-quant simulation. Failure/recovery pair verified in exports (E3 A vs B/D) and the failing min/max export also on both boards (§6) | Nothing | No |
| new | Runtime comparison | Old runtime results for the same files on the same board (z8b) | Current Espressif TFLM port | z10: ton_S80 (per-channel, failing), ton_S80_pt (per-tensor), ton_PQ80_int16 (16-bit) through both runtimes on the classic ESP32, identical files and inputs; records runtime versions, kernel config, op versions, outputs, decisions |

## 1. Operating points per run (E1, E5, E6)
Recall % / achieved test FAR % at the threshold chosen for a validation FAR budget of 1% (80 KB arms).

| dataset | arm | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| TON | F (FP32) | 96.00 / 0.68 | 95.77 / 0.66 | 98.27 / 1.18 |
| TON | F8 | 98.28 / 1.12 | 98.44 / 1.24 | 98.38 / 1.18 |
| TON | P | 96.42 / 0.91 | 95.47 / 0.95 | 97.32 / 1.00 |
| TON | QC | 98.37 / 0.91 | 97.68 / 1.06 | 97.54 / 1.06 |
| TON | R | 98.41 / 1.12 | 98.50 / 1.10 | 98.62 / 1.27 |
| TON | S80 | 98.17 / 0.85 | 98.03 / 0.93 | 97.00 / 1.16 |
| TON | SQC | 96.46 / 0.91 | 98.29 / 0.98 | 98.40 / 1.14 |
| CIC | F (FP32) | 97.82 / 0.98 | 94.54 / 0.98 | 98.32 / 0.99 |
| CIC | F8 | 99.33 / 0.98 | 99.61 / 0.93 | 99.45 / 0.90 |
| CIC | P | 98.87 / 0.98 | 99.04 / 0.85 | 99.35 / 0.98 |
| CIC | QC | 99.62 / 0.93 | 99.45 / 0.92 | 99.63 / 0.96 |
| CIC | R | 99.70 / 0.95 | 99.43 / 0.84 | 99.68 / 0.95 |
| CIC | S80 | 99.29 / 0.97 | 99.63 / 0.90 | 99.70 / 0.88 |
| CIC | SQC | 99.57 / 0.98 | 99.54 / 0.95 | 99.67 / 0.97 |
| TON hosts | F (FP32) | 98.17 / 4.77 | 98.12 / 0.00 | 98.32 / 4.77 |
| TON hosts | F8 | 98.18 / 0.00 | 97.87 / 0.00 | 98.18 / 0.00 |
| TON hosts | P | 93.99 / 0.00 | 96.42 / 0.00 | 93.94 / 0.00 |
| TON hosts | QC | 97.27 / 0.00 | 95.62 / 0.00 | 96.00 / 0.00 |
| TON hosts | R | 96.85 / 0.00 | 95.86 / 0.00 | 97.01 / 0.00 |
| TON hosts | S80 | 96.72 / 0.00 | 97.17 / 0.00 | 95.40 / 0.00 |
| TON hosts | SQC | 96.92 / 0.00 | 96.85 / 0.00 | 94.12 / 0.00 |

Q / SQ (QAT on unclipped inputs): TON 74.43 / 0.00 / 91.86 and 90.31 / 0.00 / 86.63; CIC 83.39 / 90.32 / 72.05 and
84.61 / 90.21 / 79.43 (recall per seed). Budget 5%: every arm except Q/SQ has recall >= 99.69 (TON) / 99.80 (CIC)
with test FAR 3.2-5.1%. Full rows (both budgets, 128 KB, validation FAR): `compare_arms_rows.csv` of jobs
z1d, z1g, z2b, z2e, z3b.

Validation budget vs achieved test FAR: on the random splits the achieved test FAR at the 1% budget lies in
0.66-1.27% for the principal arms (F, F8, P, QC, R, S80, SQC); it exceeds the budget on TON for F8, R and some
runs of F, S80, SQC (up to 1.27%), and stays at or below 0.99% on CIC. On unseen hosts the achieved test FAR is
**bimodal, 0.00% or 4.77%**: 121 of the 2,537 benign test flows (4.77%) come from one host cluster and flip
together, so the E6 test FAR is decided by that cluster, not by the threshold procedure.

Run-paired differences in recall (points; seeds 1 / 2 / 3), 1% budget:
| pair | TON | CIC | TON hosts |
|---|---|---|---|
| R - F | +2.41 +2.73 +0.35 | +1.88 +4.89 +1.36 | -1.32 -2.25 -1.31 |
| S80 - F | +2.17 +2.25 -1.26 | +1.47 +5.09 +1.38 | -1.45 -0.95 -2.93 |
| P - F | +0.42 -0.30 -0.95 | +1.05 +4.50 +1.03 | -4.18 -1.69 -4.38 |
| R - P | +1.99 +3.03 +1.30 | +0.84 +0.39 +0.33 | +2.86 -0.56 +3.07 |
| S80 - P | +1.75 +2.55 -0.31 | +0.42 +0.58 +0.34 | +2.74 +0.74 +1.45 |
| QC - P | +1.95 +2.21 +0.22 | +0.75 +0.41 +0.28 | +3.29 -0.81 +2.06 |
| R - S80 | +0.24 +0.47 +1.62 | +0.41 -0.20 -0.02 | +0.12 -1.30 +1.62 |

R - P is positive in all six random-split runs; S80 - P in five of six. On unseen hosts every compact model is
below the teacher in every run.

### 1b. Record-level paired comparisons (z9, `saha_checks.json` → `paired`)
Same test records, frozen thresholds, 1% budget; attacks detected only by A / only by B, exact two-sided
McNemar p. Recomputing recall and FAR from the saved scores reproduces every row of `compare_arms_rows.csv`
exactly for the INT8 arms; for the FP32 teacher at most 7 attack decisions differ (its scores are stored as
float16).
| pair | TON (s1 / s2 / s3) | CIC (s1 / s2 / s3) | TON hosts (s1 / s2 / s3) |
|---|---|---|---|
| R vs P | +352/-29, +514/-23, +219/-8 (p < 1e-54) | +712/-2, +521/-190, +298/-21 (p < 1e-35) | +661/-80, +116/-230, +714/-90 |
| S80 vs P | +304/-20, +440/-26, +260/-311 | +673/-312, +590/-93, +316/-24 | +648/-92, +184/-33, +738/-443 |
| QC vs P | +349/-33, +360/-2, +83/-47 | +640/-1, +402/-52, +275/-40 | +679/-11, +126/-290, +459/-41 |
| R vs S80 | +151/-112, +119/-42, +313/-51 | +384/-35, +99/-265, +31/-46 (p 0.11) | +88/-63, +30/-295, +487/-158 |
| R vs F | +404/-13, +449/-6, +63/-6 | +1667/-75, +4361/-204, +1250/-88 | +59/-329, +24/-483, +30/-296 |
| F8 vs F | +390/-20, +451/-18, +33/-15 | +1474/-198, +4448/-138, +1203/-237 | +57/-56 (p 1.0), +24/-74, +17/-45 |

On the random splits, R and QC detect significantly more attacks than P in every run (p <= 0.002), R more
than the teacher in every run, S80 more than the teacher in five of six (TON run 3: fewer); R vs S80 is mixed. False alarms move by tens (TON) to about two thousand (CIC)
records in each direction at the same 1% budget. On unseen hosts the teacher detects significantly more
attacks than R, S80 and P in every run, and every INT8 model avoids the same 121-flow benign cluster
the teacher flags in runs 1 and 3 (FA +0/-121).

## 2. Exact deployed models
`docs/model_hashes.csv`: file, bytes, SHA-256, weight-scale type and conversion (QAT export, or the
calibration selected by strong PTQ) for every principal export of Tables 2-3 and every file of the E7 bundle.
All exports declare the same operators and versions (QUANTIZE v1, FULLY_CONNECTED v4, LOGISTIC v2,
DEQUANTIZE v2), whether their Dense weights are per-channel or per-tensor: the old runtime accepts a
per-channel model at load time and then ignores its scales. Strong PTQ chose a per-channel TFLite export
(`tfl_max` / `tfl_clip5`) in TON P 2/3, S 2/3, F8 1/3; CIC P 1/3, S 2/3, F8 2/3; TON hosts P 1/3. All QAT
arms (Q, QC, R, SQ, SQC) are per-tensor by construction. z9 reports, for every F8/P/S run, the per-tensor
strong-PTQ replacement (same calibration draw and selection rule, per-tensor candidates only) and the two
flashed re-exports (ton_S80_pt, ton_P80_pt): recall, FAR, size, hash (§2b).

### 2b. Per-tensor replacements (z9, `saha_checks.json` → `exports`)
Strong PTQ restricted to per-tensor candidates (same 2,000-record calibration draw and selection rule).
Recall % at 1% FAR, mean ± SD over seeds (test FAR %), next to the Table 2 export:
| dataset | arm | Table 2 export | per-tensor replacement | size (KB) |
|---|---|---|---|---|
| TON | F8 | 98.37 ± 0.08 (1.18) | 98.35 ± 0.06 (1.19) | 188-189 |
| TON | P | 96.40 ± 0.92 (0.95) | 96.57 ± 1.89 (0.98) | 56-57 |
| TON | S80 | 97.73 ± 0.64 (0.98) | 97.57 ± 0.52 (1.01) | 56-57 |
| CIC | F8 | 99.46 ± 0.14 (0.94) | 99.33 ± 0.54 (0.90) | 206-207 |
| CIC | P | 99.09 ± 0.25 (0.94) | 99.12 ± 0.37 (0.92) | 65 |
| CIC | S80 | 99.54 ± 0.22 (0.92) | 99.57 ± 0.24 (0.97) | 65 |
| TON hosts | F8 | 98.08 ± 0.18 (0.00) | 98.14 ± 0.17 (0.00) | 189 |
| TON hosts | P | 94.79 ± 1.42 (0.00) | 93.81 ± 0.27 (0.00) | 57 |
| TON hosts | S80 | 96.43 ± 0.92 (0.00) | 96.48 ± 0.94 (0.00) | 56-57 |

Where the original selection was already per-tensor the replacement is the same model (identical metrics).
The largest single-run change is TON-hosts P run 2 (96.42 → 93.50) and TON P run 1 (96.42 → 97.98). The two
files flashed in E7: ton_S80_pt 97.67 / 0.93 (5%: 99.83 / 4.23), 57,184 B, SHA-256 766e6832…; ton_P80_pt
97.98 / 1.00 (5%: 99.78 / 4.11), 57,200 B, 477dfba8…. No conclusion of RQ1-RQ3 changes; the paper's Tables 2
and 3 now report the per-tensor (deployable) numbers.

## 3. E7 disagreements, resets and stream completeness
Decisions where device and host differ, at the frozen thresholds (bundle v2, identical on both boards):
| model | weights | FAR 1%: extra missed attacks / extra false alarms / fewer FA / synthetic | FAR 5% | disagreements outside near-threshold inputs (1% / 5%) |
|---|---|---|---|---|
| ton_R80 | per-tensor | 0 / 0 / 0 / 0 | 0 / 0 / 1 / 0 | 0 / 0 |
| ton_QC80 | per-tensor | 0 / 6 / 0 / 0 | 0 | 0 / 0 |
| ton_S80_pt | per-tensor | 0 | 0 / 0 / 4 / 0 | 0 / 1 |
| ton_P80_pt | per-tensor | 0 / 0 / 0 / 1 | 0 | 1 / 0 (synthetic) |
| cic_R80 | per-tensor | 0 / 11 / 9 / 0 | 0 / 13 / 4 / 0 | 0 / 1 |
| cic_S80 | per-tensor | 1 / 3 / 0 / 0 | 0 / 7 / 15 / 0 | 1 / 0 |
| cic_P80 | per-tensor | 0 / 23 / 1 / 0 | 0 / 19 / 6 / 0 | 1 / 2 |
| cic_QC80 | per-tensor | 0 / 382 / 1 / 0 (8 attacks recovered) | 0 / 1 / 1 / 0 | 2 / 0 |
| ton_S80 | per-channel | 2,069 / 0 / 23 / 161 | 49 / 0 / 89 / 55 | 1,272 / 87 |
| ton_P80 | per-channel | 929 / 2 / 10 / 157 (223 recovered) | 24 / 0 / 12 / 57 | 975 / 83 |
| ton/cic PQ80_int16 | per-channel, 16-bit | 933 / 0 / 12 / 134; 82 / 693 / 136 / 15 | 23 / 0 / 26 / 67; 38 / 139 / 400 / 29 | 923 / 90; 102 / 115 |

- Per-tensor INT8: across both datasets the device adds one missed attack (cic_S80, FAR 1%), and its other
  disagreements are false alarms raised on benign records that were selected because their host score lies
  within two output steps of a threshold; on the 1,500 random test records per model they disagree on at most
  2 records (0.13%; cic_QC80 at FAR 1%). The device errs toward alarming, never toward silence, except that one record.
- Resets (ESP32-S3 only; the ESP32 completed every stream without a reset). Job z8 used a host loop that
  moved to the next model after a reset: ton_QC80, ton_P80_pt and cic_R80 got no outputs and ton_S80_pt,
  cic_P80, cic_PQ80_int16 partial ones. All six were re-run in full in z8c with a loop that reconnects,
  reselects the model and resends the same input; z8c had one reset each in cic_R80, cic_P80 and
  cic_PQ80_int16, no input was skipped and no output is missing. The protocol is lockstep (one reply per
  input index, written once), so a resend cannot duplicate a record. The final S3 outputs are bit-identical to
  the ESP32's for all 14 models and 72,548 inputs, which also rules out lost or shifted records. The resets
  happened at different inputs in different runs and never twice at the same input: they are not input-driven.

## 4. Split provenance and freeze history (times UTC)
| time | event |
|---|---|
| 10-28 02:00-02:14 | v13 pipeline, training recipe, arms (F, F8, P, Q, R, S, SQ), budgets (80 / 128 KB, FAR 1% / 5%), threshold rule and strong-PTQ candidates committed, before any v13 result |
| 10-29 04:58 | audit results (old pipeline) |
| 10-29 05:36 | first v13 TON training (validation logs only) |
| 10-29 05:48 | **v13.1 rules** (drop impossible negatives; new test split excludes old test records) committed, before any v13 test-split result (the first, y2b, arrived 05:53 and is superseded) |
| 10-29 06:46-07:00 | z1b/z1c test results; validation FAR above budget revealed the float32 threshold bug |
| 10-29 07:24 | threshold fix (validation-side check), z1b/z1c re-run as z1d/z1e |
| 10-29 07:29 | E3/E4 design committed before any E3/E4 result |
| 10-29 15:58 | **post hoc** after E1-E4 test results: QC/SQC arms, E4 clipped-FP32 and 16-bit rows, E6 split |
| 11-05 00:57 | **post hoc** after device results: per-tensor re-exports of TON S80/P80 |

Separation: each run writes `<model>_manifest.{json,npz}` with the record ids and SHA-256 of every client
training slice, validation slice and the test split; identical hashes across all runs of a dataset and split
mode (checked by compare_arms before evaluation). Relationship to the old split: the new random test splits
contain **no** record of the old test split (re-derived by `fed_pipeline.old_test_ids`); new client
training/validation slices may contain old test records. Exploratory models of the old pipeline were trained
on its training pool, so records of the new test split were seen in training by those earlier models; no model
or threshold used in the paper was trained on them.

### 4b. Overlap with the old split (z9, `saha_checks.json` → `provenance`)
| split | new test: in old test / in old training pool | new client training: in old test | new client validation: in old test |
|---|---|---|---|
| TON random | 0 / 21,032 (all) | 18,927 of 75,716 | 2,105 of 8,412 |
| TON hosts | 0 / 22,866 (all) | 13,797 of 68,937 | 1,538 of 7,660 |
| CIC random | 0 / 499,038 (all) | 382,551 of 1,529,825 | 49,739 of 199,616 |

Every record of the new test splits was in the old pipeline's training pool, so the exploratory models of
the old pipeline (which guided the hypotheses, not the final arms) saw them in training. None of the
models, thresholds or calibration draws reported in the paper did: they were trained and selected only on
the new client slices, and the method settings were fixed before any new test result (§4). The old test
records now sit in the new training and validation slices, where no evaluation uses them. We therefore see
no concrete validity problem and no reason for another split.

## 5. Resources, latency definition, reproducibility
Memory, classic ESP32 test firmware (Arduino, TFLM 1.0.0, 14 models embedded): static RAM 77,612 B of
327,680 (linker), of which 49,152 B is the statically allocated tensor arena; the models use 1,440-1,824 B
of it. Non-arena static RAM: 28,460 B. Flash 1,418,881 B, of which 915,336 B are the 14 embedded models
(57-82 KB each); runtime + core + firmware code about 503 KB. Heap at run time (free and minimum free) is
logged from z10 on, for both runtimes. ESP32-S3: static RAM 74,828 B, flash 1,409,625 B.
Latency: `micros()` (old runtime) / `esp_timer_get_time()` (new) around `Invoke()` only. It includes the
graph's input QUANTIZE (float to int8), the three FULLY_CONNECTED layers, LOGISTIC and DEQUANTIZE. It
excludes the serial transfer, the copy of the float features into the input tensor (reported separately as
`quantize_input`, 3-13 µs) and any feature extraction from packets.
Reproducibility: configs `config/v13/*.yaml` and `config/jobs/<job>/`; seeds 1-3 (`federated.seed`); split
manifests; checkpoints (`train_fl/models/*.h5`, `compare_arms/s*/PQ_*.h5`, `R_*.h5`); exported models and
hashes (`docs/model_hashes.csv`); frozen thresholds and per-run metrics (`compare_arms_rows.csv`); saved
validation and test scores (`s*/val_scores.npz`, `test_scores.npz`); device bundles with inputs and host
outputs; firmware (`esp32_agreement/`, `esp32_agreement_idf/`); scripts `scripts/{train_fl_runs,
compare_arms,range_mechanism,device_agreement,run_e7_device,saha_checks}.py`.

## 6. Range experiments: simulation vs exported integer models
- E3 cells A-D and E4 INT8 / 16-bit / logit rows: each is an exported TFLite model (fake-quant ranges set,
  then converted by the TFLite converter) evaluated with the TFLite interpreter; nothing in E3/E4 is a
  fake-quant simulation. The `random0` exports are kept (`range_mechanism/s*/`).
- Fake-quant simulation appears only in the "QAT float" stage (E1 `qat` rows, Table 4 "QAT float").
- Failure/recovery pair: cell A (min/max input range) vs B/D (clipped) are both exported INT8 models; the
  failing min/max export (PQ80_minmax = cell A, random draw, seed 1) also ran on both boards, where it
  reproduces the host within at most 97 output steps and the same decisions on 1,448/1,500 (TON) and
  1,496/1,500 (CIC) random records; the recovered models on the boards are the per-tensor R80 / QC80 exports.
