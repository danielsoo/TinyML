# Paper plan after the v13.1 re-run (Dr. Saha's RQ1-RQ3)

Source of every number: `docs/REVISION_RESULTS.md`, section "v13.1 re-run"; audit: `docs/AUDIT.md`.
Format assumed: LCTES (ACM sigplan, 10 pages + 2 pages references/appendix) until Dr. Saha confirms the venue.

## Working title
**When Does INT8 Compression Preserve Federated Intrusion Detection on Microcontrollers?
Input Ranges, Operating Points, and Client-Data Constraints**

## Abstract (draft, ~190 words)
Federated intrusion detectors are trained on server-class hardware but must run on
microcontrollers with tens of kilobytes of memory. We ask when compressing such a detector to
INT8 preserves its usefulness at a fixed false-alarm rate, why it fails when it does, and which
remedies work when only client data are available. On CIC-IDS2017 and TON_IoT, with a
data pipeline audited for client ownership and test isolation, thresholds chosen on client
validation data, and three independent federated runs, 57-67 KB INT8 models detect 98.5-99.6% of
attacks at 1% FAR, matching the float teacher; a compact model trained federatively from scratch is as good
as a pruned and distilled teacher. Failures have one dominant cause: the input quantization range.
Ten extreme records in a 2,000-record calibration draw drive recall to 0%, and QAT fails for the
same reason because it learns that range from unclipped inputs; bounding the input range (clipping
or extreme-free calibration) removes both failures. Hidden-layer ranges and the 8-bit output grid
matter little; the remaining loss comes from coarse INT8 activations near the operating point,
recovered by 16-bit activations at unchanged weight size. On unseen hosts the ranking holds. ESP32
and ESP32-S3 reproduce the host decisions on N stress inputs at X-Y ms per flow. *(E7 numbers pending)*

## Research questions -> evidence
| RQ | question | evidence (job ids) | paper element |
|---|---|---|---|
| RQ1 | When does compression preserve useful detection under memory (80 / 128 KB) and FAR (1% / 5%) budgets? | E1 F/F8/P/Q/R/S/SQ/QC/SQC, fed and 1-client fine-tuning (z1d, z1e, z1g, z2b, z2c, z2e); E5 three seeds; E6 unseen hosts (z3b) | Table 2 (main E1 table, both datasets); Table 3 (E6) |
| RQ2 | Why does it fail? | E2 decision transitions by family (compare_arms_transitions.csv); E3 range 2x2 x calibration draws (z1f, z2d); E4 output / activation resolution (z1h, z2f); QAT loss located before conversion | Fig. 1 (E3: recall per cell and draw type, both datasets); Table 4 (E2 lost / recovered by family); Table 5 (E4 rows) |
| RQ3 | Which remedies work under memory and client-data constraints? | clipped / extreme-free calibration (E3 B, D, "no extremes"); QC/SQC vs Q/SQ; R vs P; S vs P/R; 16-bit activations; 1-client vs federated fine-tuning; E7 device agreement and cost | Table 6 (remedy, accuracy effect, memory / latency cost); Table 7 (E7 device agreement, latency, arena) |

## Contributions
1. An audited federated-to-MCU evaluation protocol: client-owned partitions before any fitted
   step, real per-client validation slices, thresholds chosen at a FAR budget on validation and
   applied once, a test split disjoint from every record used during development, three seeds,
   and an unseen-host split (docs/AUDIT.md lists the problems it fixes in common practice: SMOTE
   before partitioning, synthetic validation records, overflowed CIC fields).
2. RQ1: at 57-67 KB, INT8 models match the float teacher at 1% FAR on both datasets; a compact
   model trained federatively from scratch is a strong baseline that compression must beat;
   fine-tuning with one client's 10k records is enough.
3. RQ2: a controlled 2x2 shows the input range, not hidden ranges, causes the collapses; the same
   mechanism explains the QAT failure; output-grid effects are small, and the residual loss is
   tie groups from coarse INT8 activations at the operating point.
4. RQ3: remedies with their cost: input clipping / extreme-free calibration (free), QAT on clipped
   inputs (free), 16-bit activations (same flash, larger arena), distillation (+0.6 points on TON).
5. Bit-exact device agreement on thousands of stress inputs on two MCU cores, with a memory and
   latency breakdown.

## Section outline and page budget (10 pages)
| § | content | pages | from current main.tex |
|---|---|---|---|
| 1 | Introduction: problem, RQ1-RQ3, contributions | 1.2 | rewrite |
| 2 | Background and related work: FL IDS, TinyML IDS, PTQ calibration (ACIQ, percentile, KL), QAT ranges (PACT, LSQ), distillation | 0.9 | condense §2 |
| 3 | Experimental design: datasets and audited pipeline (Table 1: dataset / audit summary), federated training, arms F/F8/P/Q/QC/R/S/SQ/SQC, budgets, FAR-threshold protocol, splits, seeds | 1.5 | rewrite §3 |
| 4 | RQ1: Table 2 (E1), Table 3 (E6), fine-tuning data constraint | 1.5 | replaces §4.2, §4.7, §4.8 |
| 5 | RQ2: Fig. 1 (E3), Table 4 (E2 by family), Table 5 (E4), QAT failure location | 2.0 | replaces §4.3, §4.4, §4.5 |
| 6 | RQ3: Table 6 remedies and cost | 1.0 | new |
| 7 | On-device agreement and cost: Table 7 (E7) + latency on ESP32 / ESP32-S3 | 0.7 | rewrite §4.10 |
| 8 | Threats to validity: two datasets, binary task, one model family, TON test FAR noise on few hosts, operating-point sensitivity of the FP32 teacher | 0.5 | rewrite §6 |
| 9 | Conclusion | 0.3 | rewrite |
| | figures / tables slack | 0.4 | |
| App. A | Audit table (D1-D7, C1-C3, N1-N5) | 0.6 | docs/AUDIT.md |
| App. B | Toolchain pitfalls: the four silent failures plus the NumPy float32 threshold / INT8 tie pitfall found in this re-run | 0.5 | condense §5 |
| App. C | BN folding, hyper-parameters, artifact map (job ids) | 0.4 | §A + PROVENANCE.md |

## Cut or moved
- §4.1 "where federation helps" (FL vs local-only): one sentence in §3; not an RQ.
- §4.3-4.4 calibration estimator sweeps on the old pipeline: superseded by E3; estimators survive
  as the candidates of "strong PTQ" (P).
- §4.5 QAT inside FL, §4.6 client-local distillation students, §4.7 combination table, §4.8
  recall-priority deployment: superseded by the E1 arms and the FAR-budget protocol.
- §4.9 adversarial robustness and App. B "how the TON recipe was chosen": out of scope (Dr. Saha:
  defer a new adversarial study).
- All old-pipeline numbers: not reported as results; App. C states they were exploratory and why
  (AUDIT.md).

## Open items
- Venue and format (Dr. Saha), authorship, what to do with the arXiv version.
- E7 numbers (runs on the two boards; esp32_agreement + scripts/device_agreement.py).
- Optional, cheap if wanted: a non-IID (Dirichlet) partition for TON with the same arms (~1 h of
  PC time); a fourth/fifth seed if a reviewer asks about variance at FAR 1% (the FP32 teacher's SD
  is 1.4-2.1 points).
- Figures: Fig. 1 (E3 bars) and the remedy/cost table are not drawn yet.
