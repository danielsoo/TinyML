---
name: lctes-revision-status
description: Full status of the LCTES'26 WIP paper revision — what was reviewed, what was changed, what data already exists, what is still missing, and exact next-step recipes. Read this before doing any further work on the TinyML / federated IDS paper.
type: project
updated: 2026-10-02
---

# LCTES'26 Paper #81 — Revision Status (read this first)

> 한 줄 요약(한국어): LCTES'26 WIP 논문(#81, "Reliable Federated TinyML Deployment for IoT Security")이 리뷰에서 떨어졌고(81A 약한 승인, 81B/81C/81D 약한 거부, meta-review 불합격), 받은 피드백을 전부 반영한 21페이지짜리 완성 초안(`TinyML_Federated_IDS_Revised.docx`)을 이미 만들어뒀습니다. 남은 일은 ESP32 실측, centralized baseline, non-IID 재학습 등 "새 실험" 몇 개입니다. 아래에 전부 정리했습니다.

## 1. Where things stand

- Original submission: LCTES'26 Work-in-Progress track, Paper #81, "Reliable Federated TinyML Deployment for IoT Security." Authors: Younsoo Park, Seokhyeon Bae (Penn State University). Supervisors: Dr. Peilong Li (Elizabethtown College), Dr. Suman Saha (Penn State).
- Outcome: **not accepted**. Reviews: 81A = weak accept, 81B = weak reject, 81C = weak reject, 81D = weak reject, plus a meta-review citing (1) incremental novelty / unclear contribution, (2) limited ML model, (3) limited evaluation (CIC-IDS2017 only, no on-device measurement). Full review text is in the chat history of the session that did this work (not re-pasted here for length — ask the user for it if needed, or see the "Note to reviewers" section at the top of the revised docx, which paraphrases every point).
- **A full revision has already been written and delivered**: `TinyML_Federated_IDS_Revised.docx` (21 pages), saved at `/Users/younsoopark/Documents/Privacy/Research/TinyML/TinyML_Federated_IDS_Revised.docx` on the user's machine, and also sent to the user via chat. This is the current canonical draft — treat it as the base to keep improving, not the original WIP PDFs.
- Target venue for the revision is **not yet decided** — the user said "일단 가장 완성도 높은 초안" (just want the strongest complete draft for now). Don't assume a specific page limit or template until the user names a venue.

## 2. What the revision already did (do not redo this)

Using data that was already sitting in the project's folders but never written into the paper, the revision:

- Decomposed the headline 12.28× compression ratio into distillation (~10.5×) / pruning (~2.2×) / INT8 PTQ (~2.5×) contributions, using the 48-configuration grid sweep (`TinyML-friend/sweep_results_3_4_2026 - sweep_results.csv`).
- Restructured the paper to lead with the finding that **training-time QAT helps at aggressive compression and hurts at moderate compression**, illustrated with a new figure (F1 vs. compression ratio, QAT on/off) built from the same 48-config sweep.
- Added a full adversarial-robustness table (FGSM / FGM / GA / PGD, 4 deployment configs) from `FGSM/fgsm_results.csv`, which already existed but was never tabulated in the submitted paper.
- Added a PGD adversarial-training section (mean adversarial accuracy 0.679 → 0.888 after AT, degrading to 0.798 after INT8 PTQ export) from `TinyML-friend/data/processed/pgd_at/pgd_at_report.md` and `pgd_at_summary.csv` — this was never mentioned in the submitted paper at all; it's a genuine addition.
- Expanded references from 8 to 23, all verified (either well-known papers or extracted directly from PDFs already sitting in `Papers/`), reorganized Related Work by topic.
- Added an explicit baseline-decomposition table (fixed-LR vs. cosine-LR, pre- vs. post-compression), using an internal ablation run (`run id 2026-02-05_12-52-17`) that was logged but never surfaced in the paper.
- Explained the "duplicate 14.44 KB row" the reviewers flagged as a possible error — it's real (two configs that differ only in training-time QAT share the same post-PTQ footprint) — and added this explanation to the paper instead of leaving it to be misread again.
- Moved the BatchNorm-folding derivation to an appendix (reviewer asked for this).
- Added a precision/false-positive framing of Attack Recall (46.7%→93.85% recall translates to 53.3%→6.15% missed-attack rate) to answer reviewer C's "why does 93.85% recall matter" question.
- Wrote a complete, ready-to-run ESP32 benchmark protocol as Appendix B (see §4 below — the firmware already existed, it was just never run on hardware).
- Added an itemized Limitations section (§3 below) that is explicit about every gap instead of a vague "future work" paragraph.

**Do not re-derive these numbers from scratch** — they're already cross-checked against source CSVs. If asked to revise the docx further, read it first (`pandoc -t markdown` or unzip per the docx skill) rather than regenerating from the build script below, unless the change is large enough to warrant a full rebuild.

### Key numbers to remember (cross-checked, cite instead of re-deriving)

| Metric | Baseline (fixed LR) | Final (cosine LR + compression) |
|---|---|---|
| Accuracy | 93.5% | 96.02% |
| F1 | 84.1% | 89.32% |
| Attack Recall | 46.7% | 93.85% |
| Model size | 0.78 MB | 0.0635 MB (12.28×) |
| Latency (TFLite interpreter, not physical device) | 1.89 ms | 0.48 ms (−74.5%) |

- focal loss α = 0.35 is the value that converges; α ∈ [0.85, 0.92] reliably collapses training to single-class prediction under federated + class-imbalanced conditions.
- Full QAT-during-training: 96.02% acc / 89.32% F1. Post-training QAT only: 96.45% / 91.32% (slightly better on this one operating point). PTQ alone F1 0.8215 vs. QAT+PTQ combined F1 0.6367 (combining hurts).
- 48-config sweep, no-training-time-QAT path: FP32 864.1 KB → distillation only 82.45 KB (10.48×) → +pruning 37.89 KB (22.81×) → +pruning+PTQ 14.85 KB (58.18×, most aggressive grid point — more aggressive than the headline 12.28×, which was chosen to preserve recall).
- Adversarial: the one deployment config trained with training-time QAT and *without* a final PTQ step ("Balanced – Accurate") is essentially untouched by FGSM/FGM/GA/PGD (ΔF1 ≈ 0%); every PTQ-only config loses 11–40% F1 under attack. This is the cleanest single piece of evidence that training-time QAT buys adversarial robustness.
- PGD-AT: mean adversarial accuracy over ε ∈ {0.01..0.3}: pre-AT 0.679 → post-AT (Keras) 0.888 → post-AT float32 TFLite (transfer attack) 0.804 → post-AT INT8 PTQ TFLite (transfer attack) 0.798. Most of the AT gain survives TFLite export; a further, smaller amount is lost to INT8 PTQ specifically.

## 3. What is still missing (in priority order, with exact recipes)

This is the actual to-do list. Each item below states whether it's an experiment (needs compute/hardware) or a research/judgment task (no compute needed).

### 3a. ESP32 physical on-device benchmark — EXPERIMENT, hardware needed, fastest to close
Status: firmware and log-parser are both complete and were never run on physical hardware (no USB/serial access was available in the Claude session that wrote this). Exact steps (already written into Appendix B of the docx):
1. `python scripts/deploy_microcontroller.py --model models/tflite/saved_model_pruned_qat.tflite --output esp32_tflite_project/src/model_data.c --array-name ids_tflite_model`
2. `pio run --target upload` (PlatformIO project at `TinyML-friend/esp32_tflite_project/`)
3. Firmware (`src/main.cpp`) prints `BENCHMARK latency_us=<us> arena_used=<bytes> input_dim=78` per inference over serial at 115200 baud (20 runs after 1 warmup). Default tensor arena is 120 KB — bump it if `AllocateTensors()` fails.
4. `python scripts/collect_esp32_benchmark.py --port <serial port> --runs 20` → writes `data/processed/ablation/esp32_benchmark.json` with mean/median latency.
5. Report alongside Table 3's interpreter latency (1.89 ms → 0.48 ms), noting the board/clock/TFLite-Micro version used.

### 3b. Centralized-only baseline — EXPERIMENT, lightweight, no GPU strictly required
This is row (a) of the baseline-decomposition table (Table 2 in the docx), currently marked "not run." It answers: *what does a single-node (non-federated) model get with the identical recipe?*
Recipe: single-node training, same MLP (512→256→128, BatchNorm+dropout), same cosine LR schedule, same focal loss (α = 0.35), same balance_ratio = 4.0, on CIC-IDS2017, no FL simulation (no client partitioning, no rounds). Epoch budget to match: the federated run used 60 rounds × 3 local epochs ≈ 180 epoch-equivalents — use that as the centralized epoch count, or at minimum sweep to the same wall-clock/compute budget. No FL simulation overhead means this should run much faster than any of the federated experiments in `TinyML-friend/data/processed/runs/`.
Raw data lives at `/Users/younsoopark/Documents/Privacy/Research/TinyML/drive-download-20260129T185431Z-3-001/` (8 CIC-IDS2017 CSVs, ~865 MB total, cloud-only on the user's Mac — stage/download before use). Preprocessing recipe is documented in `TinyML-friend/docs/PAPER_FULL_PROCESS_SECTIONS.md` §2.

### 3c. Non-IID (Dirichlet) client partitioning — EXPERIMENT, heavier, GPU recommended
Current partitioning (`src/federated/client.py` / `src/data/loader.py`) is label-distribution-aware but near-IID: every client gets a similar normal:attack mix. Reviewer 81B specifically asked for harder non-IID partitions, citing this as making 4 clients "effectively a larger batch."
Recipe: partition the training split across the 4 clients using a Dirichlet(0.3) distribution over the label (normal/attack), and if time allows, over attack sub-type too. Keep every other hyperparameter fixed (cosine LR, FedAvgM, α = 0.35, balance_ratio = 4.0, 60 rounds × 3 local epochs). Report Attack Recall and rounds-to-convergence against the current near-IID partition. This needs GPU time comparable to the original Vast.ai runs (RTX 4090) referenced in `Research_Progress_Report_TinyML_FL_IDS.md` — do not attempt this on a CPU-only sandbox; a prior attempt in a constrained cloud container (2–4 CPU cores, no GPU, no TensorFlow preinstalled) was judged infeasible in reasonable time and was explicitly not attempted.

### 3d. Cross-dataset generalization (Bot-IoT / TON_IoT) — EXPERIMENT, largest scope
Data loaders for Bot-IoT and TON_IoT already exist in `TinyML-friend/src/data/loader.py` for compatibility testing, but neither dataset has been carried through the full federated + compression + adversarial pipeline, and neither dataset's raw files are currently present locally (would need downloading). Bot-IoT's loader is more mature than TON_IoT's — do that one first. Recipe: repeat §5.1–5.3 of the revised paper (training-stability check, then compression sweep) on Bot-IoT, and check specifically whether (i) the cosine-LR recall improvement and (ii) the "pruning acts as a regularizer" effect (§5.6) replicate on a dataset with a different feature count (38 vs. 78) and attack-type distribution.

### 3e. Comparison to other published IDS systems — RESEARCH TASK, not an experiment
Reviewer 81A asked for a comparison against other baselines/IDS systems. This is a literature-search task: find published accuracy/F1/recall numbers for FL-based or centralized IDS systems evaluated on CIC-IDS2017 (same or comparable train/test split), and cite them with verified sources. **Do not fabricate or approximate competitor numbers from memory** — this was explicitly flagged in the revision as something to resolve carefully rather than guess at.

### 3f. Client-count scaling — EXPERIMENT, lower priority
Flagged by 81D as a separate concern from non-IID-ness: 4 clients is small for a real IoT deployment. Would mean re-running at, e.g., 20–50 simulated clients. Not started; lower priority than 3a–3c.

### 3g. Deployment false-positive-rate translation — JUDGMENT CALL, needs input from the user/advisor
§5.10 of the revised paper translates Attack Recall into a missed-attack rate (53.3%→6.15%) but not into an alerts-per-day false-positive rate, because that requires assuming a reference traffic volume for a specific deployment context. Needs a decision, not a computation, before it can be written.

## 4. Codebase map (device paths — this session worked through the `mcp__remote-devices__*` bridge to the user's Mac, not a git repo in the cloud container)

All paths below are under `/Users/younsoopark/Documents/Privacy/Research/TinyML/` on the user's Mac (the connected folder for this project). Many files there show as "cloud-only" (iCloud placeholders) and need staging/downloading before they can be read or edited.

- `TinyML_Federated_IDS_Revised.docx` — **the current paper draft. Start here.**
- `LCTES__IoT_.pdf`, `TinyML Research Paper.pdf`, `Research Paper - Google Docs.pdf` — the original WIP-submission drafts (superseded by the .docx above, kept for reference; the Google Docs export has a fuller Motivation/Related Work/Conclusion than the trimmed 2-column WIP PDF).
- `Research_Progress_Report_TinyML_FL_IDS.md` (889 lines) — detailed v1–v15 experiment history, hyperparameter search log, and the reasoning behind α = 0.35, balance_ratio = 4.0, cosine LR. Read this before changing any training hyperparameter.
- `TinyML-friend/` — the actual code repository.
  - `src/federated/{client.py, server.py}` — FL client/server (FedAvgM, cosine LR, focal loss, client partitioning — this is what 3c needs to change).
  - `src/data/loader.py` — CIC-IDS2017 / Bot-IoT / TON_IoT loaders and preprocessing.
  - `src/modelcompression/{distillation.py, pruning.py, quantization.py}`, `compression.py` — compression pipeline.
  - `run.py`, `scripts/run_sweep_and_pgd.py` — main pipeline and the 48-config grid sweep driver.
  - `sweep_results_3_4_2026 - sweep_results.csv` — the 48-config grid sweep results (compression decomposition + QAT-vs-compression figure source).
  - `data/processed/pgd_at/{pgd_at_report.md, pgd_at_results.csv, pgd_at_summary.csv, model_selection.json}` — PGD adversarial-training results.
  - `data/processed/fgsm_at/` — FGSM adversarial-training attempt (incomplete — only the config roles are recorded, no results; different from the top-level `FGSM/` folder below, which does have results).
  - `data/processed/runs/` — dated run directories, v1–v24, `VERSIONS.md` / `RUNS.md` indexes.
  - `docs/LCTES_IoT_paper_draft_models_and_citation.md` — Korean-language draft paragraph + citations for the "QAT isn't always better during training" finding (Nagel et al.).
  - `docs/PAPER_FULL_PROCESS_SECTIONS.md` — ready-to-paste methodology prose (preprocessing, FL setup, compression pipeline) with a checklist of numbers to keep consistent.
  - `esp32_tflite_project/` — PlatformIO project; `src/main.cpp` is the complete ESP32 benchmark firmware (see §3a).
  - `scripts/collect_esp32_benchmark.py` — serial-log parser for the ESP32 benchmark.
  - `models/distilled/*.h5` — trained distilled model checkpoints.
  - `.venv/`, `requirements.txt` — Python env (tensorflow, flwr<2, tensorflow-model-optimization, adversarial-robustness-toolbox, etc.).
- `FGSM/fgsm_results.csv` (top-level, **not** inside `TinyML-friend/`) — FGSM/FGM/GA/PGD attack results across 4 deployment configs (clean vs. adversarial F1/accuracy). This is the one used in the revised paper's Table 5.
- `Papers/` — reference PDFs already on hand and cited in the revision: Gholami et al. (quantization survey, arXiv:2103.13630), Nagel et al. (white paper, arXiv:2106.08295), Gorsline et al. / Song et al. / Ayaz et al. (adversarial robustness of quantized NNs — arXiv:2105.00227, 2012.14965, 2304.12829 respectively).
- `distillation_f1.png`, `pruning_accuracy.png` — the team's own figures, embedded in the revised docx as Figures 2–3.
- `drive-download-20260129T185431Z-3-001/` — raw CIC-IDS2017 CSVs (8 files, ~865 MB total), needed for 3b/3c.

### Reproducing / editing the .docx itself
The revised paper was generated with a docx-js script, **not written directly in Word** — this makes future edits easy if the build script is kept. It was built at `/home/claude/tinyml_paper/{build.js, main.js}` inside the Claude session's ephemeral cloud container, which does **not** persist after the session ends. If a future session needs to regenerate or heavily edit the document (not just a quick find-and-replace, which the docx skill's unzip/edit/rezip flow handles fine), it will need to rewrite this script from scratch unless it was separately saved — check whether a `paper_build/` folder exists under the TinyML root before assuming it needs to be rewritten.

## 5. Environment constraints learned the hard way (don't re-discover these)

- The `mcp__remote-devices__device_bash` sandbox (runs on the user's Mac, isolated Linux VM) has **no GPU, no USB/serial passthrough, 4 CPU cores, ~3.8 GB RAM, and no TensorFlow preinstalled**. It cannot run FL training at any realistic scale, and cannot reach a physical ESP32 board even if one is plugged into the Mac.
- The cloud container (plain `Bash` tool) has **2 CPU cores, ~7.8 GB RAM, no GPU**, and also cannot reach any physical hardware. It can install packages via pip/npm.
- **Conclusion: none of §3a/3c/3d can be executed by a Claude session's own sandboxed tools.** They require either the user's own machine with a real GPU (or the Vast.ai setup referenced in the progress report) for 3c/3d, or physical board access for 3a. A future session should not attempt to "just run" these in its own sandbox — it will not have the compute or hardware for it. §3b (centralized baseline) is the one exception that might be light enough to attempt in a sandbox if the raw data is staged first — but 865 MB of raw CSVs plus full preprocessing (deduplication, etc. — see `docs/PAPER_FULL_PROCESS_SECTIONS.md` §2) is itself nontrivial; budget time for that step.
