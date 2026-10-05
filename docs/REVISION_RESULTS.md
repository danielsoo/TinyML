# Revision experiment results (running log)

> ⚠️ **2026-10-03: two pipeline bugs found; every number below this box from before the fix is invalid
> as a federated result.** See "Bugs found" at the end. Rerun: job `2026-10-03_v2_cic_full`.

All runs: paper recipe `config/paper_v12/` (60 rounds × 3 local epochs, lr 1e-3 cosine → 1e-4,
focal α = 0.35, FedAvgM, full CIC-IDS2017, dedup, 80/20 stratified split, balance_ratio 4.0, SMOTE),
test split, threshold 0.3, run on the user's PC (WSL2, 16 cores, CPU).

## Run 2026-10-03_01-46-48_full (no tuning)
Source: `data/processed/revision/2026-10-03_01-46-48_full/SUMMARY.md`

| Row | Model | Acc | Prec | Attack recall | F1 | False-alarm rate | Size |
|---|---|---|---|---|---|---|---|
| (a) | Centralized, same recipe, 180 epochs (Keras FP32) | 95.00% | 77.36% | 99.91% | 87.20% | 6.00% | — |
| (b) | Federated near-IID, pre-compression (Keras) | 90.55% | 64.77% | 97.57% | 77.86% | 10.90% | — |
| (c) | (b) + prune 50% + fine-tune + INT8 PTQ | 96.34% | 87.06% | 92.27% | 89.59% | 2.82% | 75.1 KB |
| (d) | (b) + prune 50% + fine-tune + QAT → INT8 (deploy recipe) | **96.61%** | 87.05% | **94.12%** | **90.45%** | 2.88% | 65.4 KB |
| non-IID | Federated Dirichlet(0.3), 4 clients, pre-compression | 84.91% | 54.00% | 77.16% | 63.54% | 13.50% | — |

Wall time: baseline ablation (a–d) 322 min, non-IID 155 min.

### Reading
1. **Headline reproduces.** Row (d) re-run from scratch with the documented recipe gives
   96.61 / 90.45 / 94.12 (acc / F1 / recall) vs. the paper's 96.02 / 89.32 / 93.85 — no tuning involved.
2. **Centralized vs. federated (pre-compression):** F1 87.2 vs 77.9. Federation costs ~9 F1 points,
   almost entirely through precision (false alarms 6.0% → 10.9%); recall is ≥ 97.5% for both.
3. **Non-IID (Dirichlet 0.3):** F1 77.9 → 63.5, recall 97.6 → 77.2, false alarms 10.9 → 13.5%.
   Partition is extreme (one client holds 73% of samples; two clients have < 3% attacks), and the
   per-round federated evaluation oscillates (recall 0.77–0.97 between rounds 40–60) where the near-IID
   run rises steadily (0.84 → 0.98). This is a genuine result for reviewer 81B, not a broken run.
4. **Open question — compression gain:** (b) → (d) raises F1 77.9 → 90.4. `compression.py` fine-tunes the
   pruned model (3 ep) and the QAT model (2 ep) on the first 10,000 samples of the *pooled* training set,
   which a federated server would not hold. The paper (§5.6) attributes the gain to pruning acting as a
   regularizer; the gain may instead come from this server-side fine-tuning. Job
   `2026-10-03_compression_ablation` separates the two (no fine-tune / fine-tune only / prune + fine-tune,
   pooled vs. one client's local data). Paper text on §5.6 and Table 2 should wait for it.

## Second dataset: TON_IoT (queued)
Job `2026-10-03_toniot_full` runs the identical recipe (`config/paper_v12_toniot/`) on
`train_test_network.csv`. Loader changes (`load_ton_iot`) so results are not shortcut-driven and
match the CIC-IDS2017 preprocessing: drop `ts`, `src_ip`, `dst_ip`, `src_port`; label-encode text
columns with ≤ 50 values; drop high-cardinality free text (DNS query, URI, user agent, SSL subject…);
deduplicate; stratified 80/20 split; same undersampling / scaling / SMOTE. The first run's log lists
the kept features for review.

## Bugs found (2026-10-03) — affect all earlier runs and the paper's compressed-model numbers

1. **FL int8 communication (use_qat: true).** `FlowerClient._quantize_weights` sent raw int8 codes
   without their scales; the server averaged the codes as if they were weights, so every tensor was
   rescaled to max |w| = 127 each round (unit check: [0.10, −0.50, 0.25, 0.02] arrives as
   [25, −127, 64, 5]). Saved FL models show every tensor at exactly ±127 / ±126.0 and all QAT
   quantizer ranges at ±127. Fix: send the int8-rounded values dequantized with the sender's scale.
2. **QAT stripping in compression (since the first commit).** `compression.strip_qat_layers` copied
   weights positionally from `wrapper.layer.weights`, which lacks the kernel; every `set_weights`
   failed and the log printed `QAT layers stripped manually (0/6 layers with weights)`. Every
   compressed model therefore started from **random initialisation** and was then trained for
   3 + 2 epochs on the first 10,000 pooled training samples. Evidence: compression ablation `fp32`
   variant = 17% accuracy (all-attack) for a model that scores 90.6% unstripped; reproduction on a
   trained model gives correlation −0.01 between QAT and stripped outputs. Fix: copy by variable
   name, raise if nothing is copied; after both fixes the stripped model matches the QAT model
   (corr 0.999).

Consequences: the headline compressed results (paper Table 3, 96.02 / 89.32 / 93.85, and the
reproduction 96.61 / 90.45 / 94.12 above) are a small centrally trained model, not the federated
model; §5.6's "pruning improves recall" and the training-time-QAT findings built on QAT-trained
models need to be re-derived. Model sizes / compression ratios and the ESP32 latency benchmark
(architecture-only) are unaffected. The `2026-10-03_compression_ablation` job ran on the buggy
strip and is superseded.

Open design decision: compression fine-tuning still uses 10k pooled training samples (server-side
data). Options: fine-tune on one client's local data, no fine-tuning, or keep pooled and disclose it
as a server-side proxy set. `scripts/compression_ablation.py` reports all of these once v2 models exist.

## Run 2026-10-03_v2_cic_full (both bugs fixed, training-time QAT on)

| Row | Model | Acc | Prec | Attack recall | F1 | FAR | Size |
|---|---|---|---|---|---|---|---|
| (a) | Centralized (Keras FP32) | 95.74% | 80.01% | 99.95% | 88.87% | 5.13% | — |
| (b) | FL near-IID, QAT Keras model as trained | 34.74% | 20.70% | 99.97% | 34.30% | 78.66% | — |
| (b') | same FL weights, QAT stripped to float (Keras evaluate @0.5) | 89.40% acc | | | | | |
| (c) | FL → prune → FT (pooled 10k) → PTQ | 91.96% | 69.01% | 95.88% | 80.26% | 8.84% | 75.1 KB |
| (d) | FL → prune → FT (pooled 10k) → QAT → INT8 | 94.85% | 81.04% | 91.08% | 85.76% | 4.38% | 65.4 KB |
| non-IID | Dirichlet(0.3), QAT Keras model | 18.06% | 17.21% | 100% | 29.37% | 98.77% | — |

Fix check: compression log now reports `4 layers with weights copied`; FL weights are |w| ≲ 10 (no ±127).

**New finding — training-time QAT does not work on this data.** The QAT model's learned INT8 input
range is [−1231, 405] (activations up to 2342): standardized CIC-IDS2017 features are heavy-tailed,
and the moving-average min/max quantizers cover the outliers. One INT8 step ≈ 6.4 standard deviations,
so ordinary inputs collapse to the zero point and the QAT model outputs ~constant (34.7% acc), while
the same weights in float reach 89.4%. The earlier "QAT" results were produced by the bugs above,
not by working QAT. PTQ calibration (min/max on representative data) is exposed to the same outliers.

Next jobs: `2026-10-03_a_v2_compression_ablation` (float FL model accuracy at threshold 0.3; pooled vs
client-local fine-tuning) and `2026-10-03_b_v3_float_cic` (same recipe, `use_qat: false`). TON_IoT
config switched to `use_qat: false` for the same reason. Open: robust feature scaling (clipping or
log transform before standardization) so INT8 ranges are not set by outliers — to be chosen on the
validation split.

## Compression ablation on v2 models (`2026-10-03_a_v2_compression_ablation`)
Test split, threshold 0.3. FT = fine-tuning on 10k samples (pooled = first 10k of the pooled training
set, 20% attack; client = 10k from one client's partition).

| Model | fp32 (FL weights, float) | prune, no FT → PTQ | FT only → PTQ (pooled / client) | prune+FT → PTQ (pooled / client) | prune+FT → QAT (pooled / client) |
|---|---|---|---|---|---|
| near-IID FL | F1 46.8 (acc 61.5) | 30.8 | 80.6 / 72.7 | 80.3 / 70.4 | 82.7 / 74.6 |
| Dirichlet FL, FT client 1 (60% attack) | 31.0 (acc 24.1) | 37.2 | 86.5 / 73.6 | 81.6 / 77.6 | 87.9 / 80.4 |
| Dirichlet FL, FT client 3 (25% attack) | 31.0 | 37.2 | 86.5 / 82.9 | 81.6 / 81.7 | 87.9 / 87.6 |

Reading:
1. With training-time QAT on CIC-IDS2017, the FL model itself is poor both as trained (QAT, 34.7%)
   and as float weights (F1 46.8 / 31.0 at threshold 0.3). Every usable compressed CIC model owes its
   accuracy to the post-hoc fine-tuning, not to federated training — the non-IID model even ends up
   *better* than near-IID after fine-tuning.
2. Pruning without fine-tuning destroys the model; pruning does not act as a regularizer here (§5.6
   is not supported).
3. Client-local fine-tuning works only when that client's class mix resembles the global one
   (client 3, 25% attack: F1 87.6 ≈ pooled 87.9; client 0, 50% attack: 74.6). Pooled fine-tuning is
   server-side data and must be disclosed if used.
→ The float-FL rerun (`2026-10-03_b_v3_float_cic`) is the decisive experiment for CIC-IDS2017.

## TON_IoT (`2026-10-03_toniot_full`, use_qat: true, test split, threshold 0.3)
Preprocessing log: 211,043 rows → 105,176 after dedup (normal 24,106 / attack 81,070); dropped
src_ip, dst_ip, src_port (no `ts` column in this file version); 37 features. Runtime 24 min.

| Row | Model | Acc | Prec | Attack recall | F1 | FAR |
|---|---|---|---|---|---|---|
| (a) | Centralized | 99.31% | 99.39% | 99.72% | 99.55% | 2.05% |
| (b) | FL near-IID (QAT Keras) | 98.22% | 98.96% | 98.72% | 98.84% | 3.48% |
| (c) | FL → prune → FT (pooled) → PTQ | 98.96% | 99.02% | 99.64% | 99.33% | 3.32% |
| (d) | FL → prune → FT (pooled) → QAT | 98.91% | 98.99% | 99.59% | 99.29% | 3.40% |
| non-IID | Dirichlet(0.3) FL | 97.41% | 97.63% | 99.05% | 98.33% | 8.09% |

Training-time QAT works on TON_IoT (FL QAT model 98.2% as trained), unlike CIC-IDS2017 — consistent
with the heavy-tail diagnosis. Non-IID costs little in F1 but more than doubles false alarms
(3.5% → 8.1%). Queued: TON_IoT compression ablation and a use_qat: false run for symmetry with v3.

## v3: CIC-IDS2017 with training-time QAT off (`2026-10-03_b_v3_float_cic`) and TON_IoT float (`2026-10-04_d_toniot_float`)

| Dataset | Row | Model | Acc | Prec | Recall | F1 | FAR |
|---|---|---|---|---|---|---|---|
| CIC | (a) | Centralized | 95.07% | 77.60% | 99.90% | 87.35% | 5.92% |
| CIC | (b) | **FL near-IID (float)** | 94.32% | 75.01% | 99.94% | **85.70%** | 6.84% |
| CIC | non-IID | FL Dirichlet(0.3) (float) | 94.27% | 74.91% | 99.83% | 85.59% | 6.87% |
| CIC | (c) | FL → prune → FT → PTQ (old BN folding) | 22.03% | — | — | 30.41% | 93.98% |
| CIC | (d) | FL → prune → FT → QAT (old BN folding) | 96.90% | 87.45% | 95.51% | 91.30% | 2.81% |
| TON | (a) | Centralized | 99.20% | 99.32% | 99.64% | 99.48% | 2.28% |
| TON | (b) | FL near-IID (float) | 99.08% | 99.18% | 99.62% | 99.40% | 2.76% |
| TON | non-IID | FL Dirichlet(0.3) (float) | 97.63% | 98.92% | 97.99% | 98.45% | 3.59% |
| TON | (c)/(d) | compressed (old BN folding) | 80.58 / 89.31% | | | 88.81 / 93.50% | 84.7 / 45.7% |

Reading: **with QAT off, federated training is healthy**: CIC FL is within 1.7 F1 of centralized, and
the Dirichlet(0.3) partition costs almost nothing on CIC (85.59 vs 85.70) and ~1 F1 on TON_IoT. The
earlier "non-IID collapses" result was an artifact of the communication bug.

**Bug 3 — BatchNorm folding.** `make_mlp` is Dense(ReLU) → BatchNorm → Dropout, but the TFLite/QAT
export helpers folded each BN into the Dense *before* the ReLU, which is not equivalent (max output
error 0.86, 11.6% label flips on the v3 FL model). This corrupted every TFLite export of a BN model
(FP32 "original" exports, PTQ exports, (c) above). Fixed by folding each BN into the *next* Dense
(`fold_batchnorm`; max error ≤ 2e-5 on v3/TON models).
**Bug 4 — pruning drops BN statistics.** `apply_structured_pruning` re-creates BN layers from config
(fresh γ=1, β=0, μ=0, σ²=1), so pruned BN models fell to 32.7% before fine-tuning. Fixed by folding
BN before pruning. FL training is unaffected by bugs 3–4, so v3/TON float models are reused:
job `2026-10-04_e_float_compression_ablation` re-compresses them.

TON_IoT compression ablation on the QAT-trained models (`2026-10-04_c_…`, valid: no BN) — the FL
QAT model as float is weak (F1 91.0, FAR 66%) and the deployable models again come from fine-tuning
(pooled 99.3–99.4; client-local 98.7–99.2).

## Float FL models re-compressed with all fixes (`2026-10-04_e_float_compression_ablation`)
Test split, threshold 0.3; F1 / FAR in %. Log confirms BN folding (4 Dense layers) and the fp32
rows reproduce the Keras FL numbers exactly (CIC near-IID 85.70, TON 99.40).

| Model | fp32 | INT8 PTQ only (227/206 KB) | prune, no FT | prune+FT+QAT, **client-local** (65/55 KB) | prune+FT+QAT, pooled (65/55 KB) |
|---|---|---|---|---|---|
| CIC centralized | 87.35 / 5.92 | 83.82 / 7.85 | 42.05 | 81.84 / 8.84 | 87.84 / 3.32 |
| CIC FL near-IID | 85.70 / 6.84 | 84.75 / 7.37 | 42.28 | **85.59 / 6.82** | 90.72 / 2.98 |
| CIC FL Dirichlet (FT client 3) | 85.59 / 6.87 | 85.14 / 7.12 | 51.27 | **91.08 / 2.75** | 87.52 / 4.98 |
| TON centralized | 99.48 / 2.28 | 98.11 / 2.18 | 86.65 | 98.94 / 2.47 | 99.11 / 4.56 |
| TON FL near-IID | 99.40 / 2.76 | 98.38 / 2.70 | 96.78 | **98.74 / 2.88** | 99.17 / 3.28 |
| TON FL Dirichlet (FT client 2) | 98.45 / 3.59 | 97.85 / 7.03 | 87.03 | **98.72 / 3.11** | 99.13 / 3.19 |

Reading:
1. **INT8 PTQ works on float FL models** (−0.5 to −1.1 F1 at 3.5×): heavy tails break *training-time*
   QAT ranges, not post-training calibration of a float model. No robust-scaling change is required
   for the PTQ/QAT-fine-tune pipeline.
2. **FL-faithful compression is lossless at 12.3×**: prune 50% → fine-tune 3 ep + QAT 2 ep on one
   participating client's own data → INT8 (65.4 KB vs 802.5 KB) keeps the FL model's accuracy
   (CIC 85.59 vs 85.70; TON 98.74 vs 99.40; non-IID 91.08 / 98.72). Outcome depends on the chosen
   client's class mix (CIC centralized model with a 50%-attack client: 81.84), which must be stated.
3. Pooled (server-side) fine-tuning is an upper bound, not an FL result; its extra gain over fp32 on CIC
   is mostly re-calibration toward precision (FAR 6.8 → 3.0).
4. Pruning without fine-tuning is destructive (42–97 F1); §5.6's "pruning as regularizer" is not
   supported.
5. 802.5 / 65.4 KB = 12.27× — the paper's 12.28× ratio survives with the corrected pipeline.

Proposed paper pipeline: float FL (FedAvgM, cosine LR, focal α 0.35) → server folds BN → prune 50% →
one client fine-tunes (3 ep) + QAT fine-tunes (2 ep) on local data → INT8 TFLite (65 KB) → ESP32.
Still to re-run on the final models: FGSM/PGD robustness (old robustness numbers used the buggy
compressed models) and the ESP32 benchmark with the new deploy model.

## Robustness of the final models (`2026-10-04_f_robustness`)
Accuracy on 20k test samples, ε = 0.1 (standardized space), perturbations from the float FL model.

| Model | CIC clean / FGSM / PGD | TON clean / FGSM / PGD |
|---|---|---|
| FL FP32 (source) | 94.3 / 68.6 / 55.6 | 99.1 / 23.0 / 23.0 |
| Centralized | 95.1 / 30.8 / 29.0 | 99.2 / 23.0 / 23.0 |
| INT8 PTQ only | 93.9 / 70.4 / 60.4 | 97.5 / 23.0 / 23.0 |
| prune → client FT → PTQ | 94.8 / 32.5 / 23.9 | 86.1 / 23.0 / 23.0 |
| prune → client FT → QAT (deployed) | 94.2 / 44.3 / 45.6 | 98.1 / 23.0 / 23.0 |
| prune → pooled FT → QAT | 96.3 / 44.7 / 46.2 | 98.7 / 56.2 / 55.6 |

PTQ keeps robustness; pruning+FT lowers it; QAT fine-tuning is more robust than PTQ after pruning;
centralized is more fragile. On TON_IoT ε = 0.1 drives nearly all models to "all benign" (23.0% =
benign share) — the unconstrained L∞ budget is too large for this feature space. Added to the paper
as Table 5 with caveats (unconstrained perturbations, single seed).

## Pruning-ratio sweep (`2026-10-04_g_prune_sweep`, client 0 fine-tuning)
CIC (F1 client-FT QAT / PTQ, size): 30% 87.63/89.68 (112/126 KB) · 50% 85.59/86.90 (65/75 KB) ·
70% 84.42/84.97 (31/37 KB) · 85% 81.25/82.86 (14/16 KB) · 90% 78.02/77.90 (9.9/11 KB); no FT collapses
from 50%. TON: client-FT QAT 98.94 → 98.67 from 30% to 90% (98/7.9 KB); client-FT PTQ unstable
(90.5–98.4); no FT degrades. Figure: paper/figures/prune_sweep.png (scripts/plot_prune_sweep.py).

## Fixed LR on TON_IoT (`2026-10-04_h_fixed_lr_toniot`)
99.02 acc / 99.36 F1 / 99.73 recall / 3.38% FAR vs cosine 99.40 F1 — no cosine-LR effect on TON_IoT.

## Fixed LR on CIC-IDS2017 (`2026-10-04_i_fixed_lr_cic`)
95.01 acc / 77.40 prec / 99.84 recall / 87.20 F1 / 5.99% FAR vs cosine 85.70 F1. Fixed LR is equal or
better on both datasets → the WIP claim "cosine LR raises Attack Recall 46.7% → 93.85%" is withdrawn
(artifact of the weight-exchange bug). Paper Table 1 + paragraph updated.

## Quantization methods + client-local distillation (`2026-10-04_j_quant_distill`)
Calibration / fine-tuning / student training all use client 0's data only. F1 (size):

| Method | CIC federated | CIC pruned50+clientFT | TON federated | TON pruned50+clientFT |
|---|---|---|---|---|
| fp32 | 85.70 (802 KB) | 89.39 (243) | 99.40 (721) | 99.35 (202) |
| dynamic range | 85.58 (216) | 90.13 (69) | 99.40 (196) | 99.35 (59) |
| float16 | 85.70 (404) | 89.35 (124) | 99.40 (363) | 99.35 (103) |
| int8 full-integer | 85.02 (227) | 86.90 (75) | **91.24** (206, FAR 14.3%) | **90.50** (65, FAR 14.0%) |
| int16x8 | 85.48 (238) | 89.61 (81) | 99.40 (217) | 99.35 (70) |

8-bit activations are the only lossy step (int16x8 with the same int8 weights recovers fully). TON int8
PTQ: 98.38 with 500 pooled calibration samples (Table 3) vs 91.24 with 500 client-0 samples →
calibration-sensitive; job l (`2026-10-04_l_ptq_calibration`) isolates source/size/op set.
TFLM on ESP32: fp32 + int8 only.

Students (width ×1/2, ×1/4, ×1/8; F1 fp32 / int8 PTQ / int8 QAT):

| Student | CIC KD | CIC scratch | TON KD | TON scratch |
|---|---|---|---|---|
| 1/2 | 87.90 / 87.64 / 87.36 | 89.75 / 89.36 / 87.21 | 99.35 / 99.31 / 98.90 | 99.35 / 97.06 / 98.93 |
| 1/4 | 87.31 / 87.37 / 86.42 | 88.64 / 88.74 / 86.46 | 99.35 / 99.24 / 99.01 | 99.36 / 98.75 / 98.98 |
| 1/8 | 86.30 / 86.07 / 86.55 | 88.29 / 88.51 / 91.10* | 99.16 / 99.16 / 98.82 | 99.14 / 98.70 / 98.81 |

\* recall 96.32%, 3,137 missed attacks — outlier, not selected. KD students copy the teacher's
recall-heavy operating point on CIC (FAR 5.6–6.5%), scratch students trade recall for precision.
TON: KD makes PTQ robust (≤0.11 F1 loss); 1/8 KD student = 99.16 F1 at 11.2 KB.
Key finding: a scratch student on one near-IID client's data matches/beats the FL model on CIC →
the value of FL needs the local-only baseline under Dirichlet (job k, `2026-10-04_k_local_only`).
Paper: Sections 5.9 (Table 6) and 5.10 (Table 7).

## INT8 PTQ calibration sensitivity (`2026-10-04_l_ptq_calibration`)
45 full-integer PTQ conversions per model (5 sources × 3 sizes × 3 draws), plus the two single draws
from Tables 3/6. Op set (BUILTINS vs BUILTINS_INT8) has no effect. Source has no effect (median F1 per
source: TON 98.97–99.30, CIC 83.1–83.8). The draw has a huge effect, and more data is worse:

| n | TON median / min / within 1 F1 of FP32 | CIC median / min / within 1 F1 |
|---|---|---|
| 100 | 99.31 / 90.99 / 12 of 15 | 84.60 / 81.47 / 5 of 15 |
| 500 | 99.30 / 26.19 / 11 of 15 | 83.11 / 80.91 / 2 of 15 |
| 2000 | 91.23 / 12.28 / 3 of 15 | 82.60 / 29.16 / 1 of 15 |

→ The 5.9 claim "pooled vs client calibration" was wrong; it is draw-to-draw instability (likely
min/max ranges stretched by heavy-tailed outliers). Paper 5.9 rewritten + Table 7. Follow-up job m
(`2026-10-04_m_qat_stability`): QAT over 5 FT draws, clipped calibration, activation maxima.

## Local-only baseline (`2026-10-04_k_local_only`, 10 epochs, ≤200k samples per client)
| Partition / client | data (attack) | local F1 / recall / FAR (missed) | FL F1 / recall / FAR (missed) |
|---|---|---|---|
| CIC near-IID 0–3 | 681k (50%) | 88.3–90.0 / 98.85–99.78 / 4.5–5.2 (185–982) | 85.70 / 99.94 / 6.84 (52) |
| CIC Dir c0 | 9.4k (0.1%) | 62.04 / 44.97 / 0.00 (46,872) | 85.59 / 99.83 / 6.87 (144) |
| CIC Dir c1 | 1.98M (59.5%) | 90.70 / 99.75 / 4.15 (214) | |
| CIC Dir c2 | 6.7k (2.6%) | 88.74 / 91.11 / 2.92 (7,572) | |
| CIC Dir c3 | 728k (25.1%) | 89.80 / 99.60 / 4.56 (339) | |
| TON near-IID 0–3 | 32k (50%) | 99.16–99.31 / 99.65–99.79 / 3.5–4.8 (34–57) | 99.40 / 99.62 / 2.76 (61) |
| TON Dir c0 | 49.9k (0.4%) | 90.16 / 82.13 / 0.23 (2,897) | 98.45 / 97.99 / 3.59 (326) |
| TON Dir c1 | 62.9k (92.8%) | 96.90 / 99.92 / 21.20 (13) | |
| TON Dir c2 | 8.6k (65.5%) | 99.02 / 99.80 / 5.97 (33) | |
| TON Dir c3 | 8.4k (8.5%) | 96.34 / 93.22 / 1.00 (1,099) | |

FL helps attack-poor clients (recall 45–93% alone → 98–99.8%) and the attack-heavy one (FAR 21% → 3.6%);
data-rich representative clients gain nothing in F1 (operating-point differences). Paper 5.11 / Table 9.

## QAT stability + clipped PTQ calibration (`2026-10-04_m_qat_stability`)
Unclipped conversions reproduce job l exactly (same seeds).

A — deployed recipe (prune 50 → client-0 FT → QAT) over 5 FT draws, F1:
TON QAT 98.74–99.38 (PTQ of same pruned models 97.44–99.21); CIC QAT 86.77–89.28 (PTQ 79.37–89.05).
QAT is more stable than PTQ but has ~2.5 F1 draw spread on CIC; Table 3's 85.59 is below all 5 draws
(conservative, not cherry-picked).

B — 15 calibration sets × 2,000 samples, calibration inputs clipped to ±c:
| clip | TON F1 range | CIC F1 range |
|---|---|---|
| none | 12.28–99.35 | 29.16–84.99 |
| ±3 | 96.71–96.85 | 83.04–84.73 |
| ±5 | 99.32–99.35 | 83.02–84.85 |
| ±10 | 99.34–99.37 | 83.06–84.95 |
Mechanism: worst conversions have the widest float activation maxima (TON F1 12–22 ↔ ~500; CIC 29.16 ↔
9,609; most others 52–230), not strictly monotonic. Clipped calibration = one-line fix.
Paper 5.9 updated (Table 7 clipped row, mechanism, fix, QAT draw spread); abstract/contributions/
6.3/conclusion mention the fix. Remaining PENDING: ESP32 only.

## Compression pipeline combinations (`2026-10-05_n_compression_combos`, 3 FT draws, client 0 only)
All 49 runs OK. Mean F1 (range), flash / gzip KB, CIC mean missed attacks:

| Pipeline | Flash TON/CIC | gzip | TON F1 | CIC F1 | CIC FN |
|---|---|---|---|---|---|
| FP32 | 720/803 | 672/755 | 99.40 | 85.70 | 52 |
| PTQ | 206/227 | 156/150 | 98.87 (98.37–99.26) | 82.36 (81.04–83.21) | 163 |
| PTQ clipped | 206/227 | 156/150 | 99.34 | 83.97 (83.27–84.35) | 57 |
| QAT FT | 186/207 | 125/99 | 99.10 | 90.51 (89.27–91.99) | 580 |
| S50 → FT → PTQ | 65/75 | 47/47 | 98.50 (97.44–99.21) | 84.39 (79.37–87.27) | 500 |
| S50 → FT → PTQ clipped | 65/75 | 47/47 | 99.37 | 87.67 (86.86–88.09) | 830 |
| S50 → FT → QAT (deployed) | 55/65 | 39/34 | 99.08 (98.88–99.38) | 88.37 (87.64–89.28) | 498 |
| S50 → FT → QAT clipped inputs | 55/65 | 39/35 | 99.40 (99.34–99.44) | 90.36 (88.54–92.44) | 1,163 |
| S50 → KD FT → QAT | 55/65 | 39/34 | 99.13 | 86.04 | 274 |
| S50 → KD FT → PTQ clipped | 65/75 | 47/47 | 99.37 | 84.85 | 237 |
| QAT → S50 → FT → QAT | 55/65 | 39/34 | 99.12 | 88.64 (86.65–90.58) | 687 |
| M50 → PTQ clipped | 206/227 | 124/128 | 99.40 | 89.72 | 491 |
| M50 → PQAT | 186/207 | 100/90 | 99.12 | 90.70 | 487 |
| M80 → PTQ clipped | 206/227 | 71/76 | 99.34 | 87.98 | 923 |
| M80 → PQAT | 186/207 | 56/57 | 98.85 | 89.77 | 609 |
| M80 → standard QAT | 186/207 | 76/62 | 99.07 | 89.77 | 571 |
| S50 → FT → M50 → PQAT | 55/65 | 31/31 | 98.95 | 88.66 | 699 |

Readings: unstructured pruning never shrinks TFLM flash (gzip only); PQAT keeps sparsity, standard QAT
regrows (80→62% TON, 80→75% CIC). Clipping helps everything on TON (S50→QAT-clip = FP32 at 55 KB);
on CIC it fixes PTQ collapses but after pruning shifts toward precision (more missed attacks). KD FT =
fewest missed attacks at 65 KB. Order (quantize-first) no effect. Paper 5.12 / Table 10.

## Recall-priority CIC deployment (`2026-10-05_o_recall_priority`, 65 KB, 3 FT draws, client-0 val threshold)
| Model | Selection | t | missed mean (max) | recall | FAR |
|---|---|---|---|---|---|
| FP32 | fixed 0.3 | 0.30 | 52 | 99.94 | 6.84 |
| FP32 | val ≥99.99% | 0.27 | 37 | 99.96 | 7.56 |
| deployed (hard) | fixed 0.3 | 0.30 | 463 (483) | 99.46 | 5.97 |
| deployed (hard) | val ≥99.95% | 0.005 | 44 (58) | 99.95 | 23.39 |
| attack weight 4× | val ≥99.95% | 0.027 | 63 (81) | 99.93 | 20.31 |
| KD α=0.5 | val ≥99.95% | 0.164 | 37 (52) | 99.96 | 19.92 |
| KD α=0 | val ≥99.95% | 0.302 | 66 (80) | 99.92 | 18.53 |
| KD + 50k FT | val ≥99.95% | 0.159 | 40 (56) | 99.95 | 17.94 |
| **KD + clipped inputs** | val ≥99.9% | 0.214 | 77 (133) | 99.91 | 9.54 |
| **KD + clipped inputs** | val ≥99.95% | 0.198 | **44 (55)** | 99.95 | **10.43** |
| **KD + clipped inputs** | val ≥99.99% | 0.171 | 24 (25) | 99.97 | 14.33 |

Hard-label QAT puts scores in the lowest INT8 output steps (t = 1–3 × 1/256) → coarse threshold control,
high FAR. KD keeps scores mid-range; with clipped inputs best trade-off. Client-val FAR predicts test FAR
within 0.35 pt; 99.99% target not met (≈10k val attacks, SMOTE). Equal-recall cost of compression ≈ 3 FAR
pts (FP32 22 missed @ 11.0% vs 65 KB 24 @ 14.3%). Paper 5.13 / Table 11; abstract, 5.8, contributions,
conclusion, limitations updated.

## Recall-priority follow-up (`2026-10-05_p_recall_priority2`, KD α=0.5 + clipped inputs, 3 draws)
kdclip reproduces job o exactly. Missed (mean, max) / FAR at client-val targets:

| Variant | size | ≥99.95% | ≥99.99% |
|---|---|---|---|
| kdclip (ref) | 65 KB | 44 (55) / 10.43 | 24 (25) / 14.33 |
| + longer FT 6+4 | 65 KB | 54 (74) / 8.66 | 31 (45) / 10.94 |
| + T=4 | 65 KB | 39 (42) / 13.03 | 22 (28) / 16.86 |
| + federated FT (4 clients) | 65 KB | 51 (79) / 9.31 | 27 (33) / 11.28 |
| prune 30% | 112 KB | 54 (73) / 7.63 | 23 (27) / 9.65 |
| no pruning | 207 KB | 51 (52) / 6.71 | 23 (28) / 10.39 |
| FP32 (val) | 803 KB | 67 / 6.58 | 37 / 7.56 |

Residual misses at 99.99%: Bot ~6, Infiltration 4–8, PortScan 2–4, Heartbleed ≤3; FP32 misses 7
Infiltration at every recorded threshold → ceiling is the FL model / data (rare classes, label issues
[24]), not compression. Paper 5.13 extended; abstract/5.8/conclusion now: 65 KB misses 27 @ 11.3% FAR.

## CIC attack-type test counts (`2026-10-05_q_attack_counts`) and per-type misses (job p)
Test flows: DoS Hulk 34,487 · DDoS 25,569 · PortScan 18,282 · GoldenEye 2,006 · FTP-Patator 1,236 ·
Slowhttptest 1,071 · slowloris 1,063 · SSH-Patator 650 · Bot 368 · Web brute force 301 · XSS 120 ·
Infiltration 11 · Heartbleed 5 · SQL injection 4 (train before SMOTE: Infiltration 25, Heartbleed 6).
Missed (FP32 @0.3 / FP32 val99.99 / 65 KB fed val99.99 / 112 KB val99.99): Infiltration 7/7/8.3/6.3 of 11;
Heartbleed 0/0/3.3/0 of 5 (compression-induced loss at 65 KB!); SQLi 1/1/0/0.3 of 4; Bot 20/11/6.0/6.7 of 368.
Paper Table 12.

## Recall-priority recipe on TON_IoT (`2026-10-05_r_recall_priority_ton`, val 8,000 samples)
| Model | Selection | missed / 16,215 (max) | FAR |
|---|---|---|---|
| FP32 | fixed 0.3 | 61 | 2.76 |
| FP32 | val ≥99.9% | 2 (t=0.021) | 96.76 |
| 55 KB hard | fixed 0.3 | 182 (233) | 3.01 |
| 55 KB hard | val ≥99.9% | 16 (29) | 36.44 |
| 55 KB clip QAT | val ≥99.9% | 21 (25) | 7.02 (99.99% → t=0, FAR 100) |
| 55 KB KD+clip | val ≥99.9% | 19 (20) | 5.94 |
| 55 KB KD+clip | val ≥99.99% | 10 (10) | 11.17 |
| 55 KB KD+clip+fed | val ≥99.99% | 7 (10) | 12.60 |
| 98 KB p30 | val ≥99.9% | 21 (25) | 5.79 |
FP32 test sweep t=0.2: 21 missed @ 5.85. → Same recipe works on both datasets; KD is what makes client-val
thresholds usable (scores mid-range). Fed FT/p30 neutral on TON. Paper Table 13 + unified recommendation.

## Final deployment models (user decision 2026-10-05: CIC 112 KB, TON new recipe)
- CIC-IDS2017: `2026-10-05_p_recall_priority2/.../kdclip_p30_d0.tflite` (114,760 B), threshold 0.1914 (client-val
  ≥99.99%): 27 / 85,173 missed (0.032%), FAR 10.02%.
- TON_IoT: `2026-10-05_r_recall_priority_ton/.../kdclip_d0.tflite` (56,512 B), threshold 0.2148 (client-val ≥99.9%):
  19 / 16,215 missed (0.12%), FAR 6.06%.
ESP32 firmware now embeds cic_deploy / cic_fp32 / ton_deploy / ton_fp32 (1.7 MB); host build against TFLM
(TensorFlowLite_ESP32 1.0.0): FP32 exact, INT8 ≤0.012 (3/256), 8/8 decisions at deployment thresholds,
arena 2.2–4.6 KB. Paper: "deployed recipe" renamed "base recipe"; 5.8, 6.1, Appendix B, abstract updated.

## TON_IoT federated-training sweep (`2026-10-05_s_ton_sweep`, selection on validation only)
Validation (8,414 rows, 6,485 attacks): FAR at 99.9% val recall / F1@0.3:
base 6.84 / 99.36 · alpha 0.5 3.78 / 99.35 · alpha 0.65 7.72 / 99.06 · no SMOTE 4.30 / 99.28 ·
text features 3.21 / 99.45 · 100 rounds 5.18 / 99.36 · **text + alpha 0.5 3.01 / 99.39** · text + no SMOTE 3.06 / 99.41.
Selected text + alpha 0.5 (6 row-local features from dns_query / http_uri; 43 inputs), retrained, test:
| | F1@0.3 | missed@0.3 / 16,215 | FAR@0.3 | FAR at 99.9% test recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| reference (current) | 99.40 | 61 | 2.76 | 7.16 | 0.99860 | 0.99937 |
| new (text + α 0.5) | 99.41 | **19** | 3.61 | **4.52** | 0.99895 | 0.99959 |
Missed by type @0.3 — reference: dos 16, mitm 14, ddos 10, xss 7, ransomware 4, backdoor 4, injection 4,
password 2; new: mitm 5, scanning 4, ransomware 3, dos 3, xss 2, injection 1, ddos 1.
Note: TON test split has 16,215 attacks (tp+fn), not 16,215 as written earlier — corrected in the paper.
Job t queued: recall-priority compression on the new model.

## Recall-priority compression on the improved TON model (`2026-10-05_t_recall_priority_ton2`, 43 inputs)
Missed / 16,215 (max over 3 draws) / FAR: FP32 improved @0.3 19 / 3.61; FP32 val≥99.95% 9 / 4.71;
55 KB base @0.3 168 (274) / 6.45; KD+clip val≥99.99% 3.7 (8) / 12.30 (FAR 7.2–17.3 across draws);
KD+clip+fed val≥99.9% 30 (32) / 3.68; ≥99.95% 17.7 (20) / 5.61; **≥99.99% 12.0 (14) / 6.61**;
100 KB p30 val≥99.99% 5.0 (8) / 9.07. Old-model 55 KB deployment: 19 / 5.94.
New TON deployment: `ton_near_iid_v2/kdclip_fed_d0.tflite` (58,048 B), threshold 0.2070:
12 / 16,215 missed (0.074%), FAR 6.49%. ESP32 firmware restaged; host build: INT8 parity ≤ 1/256,
8/8 decisions, arena 2.2–4.6 KB. Paper: 3.1, new 5.14 (Tables 14–15), 5.8, abstract, 6.1, App. B,
limitations updated.

## TON_IoT v2 everywhere (jobs `2026-10-05_u_toniot_v2`, `_v_toniot_v2_downstream`, `_w_toniot_v2_sweep_robustness`)
All TON numbers in the paper now use the validation-selected recipe (43 features, focal α 0.5); the
near-IID FL model is job s's `best_text_alpha05_test.h5` (source of the deployment model).
- Table 1: centralized 99.51/99.43/99.94/99.68/1.93 (10 missed); FL 99.08/98.94/99.88/99.41/3.61 (19);
  FL second run 99.04/98.91/99.86/99.38/3.71 (23); fixed LR 98.97/98.82/99.86/99.34/4.00 (23).
- Table 2: Dirichlet 97.81/98.60/99.99/9.52 (1 missed). 5.6: training-time QAT FL 98.14 acc / 98.79 F1 (166).
- Table 3: FP32 732.4 KB 99.41/3.61 (19); PTQ 209.4 98.96/3.86 (153); prune no-FT 66.3 95.30/30.20 (129);
  client FT PTQ 90.20/13.23 (2,369); base recipe 56.7 KB 98.67/2.26 (319); pooled 99.16/3.48 (105). 12.92×.
- Table 4: TON near-IID c0 98.67/2.26 (319); Dirichlet c2 98.80/2.88 (248).
- Fig. 1: client FT+QAT 98.65–98.74 from 30% to 90% (8.2 KB); PTQ 87.7–91.5; no FT collapses from 85%.
- Table 5: FL 99.1/23.0/22.9; central 99.5/23.0/23.0; PTQ 98.4; client PTQ 85.7; client QAT 98.0; pooled QAT 98.7/91.1/91.1.
- Table 6: int8 PTQ federated 97.69 (FAR 3.8), pruned 90.20 (FAR 13.2); other methods ≤0.01 loss.
- Table 7: n=100/500/2000 median 99.36/99.37/97.68, min 96.11/54.01/32.72, within-1 14/13/5 of 15; clip ±5 99.19 (15/15), ±10 99.37–99.39, ±3 98.30–98.79.
- QAT FT draws: 98.79–99.51 (PTQ 96.89–99.19).
- Table 8: KD 1/8 student 99.21 at 11.6 KB PTQ; KD PTQ loss ≤0.29 vs scratch 0.55–0.70.
- Table 9: near-IID clients 99.08–99.35 (31–51 missed) vs FL 19; Dirichlet c0 85.53% recall (2,346 missed), c3 93.39% (1,072), FL 99.99% (1).
- Table 10: TON clipped-input QAT 99.49 (99.47–99.51), 84 missed vs base 174.
- Table 13 = former Table 15 (+ base/clip-only rows); 5.14 now explains how the TON recipe was chosen.
Limitation row "Two TON_IoT federated models" removed; seed-variance note added.

## Literature quantization methods A–D (jobs `2026-10-05_x1`–`x4`, `scripts/quant_lit_methods.py`, docs/QUANT_LITERATURE.md)
Models: CIC near-IID federated (job b) and TON v2 (job s). Missed = mean over runs / all test attacks.
`fq_` = fake-quant model with ranges set directly, per-tensor weights (identical to per-tensor TFLite PTQ);
`tflite_` = TFLite PTQ, per-channel weights.

**A. PTQ range estimators, unpruned model, 15 calibration sets (F1 mean [min–max], FAR, missed)**
| estimator | CIC | TON |
|---|---|---|
| tflite max (per-channel) | 77.67 [29.16–84.99], 15.51, 349 | 77.91 [32.72–99.09], 2.99, 4,779 |
| **tflite ±5 SD clip (ours)** | 84.38 [83.02–84.85], 7.58, 68 | 99.19 [99.19–99.19], 4.94, 27 |
| tflite max per-tensor | 76.66 [29.20–83.69], 16.16, 336 | 81.66 [30.25–99.33], 4.44, 3,913 |
| fq max | 76.64 [29.20–83.69], 16.14, 397 | 81.00 [36.06–99.31], 4.48, 4,156 |
| **fq percentile 99.9** | 83.16 [82.11–84.90], 8.30, 58 | **99.39 [99.38–99.40], 3.55, 26** |
| fq percentile 99.99 | 79.27 [31.51–84.12], 13.34, 543 | 99.31 [98.81–99.40], 3.62, 50 |
| fq MSE (ACIQ objective) | 77.28 [29.28–84.14], 15.50, 630 | 88.68 [34.62–99.39], 4.72, 2,395 |
| fq KL (TensorRT entropy) | 80.93 [44.16–84.13], 10.68, 565 | 98.97 [96.69–99.39], 4.20, 131 |
| fq ±5 SD clip | 82.93 [82.09–83.54], 8.37, 239 | 99.19 [99.18–99.19], 4.94, 27 |
Only aggressive clipping (±5 SD or 99.9th percentile) removes the collapses; MSE, KL and 99.99th percentile
still collapse on some draws. With the same per-tensor weights, percentile 99.9 ≥ ±5 SD on both datasets.
Per-channel weights matter little for max calibration but cut CIC misses with ±5 clipping (239 → 68).

**B. QAT start/ranges, base recipe (prune 50% → FT → QAT), 5 FT draws (F1 mean [min–max], FAR, missed)**
| | CIC | TON |
|---|---|---|
| PTQ tflite max | 85.81 [79.37–89.05], 6.69, 562 | 98.62 [96.89–99.19], 1.90, 351 |
| PTQ fq ±5 SD | 87.93 [87.60–88.52], 5.24, 1,320 | 99.41 [99.39–99.44], 1.91, 99 |
| QAT tfmot default | 88.02 [86.77–89.28], 5.44, 513 | 99.00 [98.79–99.51], 2.49, 204 |
| QAT init from calibration (EMA continues), 5 estimators | 87.79–88.18, 5.34–5.56, 447–664 | 98.99–99.03, 2.47–2.53, 194–208 |
| QAT fixed at max | **89.49** [87.41–91.43], 4.62, 757 | 99.46 [99.40–99.53], 2.10, 74 |
| QAT fixed at percentile 99.99 | 88.58 [87.42–89.95], 5.18, **392** | **99.47 [99.44–99.51], 2.14, 70** |
| QAT fixed at MSE | 88.62 [87.25–90.08], 5.03, 839 | 99.47 [99.43–99.56], 1.97, 76 |
| QAT fixed at ±5 SD | 88.10 [87.43–89.01], 5.37, 598 | 99.34 [99.16–99.43], 2.79, 79 |
Mechanism (draw-0 ranges): tfmot default QAT keeps hidden ranges near the ±6 start ([-5.21, ~10]) and its input
quantizer tracks the full FT-input min/max (CIC max 61.2, TON 84.6). "init" changes the hidden ranges
(e.g. TON d1 [0, 23]) but keeps that input range and gives the same results as default; "fixed" also keeps
the calibrated input range (CIC 12.1, TON 26.7) and is the only variant that helps (TON misses 204 → 70,
FAR 2.49 → 2.14). The input range, not the hidden ranges, is the lever — consistent with A and with job n
(QAT on clipped inputs best for TON; the deployed recall recipe already clips FT inputs).

**C. Data-free (server-side) calibration, unpruned model (F1, FAR, missed)**
| | CIC | TON |
|---|---|---|
| real client data, tflite max (3 draws) | 77.57 [69.11–83.46], 12.10, 291 | 76.93 [32.72–99.09], 3.06, 4,435 |
| real client data, tflite ±5 SD | 84.19, 7.69, 65 | 99.19, 4.94, 27 |
| synthetic N(0,1), tflite (3 draws) | 82.22 [80.58–83.22], 8.87, 54 | 99.16 [99.08–99.22], 5.06, 29 |
| BN statistics, k = 4 / 6 | 83.03, 8.37, 72 / 82.51, 8.69, 66 | 99.17, 5.06, 26 / 99.26, 4.44, 27 |
| **CLE + BN statistics k = 6** | **84.64, 7.44, 55** | 99.25, 4.48, 30 |
| real ±5 SD, per-tensor (fq) without / with CLE | 82.86, 8.42, 224 / 84.46, 7.54, 58 | 99.19, 4.93, 28 / 99.18, 4.93, 29 |
BN statistics k = 3 is too tight (CIC 532, TON 282 missed). CLE float output change ≤ 1e-5.
Server-side calibration from synthetic inputs or the federated BN statistics matches clipped real-data
calibration; no client data leaves the clients. CLE mainly fixes per-tensor weight quantization on CIC.

**D. Learned (PACT-style) clipping of hidden ReLU outputs, 5 FT draws**
CIC: default 88.02 / 5.44 / 513; learned from max 88.21 / 5.29 / 718; from pct99.99 87.67 / 5.58 / 652;
from ±5 SD 88.73 / 5.03 / 676. TON: all 98.99–99.00, FAR 2.46–2.59, 199–208 missed (default 204).
No gain: the hidden activation ranges are not the bottleneck (the input range is, see B).

## Deployed recall-priority recipes + fixed QAT ranges / CLE (jobs `2026-10-05_x5_recall_fixed_cic`, `_x6_recall_fixed_ton`)
Same seeds and client validation split as jobs p / t; reference rows reproduce them exactly.
Missed (mean, max of 3 draws) / FAR mean at val recall ≥ 99.99%:
| variant | CIC 112 KB (kdclip_p30) | TON 57 KB (kdclip_fed) |
|---|---|---|
| deployed recipe | 23.3 (27) / 9.65% | 12.0 (14) / 6.61% |
| + ranges fixed at pct 99.99 | 28.7 (33) / 8.63% | 8.3 (10) / 7.12% |
| + ranges fixed at max | 26.3 (34) / 9.33% | 11.0 (16) / 6.60% |
| + CLE | 29.3 (43) / 35.27% (one draw 84%) | 12.7 (17) / 7.42% |
| + fixed pct 99.99 + CLE | 36.3 (59) / 9.97% | 14.3 (17) / 7.30% |
At matched FAR the curves coincide (TON fixed 0.2: 13.3 vs 14.3 missed at 6.3%; val ≥99.9%: 30 vs 30 at 3.6–3.7%),
so fixed ranges shift the operating point, not the trade-off. No deployment change: the recipe already
fine-tunes on clipped inputs, which bounds the input range (the lever found in job x2). Paper: LaTeX §4.8
paragraph + provenance X5/X6; docx paragraph after Table 13.
