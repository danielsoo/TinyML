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
