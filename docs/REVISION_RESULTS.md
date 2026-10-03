# Revision experiment results (running log)

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
