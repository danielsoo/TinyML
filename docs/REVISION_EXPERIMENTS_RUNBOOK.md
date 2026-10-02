# Revision experiments runbook (LCTES #81 follow-ups)

Companion to `docs/LCTES_REVISION_STATUS.md` §3. Everything below uses the configs in
`config/paper_v12/`, which reproduce the **paper's** recipe (Sec. 3.2 of the revised draft):
60 rounds × 3 local epochs, lr 1e-3 cosine → 1e-4, focal α = 0.35, balance_ratio 4.0,
FedAvgM (server momentum 0.5, server lr 0.1), full CIC-IDS2017 (`max_samples: null`).
Source of truth for that recipe: `v12/2026-02-06_23-25-39/federated_local_sky.yaml`.

> ⚠️ Do **not** use `config/ablation/*.yaml` or `config/federated.yaml` for paper numbers:
> they use α = 0.7, 80 rounds × 2 epochs, lr 5e-4, server lr 1.0 and a 2M-sample cap, so their
> results are not comparable to the headline run in the paper.

| File | Paper item | Differs from `fl_baseline.yaml` in |
|---|---|---|
| `fl_baseline.yaml` | headline FL run (near-IID, label_balanced) | — |
| `fl_dirichlet.yaml` | §3c non-IID | `partition_strategy: dirichlet` (α_dir = 0.3) |
| `centralized.yaml` | §3b / Table 2 row (a) | trained by `train_centralized.py` (no FL, no QAT) |
| `failed_config.yaml` | Table 2 row (b) fixed-LR | `lr_decay_type: none` |

## 0. Setup (GPU machine: Vast.ai / PSU server / local)

```bash
make setup && source .venv/bin/activate
# Put the 8 CIC-IDS2017 *.pcap_ISCX.csv files in data/raw/CIC-IDS2017/
# (or edit data.path in the config). The loader globs "*.pcap_ISCX.csv".
```

Smoke test first (5 rounds × 1 epoch, few minutes):

```bash
python scripts/run_non_iid_ablation.py --base-config config/paper_v12/fl_baseline.yaml \
  --strategies label_balanced,dirichlet --client-counts 4 --quick
```

## 1. §3b Centralized baseline (Table 2 row a)

```bash
python scripts/train_centralized.py --config config/paper_v12/centralized.yaml \
  --save-model models/paper_v12/centralized.h5
```

- Trains 60 × 3 = **180 epochs** by default and applies the **same per-round cosine LR**
  formula the FL client uses (each block of 3 epochs = one "round").
  (Before this fix the script ignored `lr_decay_type` and capped epochs at 100.)
- Rough cost: ~19k samples/s on a 4-core CPU ⇒ hours on CPU; prefer GPU.
- Evaluate it (+ the FL rows) in one table:

```bash
python scripts/run_baseline_ablation.py --config-dir config/paper_v12 --with-failed
# --skip-train re-uses models under the output dir; see --help
```

Output: `data/processed/ablation/<timestamp>/baseline_ablation.{md,csv,json}`.

## 2. §3c Non-IID Dirichlet(0.3)

```bash
python scripts/run_non_iid_ablation.py --base-config config/paper_v12/fl_baseline.yaml \
  --strategies label_balanced,dirichlet --client-counts 4
```

Reports accuracy / precision / Attack Recall / F1 / false-alarm rate per strategy plus each
client's attack:normal mix. For "rounds-to-convergence", read the per-round Flower metrics in
the training log (`[ROUND n]` lines) of each run.

## 3. §3f Client-count scaling

Same script, more clients:

```bash
python scripts/run_non_iid_ablation.py --base-config config/paper_v12/fl_baseline.yaml \
  --strategies label_balanced,dirichlet --client-counts 4,20,50
```

## 4. §3a ESP32 (hardware)

Unchanged — see Appendix B of the paper / `docs/LCTES_REVISION_STATUS.md` §3a.

## 5. Rebuilding the paper

```bash
cd paper && npm install && npm run build   # -> paper/TinyML_Federated_IDS_Revised.docx
```

Edit text/tables in `paper/main.js`; figures live in `paper/figures/`.
