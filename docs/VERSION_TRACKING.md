# Version & Experiment Tracking

For paper/report use, each run records configuration and results.

## Run-Level Comparison (per-run by timestamp)

When multiple runs (timestamps) exist for the same version:

```bash
python scripts/generate_run_level_changelog.py --analysis-dir TinyML-results/processed/analysis
```

→ Generates `RUN_LEVEL_DIFFERENCES.md` (metrics + estimated changes).

## Per-Run Output (analysis/version/run_id/)

- **compression_analysis.md** – Full report with **Configuration** section (data, model, FL params)
- **compression_analysis.json** – JSON with `config` + `results`
- **config_snapshot.yaml** – Full config used for that run

## Cross-Version Summary

Generate `VERSIONS.md` to compare all runs:

```bash
python scripts/generate_version_summary.py --analysis-dir data/processed/analysis
# or for local results:
python scripts/generate_version_summary.py --analysis-dir TinyML-results/processed/analysis --output TinyML-results/processed/analysis/VERSIONS.md
```

Output: table with Version | Run | Config | Best Acc | Best F1 | P/R (Orig | Comp) | Size | Ratio.

## Config Summary Abbreviations

| Abbrev | Meaning |
|--------|---------|
| maxNk | max_samples (e.g. max1500k = 1.5M) |
| balR | balance_ratio (e.g. bal1.0 = 50:50) |
| smote | use_smote=true |
| rN | num_rounds |
| epN | local_epochs |
| flα | focal_loss + alpha |
| FedAvgM | server momentum |
