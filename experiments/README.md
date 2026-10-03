# Experiment queue (PC worker)

`scripts/pc_worker.sh` runs on the lab/home PC (WSL2). Every few minutes it pulls this
branch, runs any job in `experiments/queue/*.yaml` that has no results yet, and pushes
the results back. Claude adds jobs here after reading previous results.

Job file (`experiments/queue/<id>.yaml`):

```yaml
id: 2026-10-04_noniid_tune1        # results go to data/processed/revision/<id>/
config_dir: config/tuning/noniid_tune1
steps: non_iid                     # baseline_ablation,non_iid,client_scaling
flags: []                          # allowed: --quick, --with-failed, --with-scaling
why: "one line: what this job tests and why"
```

The worker only ever runs `scripts/run_revision_experiments.sh` with these whitelisted
options; job files are data, not shell. Tuning jobs must use `eval_split: val` in their
configs so model selection never looks at the test set (see docs/LOCAL_PC_EXPERIMENTS_GUIDE.md).
