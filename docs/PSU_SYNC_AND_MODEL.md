# PSU Server Sync & Finding Models on Server

## Server → Local (pull results only)

**Use TinyML-main only.** Results pulled from the server (analysis, models, eval, outputs) go **entirely into TinyML-main**.

```bash
cd /path/to/TinyML-main
bash scripts/pull_results_to_tinyml_results.sh
```

Or from the Research folder:

```bash
cd /path/to/Research
bash TinyML-main/scripts/pull_results_to_tinyml_results.sh
```

- **Destination**: **TinyML-main** (`data/processed/runs/`, `models/`, `outputs/`)
- **Folder layout**: one run = one folder  
  `TinyML-main/data/processed/runs/<version>/<datetime>/`  
  - `run_config.yaml` — full config used for that run (rounds, epochs, balance_ratio, etc.)  
  - `analysis/` — compression_analysis.md (includes **Run / Training Configuration**), visualizations  
  - `models/` — tflite, h5, etc.  
  - `outputs/`  
  - `eval/` — ratio_sweep_report.md (training config summary)  
- **Example**: `TinyML-main/data/processed/runs/v11/2026-02-02_23-28-45/models/tflite/saved_model_original.tflite`

**Backfill training config into existing run reports (run once):**  
To add "Run / Training Configuration" to older runs that lack it in `compression_analysis.md` / `ratio_sweep_report.md`:

```bash
cd /path/to/TinyML-main
# after activating conda/venv (pyyaml required)
python scripts/backfill_run_config_to_reports.py
```

- `--dry-run`: show what would change without modifying files  
- If `run_config.yaml` is missing in a run folder, infer from version and save  
- Insert section into `analysis/compression_analysis.md`, `eval/ratio_sweep_report.md` if missing

To use rsync directly:

```bash
cd /path/to/TinyML-main
rsync -avz yqp5187@e5-cse-135-01.cse.psu.edu:/scratch/yqp5187/TinyML-main/data/processed/runs/ data/processed/runs/
rsync -avz yqp5187@e5-cse-135-01.cse.psu.edu:/scratch/yqp5187/TinyML-main/models/ models/
rsync -avz yqp5187@e5-cse-135-01.cse.psu.edu:/scratch/yqp5187/TinyML-main/outputs/ outputs/
```

**Option:** `sync_results_from_psu.sh` syncs only `data/processed/` (analysis, runs). For full models and eval per run, use `pull_results_to_tinyml_results.sh` above.

**When you see `command not found` / `invalid option` / `hostname contains invalid characters`:** The script may have been saved with CRLF (Windows line endings). From the TinyML-main folder run once:

```bash
sed -i '' 's/\r$//' scripts/pull_results_to_tinyml_results.sh
```

Then run `bash scripts/pull_results_to_tinyml_results.sh` again.

---

## Local → Server (push code)

To push changed local files to the server:

```bash
cd /path/to/TinyML-main
bash scripts/sync_to_psu.sh
```

- **Included**: `config/`, `scripts/`, `src/` (code), `docs/`, etc.
- **Excluded**: `data/raw`, `data/processed`, `models/`, `src/models/*.h5`, `outputs`, `TinyML-results`, `.git` (large/generated)

---

## Running on Vast.ai

On Vast instances use a **single data path** (`data/raw/CIC-IDS2017`), so one full rsync/sync brings both code and data. Do **not** use PSU's `config/federated_scratch.yaml`; use the **Vast** config.

**0. config must exist on Vast**  
When uploading the project to Vast, include the **entire `config/`** folder. Without `config/federated_vast.yaml` you get `Configuration file not found: .../config/federated_vast.yaml` / `Could not verify data path`.  
If the file is missing, copy it from local once or create it on Vast manually:

```bash
# On Vast instance (when config/ exists but federated_vast.yaml is missing)
mkdir -p /workspace/TinyML-main/config
cat > /workspace/TinyML-main/config/federated_vast.yaml << 'EOF'
# config/federated_vast.yaml - Vast: single data path (data/raw/CIC-IDS2017)
version: "v12"
data:
  name: cicids2017
  path: "data/raw/CIC-IDS2017"
  num_clients: 4
  max_samples: 1500000
  binary: true
  use_smote: true
  balance_ratio: 4.0
model:
  name: mlp
federated:
  num_rounds: 50
  fraction_fit: 1.0
  fraction_evaluate: 1.0
  local_epochs: 5
  batch_size: 128
  learning_rate: 0.0005
  lr_decay: 0.99
  use_class_weights: true
  use_focal_loss: true
  focal_loss_alpha: 0.92
  use_callbacks: false
  server_momentum: 0.9
  server_learning_rate: 1.0
  use_qat: true
  min_fit_clients: 4
  min_evaluate_clients: 4
  min_available_clients: 4
EOF
```

**1. Install dependencies (once)**  
Vast base images may not include `flwr` etc. After connecting to the instance:

```bash
cd /workspace/TinyML-main
pip install -r requirements.txt
```

**2. Run**  
Use **`config/federated_vast.yaml`** only (data path: `data/raw/CIC-IDS2017`):

```bash
cd /workspace/TinyML-main
python run.py --config config/federated_vast.yaml
```

Or:

```bash
bash scripts/run_vast.sh
```

**2b. Keep training when disconnected: use tmux**  
If your Mac sleeps or SSH drops, the training in that terminal will stop. Running inside **tmux** keeps the session on the server.

```bash
# After SSH to Vast
tmux new -s train
cd /workspace/TinyML-main
python run.py --config config/federated_vast.yaml
```

- **Detach**: `Ctrl+B` then `D` — training keeps running.
- **Reattach**: After SSH, `tmux attach -t train` to see logs again.
- If tmux is missing: `apt-get update && apt-get install -y tmux` (or skip if already in image)

**3. Summary**

| Item | PSU server | Vast.ai |
|------|------------|---------|
| config | `config/federated_scratch.yaml` | `config/federated_vast.yaml` |
| Data path | `/scratch/yqp5187/Bot-IoT` | `data/raw/CIC-IDS2017` (inside project) |
| Run | `bash scripts/run_psu_server.sh` | `bash scripts/run_vast.sh` or `python run.py --config config/federated_vast.yaml` |

Using `federated_scratch.yaml` on Vast points to `/scratch/...` which does not exist there and can fail; without `flwr` you get `ModuleNotFoundError: No module named 'flwr'`. Following the steps above fixes both.

---

## When OOM occurs / Pre-run cache and memory cleanup

OOM is a **RAM** issue; clearing disk cache (pip/conda) rarely helps. What helps is **cleaning leftover Ray/Python processes** from previous runs.

**Before running (on server):**

1. **Stop Ray** — Ray workers can keep using memory in the background even after a run dies  
   ```bash
   ray stop --force
   ```
2. **Use run_psu_server.sh** — It runs `ray stop` at start, so `bash scripts/run_psu_server.sh` gives a clean state each time.
3. **Cap data size** — If OOM persists, lower `max_samples` in `config/federated_scratch.yaml` (e.g. 1500000 or less).

**Summary:** Cleaning Ray/previous processes is effective; `run_psu_server.sh` includes that step.

---

## Freeing server disk space: what is safe to remove

When server disk (`/scratch/yqp5187/`) is low, remove **only what you don't need**. Check sizes with `du -sh <path>` first.

**Check sizes on server:**

```bash
cd /scratch/yqp5187/TinyML-main

# Entire runs (can be several GB if many old runs)
du -sh data/processed/runs/

# Per run (to decide what to delete)
du -sh data/processed/runs/*/*

# Other
du -sh outputs/ models/ 2>/dev/null
du -sh /scratch/yqp5187/conda_envs/research   # conda env
```

**Safe to remove (already pulled locally or no longer needed):**

| Target | Description | Example command |
|--------|-------------|-----------------|
| **Old runs** | v11/v12 etc. Run folders; pull needed ones locally first | `rm -rf data/processed/runs/v11/2026-02-02_*` |
| **outputs/** | Previous pipeline output (duplicate if in runs) | `rm -rf outputs/*` |
| **models/** root | `models/*.h5`, `models/tflite/*` — duplicate if in runs | `rm -f models/*.h5 models/tflite/*.tflite` |
| **pip cache** | Re-downloaded on reinstall | `pip cache purge` |
| **conda cache** | Re-downloaded on reinstall | `conda clean -a` |
| **Ray /tmp** | Ray temp files | `rm -rf /scratch/yqp5187/tmp/ray/*` (after stopping Ray) |

**Do not remove:**

- `data/raw/` (original CSV) — needed for training
- `src/`, `config/`, `scripts/` — code
- **Current run** — back up with `pull_results` etc. before deleting

**One-liner:** Removing old `data/processed/runs/<version>/<date>` and `outputs/`, `models/` frees a lot of space.

---

## Finding models on server

After training with `train.py`, models are saved at:

| File | Description |
|------|-------------|
| `src/models/global_model.h5` | Latest copy (overwritten each run) |
| `src/models/global_model_YYYYMMDD_HHMMSS.h5` | Timestamped file (e.g. from v11 run) |

**After SSH to server:**

```bash
ssh yqp5187@e5-cse-135-01.cse.psu.edu
cd /scratch/yqp5187/TinyML-main
ls -la src/models/
```

**9:1 evaluation only (skip training):**

```bash
cd /scratch/yqp5187/TinyML-main
conda activate /scratch/yqp5187/conda_envs/research   # or your env
python scripts/evaluate_9to1.py --config config/federated_local.yaml --model src/models/global_model.h5 --ratio 9
```

To test a specific timestamped model:

```bash
python scripts/evaluate_9to1.py --config config/federated_local.yaml --model src/models/global_model_20260202_232845.h5 --ratio 9
```

(Check actual filenames with `ls src/models/`)

---

## Threshold recommendation for a given ratio (tune_threshold.py)

To get a **recommended threshold** for a given ratio (e.g. 9:1):

```bash
cd /path/to/TinyML-main
python scripts/tune_threshold.py --config config/federated_scratch.yaml --model models/tflite/saved_model_original.tflite --ratio 90
```

- `--ratio 90`: 90% normal : 10% attack test set for threshold search
- `--metric f1`: Recommend by F1 (default)
- `--metric attack_recall`: Maximize attack recall (catch more attacks)
- `--metric balanced`: 0.5*F1 + 0.5*attack_recall
- `--out threshold_90.csv`: Save per-interval metrics CSV

Output: shows Accuracy, F1, Normal Recall, Normal Precision, Attack Recall for several thresholds and prints the **recommended threshold** for the chosen metric.

---

## Evaluation-only with a specific model (folder layout)

To run only ratio sweep + threshold tuning with a **given model** (no train/compression) and save the report, use `run.py --model <path>`. Report location:

| Case | Report location |
|------|------------------|
| `--model` is under **runs**<br/>e.g. `data/processed/runs/v11/2026-02-02_23-28-45/models/tflite/saved_model_original.tflite` | **That run's `eval/`**<br/>→ `data/processed/runs/v11/2026-02-02_23-28-45/eval/ratio_sweep_report.md` |
| `--model` set but path outside runs<br/>+ `.last_run_id` exists (analysis was run before) | **Current analysis folder**<br/>→ `data/processed/analysis/<version>/<datetime>/ratio_sweep_report.md` |
| Only `--model` set, no analysis/run folder | **Eval-only folder**<br/>→ `data/processed/eval/<YYYY-MM-DD_HH-MM-SS>/ratio_sweep_report.md` |
| Pipeline run without `--model` (default) | **Current analysis folder**<br/>→ `data/processed/analysis/<version>/<datetime>/ratio_sweep_report.md` |

**Example (skip train/compression/analysis, evaluate one run's model):**

```bash
python run.py --skip-train --skip-compression --skip-analysis \
  --model data/processed/runs/v11/2026-02-02_23-28-45/models/tflite/saved_model_original.tflite
```

→ Ratio sweep + threshold tuning go to `data/processed/runs/v11/2026-02-02_23-28-45/eval/ratio_sweep_report.md`. Re-running evaluation for the same run overwrites that `eval/` folder.

**Folder layout summary**

- **One training run** → `data/processed/runs/<version>/<datetime>/`  
  - `models/`, `outputs/`, `analysis/` (snapshot)  
  - **Evaluation-only with a given model** → same run's `eval/ratio_sweep_report.md`
- **Evaluation-only with arbitrary model** → `data/processed/eval/<timestamp>/ratio_sweep_report.md`

---

## Checking if compression_analysis and ratio_sweep use the same data

To verify both reports used **the same test data**:

1. **From run logs**
   - When running `run.py` or `analyze_compression.py`: `[load_cicids2017] data_path=...` and `Test set loaded: N samples`.
   - When running `evaluate_ratio_sweep.py`: same `[load_cicids2017] data_path=...` and `Test: N (Normal=..., Attack=...)`.
   - Same **data_path** and same **test set size (N)** means same data.

2. **When running everything on server**
   - A single `run.py` run executes compression analysis and ratio sweep in the **same environment** in order; without `CICIDS2017_DATA_PATH` both use config's `data.path` (e.g. `data/raw/Bot-IoT`).
   - On server, `export CICIDS2017_DATA_PATH=/scratch/yqp5187/Bot-IoT` then `run.py` — both use that path.
   - So if you ran `run.py` **once in the same shell**, both steps used the same data.

3. **If results differ**
   - High compression metrics but ratio_sweep Normal Recall 0: they may have been run in **different environments** (e.g. local vs server, or before/after env change).
   - **Different models** give different results: compression_analysis uses **TFLite** (Original/PTQ), ratio_sweep can use **Keras .h5**; `run.py` uses the **same Original TFLite** (`models/tflite/saved_model_original.tflite`) for ratio_sweep so same run ⇒ same model for both reports.
   - From then on, compare `data_path` and test sample count in logs as in step 1.

---

## Finding data on the server

After SSH to the server, to find where CIC-IDS2017 CSV files are:

```bash
# 1) Search under scratch for *.pcap_ISCX.csv (may take a while)
find /scratch/yqp5187 -name "*.pcap_ISCX.csv" -type f 2>/dev/null

# 2) CSV with CIC or ISCX in name
find /scratch/yqp5187 -name "*CIC*" -o -name "*ISCX*" 2>/dev/null | head -50

# 3) Check top-level dirs first for candidate data folders
ls -la /scratch/yqp5187/
ls -la /scratch/yqp5187/TinyML-main/data/raw/ 2>/dev/null
```

Use the directory path you find (e.g. `/scratch/yqp5187/datasets/cicids2017`) as `CICIDS2017_DATA_PATH`.

---

## When data path is different on server (CIC-IDS2017)

If you see `No CIC-IDS2017 CSV files (*.pcap_ISCX.csv) found in data/raw/Bot-IoT` during 9:1 evaluation or training, data is in a different folder on the server.

**Option 1: Set path via environment variable**

On the server, set the directory that contains CIC-IDS2017 CSV files, then run:

```bash
export CICIDS2017_DATA_PATH=/scratch/yqp5187/your_path/   # folder that has *.pcap_ISCX.csv
python scripts/evaluate_9to1.py --config config/federated_local.yaml --model src/models/global_model.h5 --ratio 9
```

All scripts that read data (`tune_threshold_all_ratios.py`, `evaluate_ratio_sweep.py`, `run.py`, etc.) should use the same `CICIDS2017_DATA_PATH` when run.

**Option 2: Put data at config path**

Place `*.pcap_ISCX.csv` under `data/raw/Bot-IoT` (or config's `data.path`), or symlink that path to the real folder:

```bash
mkdir -p /scratch/yqp5187/TinyML-main/data/raw/Bot-IoT
# Copy CIC-IDS2017 CSV here or: ln -s /actual_path/*.pcap_ISCX.csv .
```

If data is not on the server yet, download from [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), extract to a folder on the server, and set that path as `CICIDS2017_DATA_PATH`.
