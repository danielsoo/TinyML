# TinyML Research Results

## Folder Structure (version/date_time)

```
TinyML-results/
├── processed/
│   ├── analysis/           # Compression analysis results
│   │   └── {version}/      # v1, v2, v3...
│   │       └── {date_time}/ # 2026-01-30_14-34-39
│   │           ├── compression_analysis.md
│   │           ├── compression_analysis.json
│   │           └── *.png
│   │
│   └── runs/               # Full snapshots (analysis + models + outputs)
│       ├── RUNS.md        # Run history
│       └── {version}/
│           └── {date_time}/
│               ├── analysis/
│               ├── models/
│               └── outputs/
│
├── models/                 # Latest models (root, for convenience)
├── outputs/                # Latest FL outputs (root, for convenience)
└── README.md
```

## Download from Server (전체 복구)

로컬 데이터가 망가졌을 때 서버에서 전부 받아오기:

```bash
cd /path/to/TinyML-results
chmod +x scripts/sync_from_server.sh
./scripts/sync_from_server.sh
```

또는 수동으로:

```bash
SERVER="yqp5187@e5-cse-135-01.cse.psu.edu"
REMOTE="/scratch/yqp5187/TinyML-main/data"

# processed 전체 (analysis + runs)
rsync -avz --progress $SERVER:$REMOTE/processed/analysis/   TinyML-results/processed/analysis/
rsync -avz --progress $SERVER:$REMOTE/processed/runs/       TinyML-results/processed/runs/

# 서버에 있으면 models, outputs도
rsync -avz --progress $SERVER:$REMOTE/models/   TinyML-results/models/   2>/dev/null || true
rsync -avz --progress $SERVER:$REMOTE/outputs/ TinyML-results/outputs/  2>/dev/null || true
```
