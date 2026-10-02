#!/usr/bin/env bash
# One-time setup inside WSL2 Ubuntu (see docs/LOCAL_PC_EXPERIMENTS_GUIDE.md).
# Usage (from the repo folder): bash scripts/wsl_setup.sh
set -euo pipefail

VENV="${VENV:-$HOME/tinyml-venv}"
cd "$(dirname "$0")/.."

echo "== 1/3 system packages (sudo password may be asked)"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git

echo "== 2/3 python venv at $VENV (CPU TensorFlow; FL simulation runs on CPU by design)"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

echo "== 3/3 checks"
python - <<'PY'
import os, multiprocessing
import tensorflow as tf, flwr, ray
print(f"TensorFlow {tf.__version__}, Flower {flwr.__version__}, Ray {ray.__version__}")
print(f"CPU cores visible to WSL: {multiprocessing.cpu_count()}")
mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
print(f"RAM visible to WSL: {mem:.1f} GB" + ("  <-- raise to 24GB via .wslconfig (see guide)" if mem < 20 else ""))
PY
n=$(ls data/raw/CIC-IDS2017/*.pcap_ISCX.csv 2>/dev/null | wc -l)
if [ "$n" -eq 8 ]; then
  echo "CIC-IDS2017: 8 CSV files found ✅"
else
  echo "CIC-IDS2017: found $n/8 *.pcap_ISCX.csv in data/raw/CIC-IDS2017/ ❌ (see guide step 3)"
fi
echo "Done. Next: bash scripts/run_revision_experiments.sh --quick"
