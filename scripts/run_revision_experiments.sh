#!/usr/bin/env bash
# Runs the LCTES revision experiments with the paper recipe (config/paper_v12/).
#   (a) centralized baseline, (b) FL near-IID + compression, Dirichlet(0.3) non-IID FL.
# Usage:
#   bash scripts/run_revision_experiments.sh --quick     # ~10-20 min smoke test (5 rounds x 1 epoch)
#   bash scripts/run_revision_experiments.sh            # full runs (many hours, CPU)
#   bash scripts/run_revision_experiments.sh --with-failed --with-scaling   # + fixed-LR row, 20/50 clients
#   bash scripts/run_revision_experiments.sh --config-dir config/tuning/<job>   # alternate configs
#   bash scripts/run_revision_experiments.sh --steps non_iid          # subset of: baseline_ablation,non_iid,client_scaling,
#                                                                     #   fixed_lr,robustness,compression_ablation,quant_distill,local_only,ptq_calibration,qat_stability,compression_combos,recall_priority
#   bash scripts/run_revision_experiments.sh --steps compression_ablation --config-dir <dir with compression_ablation.yaml>
# Re-running with the same --out resumes: finished steps are skipped.
set -uo pipefail

VENV="${VENV:-$HOME/tinyml-venv}"
QUICK=""; WITH_FAILED=""; WITH_SCALING=0; OUT=""; CONFIG_DIR="config/paper_v12"; STEPS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK="--quick" ;;
    --with-failed) WITH_FAILED="--with-failed" ;;
    --with-scaling) WITH_SCALING=1 ;;
    --out) OUT="$2"; shift ;;
    --config-dir) CONFIG_DIR="$2"; shift ;;
    --steps) STEPS="$2"; shift ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
  shift
done

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
# Exit code 3 = dataset not in place yet (pc_worker.sh retries the job later)
DATA_NAME=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['data']['name'])" "$CONFIG_DIR/fl_baseline.yaml" 2>/dev/null || echo cicids2017)
DATA_PATH=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['data']['path'])" "$CONFIG_DIR/fl_baseline.yaml" 2>/dev/null || echo data/raw/CIC-IDS2017)
if [ "$DATA_NAME" = "cicids2017" ]; then
  if [ "$(ls "$DATA_PATH"/*.pcap_ISCX.csv 2>/dev/null | wc -l)" -ne 8 ]; then
    echo "❌ Need the 8 CIC-IDS2017 *.pcap_ISCX.csv files in $DATA_PATH/"; exit 3
  fi
elif [ -z "$(find -L "$DATA_PATH" -name '*.csv' 2>/dev/null | head -1)" ]; then
  echo "❌ No $DATA_NAME CSV files under $DATA_PATH/"; exit 3
fi

TAG=$([ -n "$QUICK" ] && echo quick || echo full)
OUT="${OUT:-data/processed/revision/$(date +%Y-%m-%d_%H-%M-%S)_$TAG}"
mkdir -p "$OUT"
echo "Output: $OUT"
{ git rev-parse HEAD 2>/dev/null || git.exe rev-parse HEAD 2>/dev/null; } | tr -d '\r' > "$OUT/git_commit.txt" || true
mkdir -p "$OUT/configs" && cp "$CONFIG_DIR"/*.yaml "$OUT/configs/" 2>/dev/null || true

step() {  # step <name> <command...>
  local name="$1"; shift
  if [ -f "$OUT/$name.done" ]; then echo "⏭  $name already done"; return 0; fi
  echo "▶ $name  ($(date '+%F %T'))"
  local t0=$SECONDS
  if "$@" > "$OUT/$name.log" 2>&1; then
    echo "$((SECONDS - t0))" > "$OUT/$name.done"
    echo "✅ $name finished in $(( (SECONDS - t0) / 60 )) min"
  else
    echo "❌ $name failed after $(( (SECONDS - t0) / 60 )) min — see $OUT/$name.log (last lines below)"
    tail -n 25 "$OUT/$name.log"
    return 1
  fi
}

if [ -z "$STEPS" ]; then
  STEPS="baseline_ablation,non_iid"
  [ "$WITH_SCALING" -eq 1 ] && STEPS="$STEPS,client_scaling"
fi
wanted() { [[ ",$STEPS," == *",$1,"* ]]; }

FAILED=0
if wanted baseline_ablation; then
  step baseline_ablation python scripts/run_baseline_ablation.py \
    --config-dir "$CONFIG_DIR" --output-dir "$OUT/baseline" $QUICK $WITH_FAILED || FAILED=1
fi
if wanted non_iid; then
  step non_iid python scripts/run_non_iid_ablation.py \
    --base-config "$CONFIG_DIR/fl_baseline.yaml" --strategies dirichlet --client-counts 4 \
    --output-dir "$OUT/non_iid" $QUICK || FAILED=1
fi
if wanted fixed_lr; then
  # Table 2 row (b): same recipe with a fixed learning rate (failed_config.yaml)
  step fixed_lr python scripts/run_non_iid_ablation.py \
    --base-config "$CONFIG_DIR/failed_config.yaml" --strategies label_balanced --client-counts 4 \
    --output-dir "$OUT/fixed_lr" $QUICK || FAILED=1
fi
if wanted robustness; then
  step robustness python scripts/run_robustness.py \
    --spec "$CONFIG_DIR/robustness.yaml" --output-dir "$OUT/robustness" || FAILED=1
fi
if wanted quant_distill; then
  step quant_distill python scripts/quant_distill_ablation.py \
    --spec "$CONFIG_DIR/quant_distill.yaml" --output-dir "$OUT/quant_distill" || FAILED=1
fi
if wanted local_only; then
  step local_only python scripts/local_only_baseline.py \
    --spec "$CONFIG_DIR/local_only.yaml" --output-dir "$OUT/local_only" || FAILED=1
fi
if wanted ptq_calibration; then
  step ptq_calibration python scripts/ptq_calibration_check.py \
    --spec "$CONFIG_DIR/ptq_calibration.yaml" --output-dir "$OUT/ptq_calibration" || FAILED=1
fi
if wanted qat_stability; then
  step qat_stability python scripts/qat_stability_check.py \
    --spec "$CONFIG_DIR/qat_stability.yaml" --output-dir "$OUT/qat_stability" || FAILED=1
fi
if wanted compression_combos; then
  step compression_combos python scripts/compression_combos.py \
    --spec "$CONFIG_DIR/compression_combos.yaml" --output-dir "$OUT/compression_combos" || FAILED=1
fi
if wanted recall_priority; then
  step recall_priority python scripts/recall_priority.py \
    --spec "$CONFIG_DIR/recall_priority.yaml" --output-dir "$OUT/recall_priority" || FAILED=1
fi
if wanted compression_ablation; then
  step compression_ablation python scripts/compression_ablation.py \
    --spec "$CONFIG_DIR/compression_ablation.yaml" --output-dir "$OUT/compression_ablation" || FAILED=1
fi
if wanted client_scaling; then
  step client_scaling python scripts/run_non_iid_ablation.py \
    --base-config "$CONFIG_DIR/fl_baseline.yaml" --strategies label_balanced,dirichlet \
    --client-counts 20,50 --output-dir "$OUT/client_scaling" $QUICK || FAILED=1
fi

{
  echo "# Revision experiments ($TAG)"
  echo
  echo "- commit: $(cat "$OUT/git_commit.txt" 2>/dev/null)"
  echo "- configs: $CONFIG_DIR (eval_split: $(grep -h "eval_split" "$OUT"/configs/*.yaml 2>/dev/null | sort -u | tr -d ' ' | tr '\n' ' ' || true))"
  echo "- host: $(uname -srm), $(nproc) cores"
  for f in "$OUT"/*.done; do [ -f "$f" ] && echo "- $(basename "$f" .done): $(( $(cat "$f") / 60 )) min"; done
  echo
  for md in "$OUT"/baseline/baseline_ablation.md "$OUT"/non_iid/non_iid_ablation.md "$OUT"/client_scaling/non_iid_ablation.md "$OUT"/compression_ablation/compression_ablation.md "$OUT"/fixed_lr/non_iid_ablation.md "$OUT"/robustness/robustness.md "$OUT"/quant_distill/quant_distill.md "$OUT"/local_only/local_only.md "$OUT"/ptq_calibration/ptq_calibration.md "$OUT"/qat_stability/qat_stability.md "$OUT"/compression_combos/compression_combos.md "$OUT"/recall_priority/recall_priority.md; do
    [ -f "$md" ] && { cat "$md"; echo; }
  done
} > "$OUT/SUMMARY.md"
echo
echo "Summary: $OUT/SUMMARY.md"
[ "$FAILED" -eq 0 ] && echo "All steps OK." || echo "Some steps failed (see logs above)."
exit "$FAILED"
