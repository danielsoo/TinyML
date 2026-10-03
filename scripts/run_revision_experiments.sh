#!/usr/bin/env bash
# Runs the LCTES revision experiments with the paper recipe (config/paper_v12/).
#   (a) centralized baseline, (b) FL near-IID + compression, Dirichlet(0.3) non-IID FL.
# Usage:
#   bash scripts/run_revision_experiments.sh --quick     # ~10-20 min smoke test (5 rounds x 1 epoch)
#   bash scripts/run_revision_experiments.sh            # full runs (many hours, CPU)
#   bash scripts/run_revision_experiments.sh --with-failed --with-scaling   # + fixed-LR row, 20/50 clients
#   bash scripts/run_revision_experiments.sh --config-dir config/tuning/<job>   # alternate configs
#   bash scripts/run_revision_experiments.sh --steps non_iid          # subset of: baseline_ablation,non_iid,client_scaling
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
if [ "$(ls data/raw/CIC-IDS2017/*.pcap_ISCX.csv 2>/dev/null | wc -l)" -ne 8 ]; then
  echo "❌ Need the 8 CIC-IDS2017 *.pcap_ISCX.csv files in data/raw/CIC-IDS2017/"; exit 1
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
  for md in "$OUT"/baseline/baseline_ablation.md "$OUT"/non_iid/non_iid_ablation.md "$OUT"/client_scaling/non_iid_ablation.md "$OUT"/compression_ablation/compression_ablation.md; do
    [ -f "$md" ] && { cat "$md"; echo; }
  done
} > "$OUT/SUMMARY.md"
echo
echo "Summary: $OUT/SUMMARY.md"
[ "$FAILED" -eq 0 ] && echo "All steps OK." || echo "Some steps failed (see logs above)."
exit "$FAILED"
