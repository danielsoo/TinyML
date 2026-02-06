#!/usr/bin/env bash
# On server (/scratch/yqp5187): remove only safe targets to free space
# Run: bash scripts/cleanup_scratch.sh
# Option: --dry-run to list targets without deleting

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/yqp5187}"
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo "[DRY-RUN] Listing targets only (no delete)."
  echo ""
fi

del() {
  local target="$1"
  local desc="${2:-}"
  if [[ ! -e "$target" ]]; then
    return 0
  fi
  local size
  size=$(du -sh "$target" 2>/dev/null | cut -f1)
  if [[ "$DRY_RUN" == true ]]; then
    echo "  [would delete] $target ($size) ${desc}"
  else
    echo "  Deleting: $target ($size)"
    rm -rf "$target"
  fi
}

echo "=== Server disk cleanup (SCRATCH=$SCRATCH) ==="
echo ""

# 1) Stop Ray then clean tmp (skip if experiment in progress)
if command -v ray >/dev/null 2>&1; then
  ray stop --force 2>/dev/null || true
fi
if [[ -d "$SCRATCH/tmp" ]]; then
  size=$(du -sh "$SCRATCH/tmp" 2>/dev/null | cut -f1)
  if [[ "$DRY_RUN" == true ]]; then
    echo "  [would delete] $SCRATCH/tmp contents ($size) Ray etc."
  else
    echo "  Deleting: $SCRATCH/tmp contents ($size)"
    rm -rf "$SCRATCH/tmp"
    mkdir -p "$SCRATCH/tmp"
  fi
fi

# 2) Old miniconda backup
del "$SCRATCH/miniconda3.old" "Old miniconda backup (safe to delete if conda works)"

# 3) Miniconda installer (unneeded if already installed)
del "$SCRATCH/Miniconda3-latest-Linux-x86_64.sh" "miniconda installer"

echo ""
echo "=== TinyML-main internal (optional) ==="
echo "To remove old runs, uncomment and run the lines below."
echo "  rm -rf $SCRATCH/TinyML-main/data/processed/runs/v2_*"
echo "  rm -rf $SCRATCH/TinyML-main/data/processed/runs/v3"
echo "  rm -rf $SCRATCH/TinyML-main/data/processed/runs/v4"
echo "  rm -rf $SCRATCH/TinyML-main/data/processed/runs/v5"
echo "  rm -rf $SCRATCH/TinyML-main/data/processed/runs/v6"
echo "  rm -rf $SCRATCH/TinyML-main/data/processed/runs/v7"
echo "  rm -rf $SCRATCH/TinyML-main/data/processed/runs/v8"
echo "  rm -rf $SCRATCH/TinyML-main/data/processed/runs/v8_centralized"
echo "  rm -rf $SCRATCH/TinyML-main/data/processed/runs/v9"
echo ""
if [[ "$DRY_RUN" == true ]]; then
  echo "To actually delete: bash scripts/cleanup_scratch.sh"
fi
echo "Done."
