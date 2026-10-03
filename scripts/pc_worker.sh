#!/usr/bin/env bash
# PC-side worker for experiments/queue (see experiments/README.md).
# Start in WSL from the repo folder:  nohup bash scripts/pc_worker.sh > worker.out 2>&1 &
# Stop:                               pkill -f pc_worker.sh
# git runs through Windows git (git.exe) so it uses the Windows login for push.
set -uo pipefail

VENV="${VENV:-$HOME/tinyml-venv}"
POLL_SECONDS="${POLL_SECONDS:-600}"
cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"
# shellcheck disable=SC1091
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

if [ -n "${GIT_CMD:-}" ]; then
  GIT=("$GIT_CMD")
elif command -v git.exe >/dev/null 2>&1; then
  GIT=(git.exe)
elif [ -x "/mnt/c/Program Files/Git/cmd/git.exe" ]; then
  GIT=("/mnt/c/Program Files/Git/cmd/git.exe")
else
  echo "❌ Windows git.exe not found"; exit 1
fi
git_() { "${GIT[@]}" "$@"; }
BRANCH="$(git_ rev-parse --abbrev-ref HEAD | tr -d '\r')"
log() { echo "[$(date '+%F %T')] $*"; }

push_results() {  # push_results <results dir> <message>
  local dir="$1" msg="$2"
  git_ add -- "$dir" >/dev/null
  if git_ diff --cached --quiet -- "$dir"; then return 0; fi
  git_ commit -q -m "$msg" -- "$dir" || { log "commit failed"; return 1; }
  for i in 1 2 3 4; do
    git_ pull -q --rebase --autostash origin "$BRANCH" && git_ push -q origin "$BRANCH" && { log "pushed: $msg"; return 0; }
    sleep $((i * 15))
  done
  log "❌ push failed (results are committed locally; will retry next poll)"
  return 1
}

job_field() {  # job_field <file> <key>  -> prints value (lists joined by spaces)
  python3 - "$1" "$2" <<'PY'
import sys, yaml
job = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
v = job.get(sys.argv[2], "")
print(" ".join(map(str, v)) if isinstance(v, list) else ("" if v is None else v))
PY
}

run_job() {
  local f="$1" id cfg steps flags out arg
  id="$(job_field "$f" id)"; cfg="$(job_field "$f" config_dir)"
  steps="$(job_field "$f" steps)"; flags="$(job_field "$f" flags)"
  [[ "$id" =~ ^[A-Za-z0-9._-]+$ ]] || { log "skip $f: bad id '$id'"; return; }
  [[ "$cfg" =~ ^config/[A-Za-z0-9._/-]+$ && "$cfg" != *..* && -d "$cfg" ]] || { log "skip $id: bad config_dir '$cfg'"; return; }
  [[ -z "$steps" || "$steps" =~ ^[a-z_,]+$ ]] || { log "skip $id: bad steps '$steps'"; return; }
  local args=(--config-dir "$cfg" --out "data/processed/revision/$id")
  [ -n "$steps" ] && args+=(--steps "$steps")
  for arg in $flags; do
    case "$arg" in --quick|--with-failed|--with-scaling) args+=("$arg") ;; *) log "skip $id: flag '$arg' not allowed"; return ;; esac
  done
  out="data/processed/revision/$id"
  mkdir -p "$out"; cp "$f" "$out/job.yaml"
  log "▶ job $id: ${args[*]}"
  bash scripts/run_revision_experiments.sh "${args[@]}" > "$out/runner.out" 2>&1
  local rc=$?
  if [ "$rc" -eq 3 ]; then
    log "job $id waiting for its dataset: $(tail -n 1 "$out/runner.out")"
    return 3
  fi
  if [ "$rc" -eq 0 ]; then echo ok > "$out/JOB_STATUS"; else echo failed > "$out/JOB_STATUS"; fi
  log "job $id finished: $(cat "$out/JOB_STATUS")"
  push_results "$out" "Results: $id ($(cat "$out/JOB_STATUS"))"
}

SELF_SUM="$(md5sum scripts/pc_worker.sh | cut -d' ' -f1)"
log "worker started in $REPO_DIR on branch $BRANCH (poll every ${POLL_SECONDS}s)"
while true; do
  if pgrep -f "^bash scripts/run_revision_experiments.sh" >/dev/null; then
    log "another experiment run is active; waiting"
  else
    git_ pull -q --rebase --autostash origin "$BRANCH" || log "pull failed (will retry)"
    if [ "$(md5sum scripts/pc_worker.sh | cut -d' ' -f1)" != "$SELF_SUM" ]; then
      log "pc_worker.sh changed upstream; restarting"
      exec bash scripts/pc_worker.sh
    fi
    # Finished manual runs (SUMMARY.md present) that were never pushed
    for d in data/processed/revision/*/; do
      [ -f "$d/SUMMARY.md" ] || continue
      if [ -n "$(git_ status --porcelain -- "$d" | head -1)" ]; then
        push_results "$d" "Results: $(basename "$d")"
      fi
    done
    for f in experiments/queue/*.yaml; do
      [ -f "$f" ] || continue
      id="$(job_field "$f" id)"
      [ -f "data/processed/revision/$id/JOB_STATUS" ] && continue
      run_job "$f"
      [ $? -eq 3 ] && continue   # dataset missing: try the next job, retry this one later
      break   # re-pull before the next job so new instructions are picked up
    done
  fi
  sleep "$POLL_SECONDS"
done
