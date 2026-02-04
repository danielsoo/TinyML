#!/bin/bash
# 로컬과 PSU 서버의 파일이 정확히 같은지 확인합니다.
# Usage: bash scripts/verify_sync.sh

set -e
cd "$(dirname "$0")/.."
LOCAL_ROOT="$(pwd)"

SERVER="yqp5187@e5-cse-135-01.cse.psu.edu"
REMOTE="/scratch/yqp5187/TinyML-main"

echo "=== 로컬 vs 서버 동기화 확인 ==="
echo "Local:  $LOCAL_ROOT"
echo "Server: $SERVER:$REMOTE"
echo ""

echo "내용(checksum) 기준으로 비교 중..."
OUTPUT=$(rsync -avnc --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='data/raw' --exclude='data/processed' --exclude='* 2.*' --exclude='* 2' --exclude='TinyML-results' --exclude='outputs' "$SERVER:$REMOTE/" "$LOCAL_ROOT/" 2>/dev/null || true)

DIFF_FILES=$(echo "$OUTPUT" | grep -E '^\./.' | grep -v '^total size' | grep -v '^sent ' | grep -v '^received ' | grep -v 'speedup' || true)

if [ -z "$DIFF_FILES" ]; then
  echo ""
  echo "로컬과 서버가 동일합니다. (checksum 기준)"
  exit 0
fi

echo ""
echo "로컬과 서버가 다릅니다. 일치하지 않는 파일:"
echo "$DIFF_FILES"
echo ""
echo "서버->로컬: bash scripts/sync_from_psu.sh"
echo "로컬->서버: bash scripts/sync_to_psu.sh"
exit 1
