#!/bin/bash
# 서버(PSU)의 TinyML-main 코드를 로컬로 가져옵니다.
# Usage: bash scripts/sync_from_psu.sh

set -e
cd "$(dirname "$0")/.."
LOCAL_ROOT="$(pwd)"

SERVER="yqp5187@e5-cse-135-01.cse.psu.edu"
REMOTE="/scratch/yqp5187/TinyML-main"

echo "=== PSU server -> Local (코드 가져오기) ==="
echo "Server: $SERVER:$REMOTE"
echo "Local:  $LOCAL_ROOT"
echo ""

rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='data/raw' --exclude='data/processed' --exclude='* 2.*' --exclude='* 2' --exclude='TinyML-results' --exclude='outputs' "$SERVER:$REMOTE/" "$LOCAL_ROOT/"

echo ""
echo "서버 코드를 로컬로 복사 완료."
echo "로컬과 서버가 같은지 확인: bash scripts/verify_sync.sh"
