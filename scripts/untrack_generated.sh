#!/bin/bash
# 이미 커밋된 생성물/모델을 Git 추적에서만 제거 (로컬 파일은 유지)
# 사용: TinyML-main 루트에서  bash scripts/untrack_generated.sh

cd "$(dirname "$0")/.."

echo "Removing generated/binary files from Git tracking (files stay on disk)..."

for d in models outputs/pruning_demo outputs/test_pipeline data/processed/analysis data/processed/microcontroller; do
  [ -d "$d" ] && git rm -r --cached "$d" 2>/dev/null || true
done
for f in src/models/*.h5 training_log.txt; do
  [ -f "$f" ] && git rm --cached "$f" 2>/dev/null || true
done
[ -f esp32_tflite_project/src/model_data.c ] && git rm --cached esp32_tflite_project/src/model_data.c 2>/dev/null || true
[ -f esp32_tflite_project/src/model_data.h ] && git rm --cached esp32_tflite_project/src/model_data.h 2>/dev/null || true

git add .gitignore
echo "Done. Run:  git status   then   git commit -m 'chore: stop tracking generated/binary files'"
