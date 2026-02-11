# Conversation Application Verification

**Date:** 2026-02-10  
**Branch:** main (TinyML)  
**Last applied:** All changes re-applied (config, compression, run, analyze_compression, test_fgsm, evaluate_ratio_sweep, tune_threshold_all_ratios).

This document checks whether the code changes discussed in the conversation have been applied to the current codebase.

---

## ✅ Applied (현재 반영됨)

| Item | Status |
|------|--------|
| **No Korean in source** | `.py` / `.yaml` 파일에 한글 없음 |
| **config/federated.yaml** | `compression.always_build_traditional`, `traditional_model_path` + 영문 주석 |
| **compression.py** | 3종 TFLite (qat_pruned_float32, qat_ptq, no_qat_ptq), QAT 로드(quantize_scope), 영문 TFLite summary |
| **run.py** | Traditional 빌드(비QAT FL 1회), `--log`, 최종 요약에 3종 TFLite 출력 |
| **scripts/analyze_compression.py** | `_load_keras_h5`(quantize_scope), Pipeline overview 표, QAT+Prune only 등 기본 모델 목록 |
| **scripts/test_fgsm_attack.py** | QAT .h5 로드 시 quantize_scope |
| **scripts/evaluate_ratio_sweep.py** | .h5 로드 시 use_qat이면 quantize_scope |
| **scripts/tune_threshold_all_ratios.py** | 신규 생성, QAT+Prune only 등 스테이지 라벨, ratio sweep 후 append |
