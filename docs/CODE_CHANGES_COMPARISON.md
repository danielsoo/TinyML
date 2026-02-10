# 코드 변경 비교: 이전 대화 합의 vs 이번 적용

**이전 대화**에서 합의했던 항목과 **이번에 실제로 적용한** 항목을 나란히 정리한 문서입니다.

---

## 1. 이전 대화에서 합의/언급된 항목 (요약 기준)

대화 요약(conversation summary)에 있던 내용:

| # | 항목 | 설명 |
|---|------|------|
| A1 | **Pruning** | `_to_single_tensor()`, 레이어마다 tuple unwrapping, `_backend_for_model()` |
| A2 | **TFLite export** | `convert()` / `_convert_and_save()`를 temp dir 안에서 실행해 SavedModel 존재 시 변환 |
| A3 | **QAT 모델 로드** | `analyze_compression.py`, `test_fgsm_attack.py`에서 `.h5` 로드 시 `quantize_scope` 사용 |
| A4 | **3종 TFLite** | `saved_model_qat_pruned_float32.tflite`, `saved_model_qat_ptq.tflite`, `saved_model_no_qat_ptq.tflite` + Traditional 모델 없을 때 자동 빌드 |
| A5 | **Config** | `federated.yaml`에 `always_build_traditional`, `traditional_model_path` + 영문 주석 |
| A6 | **경고 완화** | compression.py의 strip_quantization 메시지 완화, client.py의 `fit_metrics_aggregation_fn`·HDF5 경고 필터, TFLite Interpreter deprecation 및 `experimental_preserve_all_tensors=False` + 경고 필터 (evaluate_ratio_sweep, analyze_compression) |
| A7 | **로깅** | `run.py --log <path>` 로 stdout/stderr tee |
| A8 | **메트릭** | Ratio sweep / threshold tuning 콘솔·보고서에 Attack Recall/Precision (AtkRec, AtkPrec) |
| A9 | **보고서** | `analyze_compression.py` 마크다운에 "각 단계별 파이프라인" → 영문 "Pipeline overview" 섹션 |
| A10 | **코드·주석 영문화** | 주석, print, config 주석, 보고서 생성 문구 등 한글 → 영어 |

---

## 2. 이번에 실제로 적용한 항목

"다시 다 적용해줄래?" 요청으로 수정·추가한 내용:

| # | 파일/위치 | 적용 내용 |
|---|-----------|-----------|
| B1 | **config/federated.yaml** | `compression` 블록 추가: `always_build_traditional: true`, `traditional_model_path: null`, 영문 주석 |
| B2 | **compression.py** | (1) QAT 시 `quantize_scope`로 `.h5` 로드 (2) 3종 TFLite 생성: `saved_model_qat_pruned_float32.tflite`, `saved_model_qat_ptq.tflite`, `saved_model_no_qat_ptq.tflite` (3) config의 `traditional_model_path`로 Traditional 모델 로드 후 prune→PTQ (4) 영문 TFLite summary 출력 |
| B3 | **run.py** | (1) `--log <path>` 추가, tee 처리 (2) `use_qat` + `always_build_traditional`이고 Traditional 없으면 비QAT FL 1회 실행 → Traditional 저장 후 QAT 모델 복원 (3) 최종 요약에 3종 TFLite 파일명 출력 |
| B4 | **scripts/analyze_compression.py** | (1) `_load_keras_h5()` 도입, config `use_qat`이면 `quantize_scope`로 로드 (2) `.h5` 로드하는 곳에서 `_load_keras_h5` 사용 (3) 마크다운에 "Pipeline overview (How each stage is produced)" 표 추가 (4) 기본 모델 목록에 QAT+Prune only, QAT+PTQ, noQAT+PTQ 등 추가 |
| B5 | **scripts/test_fgsm_attack.py** | `.h5` 로드 시 `quantize_scope` 시도 후 실패하면 일반 `load_model` |
| B6 | **scripts/evaluate_ratio_sweep.py** | config `use_qat`이면 `.h5` 로드 시 `quantize_scope` 사용 |
| B7 | **scripts/tune_threshold_all_ratios.py** | 신규 생성: 여러 ratio에 대해 `tune_threshold.py` 실행 후 `ratio_sweep_report.md`에 append, 모델 경로→스테이지 라벨(QAT+Prune only 등) |

---

## 3. 대조표: 이전 합의 vs 이번 적용

| 이전 합의 (위 A#) | 이번 적용 (위 B#) | 비고 |
|-------------------|-------------------|------|
| A3 QAT 로드 | B2, B4, B5, B6 | ✅ 동일 방향으로 적용 |
| A4 3종 TFLite + Traditional 빌드 | B2, B3 | ✅ 적용 |
| A5 Config | B1 | ✅ 적용 |
| A7 --log | B3 | ✅ 적용 |
| A9 Pipeline overview (영문) | B4 | ✅ 적용 |
| A10 영문화 | B2 요약 문구 등 | ✅ 적용 (이미 한글 제거된 상태에서 영문 문구 추가) |
| A1 Pruning (_to_single_tensor, tuple unwrap, _backend_for_model) | — | ❌ 이번에 적용 안 함 (pruning.py 미수정) |
| A2 TFLite export (temp dir에서 convert) | — | ❌ 이번에 적용 안 함 (export_tflite.py 미수정) |
| A6 경고 완화 (strip_quantization, client HDF5, TFLite deprecation 등) | — | ❌ 이번에 적용 안 함 |
| A8 AtkRec/AtkPrec 메트릭 | — | ❌ 이번에 적용 안 함 |

---

## 4. 요약

- **이번에 넣은 코드**는 **이전에 합의했던 항목 중 일부**만 반영한 것입니다.  
  - 반영된 것: **config, 3종 TFLite, Traditional 빌드, QAT 로드(quantize_scope), --log, Pipeline overview, 기본 모델 목록, tune_threshold_all_ratios, 영문 문구**  
  - 반영 안 된 것: **Pruning 쪽 수정, TFLite export temp dir, 각종 경고 완화, AtkRec/AtkPrec 메트릭**
- **“전에 준 코드들 그대로”**라고 하면, **A3~A5, A7, A9, A10**에 해당하는 부분은 이번 적용(B1~B7)과 **같은 의도로** 넣었고,  
  **A1, A2, A6, A8**은 이번 단계에서는 건드리지 않았습니다.

원하면 A1, A2, A6, A8도 같은 문서 형식으로 “넣을 코드 vs 이미 넣은 코드”로 구체적으로 나열해 줄 수 있습니다.
