# run.py 파이프라인 점검 (실행본 2026-02-10_17-59-49 기준)

## 1. run.py가 하는 단계 (현재 코드)

| Step | 내용 | 출력 위치 |
|------|------|-----------|
| 1 | FL 학습 (또는 Centralized) | `src/models/global_model.h5` → 복사 → `models/global_model.h5` |
| 1+ | Traditional 빌드 (use_qat + always_build_traditional이고 traditional 없을 때) | `models/global_model_traditional.h5` |
| 2 | 압축 (compression.py --use-trained) | `models/tflite/*.tflite`, `models/test_pruned_model.h5` |
| 3 | 압축 분석 (analyze_compression.py) | `runs/<version>/<run_id>/analysis/` |
| 4 | Ratio sweep (evaluate_ratio_sweep.py) | `runs/.../eval/ratio_sweep_report.md` |
| 4b | Threshold tuning (tune_threshold_all_ratios.py) | 같은 report에 append |
| 5 | 시각화 (visualize_results.py) | `runs/.../analysis/*.png` |
| 마지막 | models, outputs 복사 | `runs/.../models/`, `runs/.../outputs/` |

**FGSM은 run.py에 없음** → 별도 스크립트/호출로 실행된 것으로 보임.

---

## 2. 실행본 2026-02-10_17-59-49 상태

- **디렉터리 구조**: analysis/, eval/, fgsm/, models/, outputs/, run_config.yaml ✅
- **models/tflite/**: saved_model_original, saved_model_qat_pruned_float32, saved_model_qat_ptq, saved_model_no_qat_ptq, saved_model_pruned_quantized, saved_model_pruned_qat ✅
- **models/**: global_model.h5, global_model_traditional.h5, test_pruned_model.h5 ✅
- **analysis/**: compression_analysis.csv/json/md, png 3개 ✅
- **eval/**: ratio_sweep_report.md ✅
- **fgsm/**: fgsm_report.md, fgsm_results.json ✅ (run.py 외부에서 생성된 것으로 추정)

→ run.py로 생성되는 부분은 **전부 정상 생성된 상태**로 보임.

---

## 3. 확인된 이슈

### 3.1 분석 보고서에 한글 스테이지명

- `analysis/compression_analysis.csv` 및 `.md`에 **"QAT+일반압축"** 이 들어 있음.
- 이 실행은 **예전 코드**(기본 모델 목록에 한글 라벨 쓰이던 버전)로 돌린 결과로 보임.
- **지금 코드**에서는 기본 모델 목록이 `"QAT+Prune only:..."` 로 되어 있어, 같은 run을 다시 돌리면 **"QAT+Prune only"** 로 나옴. → 추가 수정 없이 괜찮음.

### 3.2 run_config.yaml의 use_qat

- 저장된 `run_config.yaml`에 `use_qat: false`, `use_real_qat: true` 가 함께 있음.
- run.py는 **`use_qat`만** 보고 Traditional 빌드 여부를 결정함.
- 실제로는 Traditional 모델·noQAT+PTQ가 생성되어 있으므로, 실행 시점의 **원본 config**에는 `use_qat: true` 였을 가능성이 큼. (저장된 run_config는 중간/다른 설정이 섞였을 수 있음.)

### 3.3 Step 4에서 사용하는 모델 1개

- run.py Step 4는 **모델 경로를 하나만** 넘김: 기본값 `models/tflite/saved_model_original.tflite`.
- 그런데 실행본의 `eval/ratio_sweep_report.md`에는 **5개 모델** (Original, QAT+Prune only, Compressed (QAT), QAT+PTQ, noQAT+PTQ)에 대한 결과가 있음.
- 따라서 **이번 실행**에서는 run.py가 아니라, **여러 모델을 돌리는 다른 스크립트/방식**으로 ratio sweep이 돌아갔을 가능성이 있음. (또는 예전 run.py가 여러 모델을 넘기던 버전이었을 수 있음.)
- **현재 run.py**만 쓸 경우: ratio sweep·threshold tuning은 **지정한 1개 모델**에 대해서만 수행됨.

---

## 4. 결론

- **run을 통해서 파일이 돌아가는 것**은, 실행본 `2026-02-10_17-59-49` 기준으로 보면 **전반적으로 괜찮음**.
  - 3종 TFLite + Traditional, 분석, eval, 시각화, 복사까지 모두 생성됨.
- 다만:
  1. **FGSM**은 run.py에 없으므로, fgsm/ 결과는 별도 실행으로 생성된 것임.
  2. **Ratio sweep 5개 모델**은 현재 run.py(단일 모델)와는 다른 경로로 돌아간 실행으로 보는 것이 맞음.
  3. **한글 "QAT+일반압축"** 은 예전 실행 결과이고, 현재 코드로 다시 돌리면 영문으로 나옴.

원하면 run.py에 **FGSM 단계 추가**나 **ratio sweep 다중 모델** 지원도 정리해 줄 수 있음.
