# Compression Grid Sweep (fl_qat × distillation × pruning × ptq)

## 목표 테이블 형식

한 run에서 **fl_qat × distillation × pruning × ptq** 조합 전체를 돌려, 아래 컬럼으로 CSV/테이블을 만드는 것.

| 구분 | 컬럼 | 설명 |
|------|------|------|
| **태그** | tag | `{no|yes}_qat__distill_{none|direct|progressive}__prune_{none|10x5|10x2|5x10}__ptq_{yes|no}` |
| **조합** | fl_qat, distillation, pruning, ptq | 각각 True/False, none|direct|progressive, prune_none|10x5|10x2|5x10, True/False |
| **FL** | fl_acc, fl_f1, fl_prec, fl_rec, fl_size_kb | FL 학습 직후 모델 성능·크기 |
| **디스틸** | dist_acc, dist_f1 | distillation 후 성능 (none이면 비움) |
| **프루닝** | prune_acc, prune_f1 | pruning 후 성능 |
| **최종** | final_acc, final_f1, final_prec, final_rec | 최종 TFLite 평가 |
| **크기** | tflite_size_kb, final_size_kb | TFLite 파일 크기 등 |

## 추가하려는 항목 (우리가 넣었던 것)

- **AT (Adversarial Training)**  
  - `at_enabled`, `at_attack` (fgsm|pgd) — 스윕 CSV에 포함됨.
- **PGD**  
  - 해당 조합 TFLite에 대한 PGD 공격 결과.  
  - 컬럼: `pgd_adv_acc` (비어 있으면 run_pgd 결과를 매핑해 채울 수 있음).
- **Ratio sweep**  
  - normal:attack 비율 스윕 결과.  
  - 컬럼: `ratio_sweep_f1_50_50` (비어 있으면 ratio_sweep 리포트에서 채울 수 있음).

이렇게 하면 **스윕 방식**은 “한 run에 48행(또는 서브셋) × (기존 메트릭 + AT + PGD + ratio_sweep)”이 됨.

## 조합 수

- fl_qat: 2 (no_qat, yes_qat)
- distillation: 3 (none, direct, progressive)
- pruning: 4 (none, 10x5, 10x2, 5x10)
- ptq: 2 (no, yes)  
→ **2×3×4×2 = 48** 조합.

## 구현 방향

1. **스윕 스크립트** (`scripts/sweep_compression_grid.py` 예정)  
   - 48조합(또는 설정 가능 서브셋) 루프  
   - 각 조합별로: FL 모델 선택 → (선택) 디스틸 → (선택) 프루닝 → (선택) PTQ → 평가  
   - 행 단위로 fl_*, dist_*, prune_*, final_*, tflite_size_kb, final_size_kb 수집  
   - 출력: `sweep_compression_grid.csv` (및 선택적으로 .md)

2. **AT 반영**  
   - config에 `adversarial_training.enabled` 등이 있으면, FL 직후 한 번 AT 적용한 모델을 “FL” 대신 사용할지, 또는 AT 적용 여부를 컬럼으로만 넣을지 선택  
   - 테이블에는 최소한 `at_enabled`, `at_attack` 컬럼 추가

3. **PGD 반영**  
   - 48개 TFLite에 대해 PGD(또는 FGSM) 실행  
   - 결과를 pgd_report/pgd_results.json 등에서 읽어와서 `pgd_adv_acc`, `pgd_success_rate` 등 컬럼으로 추가

4. **Ratio sweep 반영**  
   - 각 조합 TFLite에 대해 ratio sweep 실행  
   - 50:50 또는 여러 비율의 f1 등을 컬럼으로 추가 (예: `ratio_sweep_f1_50_50`)

현재 run.py는 “한 경로”만 돌리므로, 위 테이블을 만들려면 **이 스윕 전용 스크립트**가 필요함.  
이 문서는 그 스윕 방식과 추가할 컬럼(AT, PGD, ratio sweep) 정의용이다.

## 사용법 (구현됨)

- **전제:** FL 학습 + (선택) `run_distillation_first.py`로 `models/global_model.h5`, `models/global_model_traditional.h5`, `models/distilled/*.h5` 존재.

1. **48 스윕 + PGD 전체 + 병합 한 번에**
   ```bash
   python scripts/run_sweep_and_pgd.py --config config/federated_local_sky.yaml --version v1_PGD
   ```
   - `data/processed/runs/v1_PGD/<timestamp>/` 에 생성: `sweep_compression_grid.csv`, `pgd_model_list.txt`, `pgd/pgd_report.md`, `pgd/pgd_results.json`, `sweep_compression_grid_with_pgd.csv`.

2. **단계별**
   - 스윕만: `python scripts/sweep_compression_grid.py --config ... --run-dir data/processed/runs/v1_PGD/sweep_001`
   - PGD만: `python scripts/run_pgd.py --models-file .../pgd_model_list.txt --config ... --output-dir .../pgd`
   - 병합만: `python scripts/merge_sweep_pgd.py --run-dir ...`

3. **PGD 컬럼:** 스윕 CSV의 `pgd_adv_acc`, `pgd_success_rate`는 `merge_sweep_pgd.py`로 `pgd_results.json`과 조인해 채운 최종 파일이 `sweep_compression_grid_with_pgd.csv` 이다.
