# 스윕 요약 (Sweep Summary)

이 run은 **압축 그리드 스윕(fl_qat × distillation × pruning × ptq) 48조합** 실행 후, 모든 모델에 대해 **PGD**를 수행하고 결과를 병합한 결과입니다.

## CSV 컬럼 설명

| 컬럼 | 설명 |
|------|------|
| tag | 조합 식별자 (fl_qat__distill_*__pruning__ptq_*) |
| tflite_path | TFLite 모델 파일 경로 |
| fl_qat | FL 후 QAT 적용 여부 (True/False) |
| distillation | 지식 증류 방식: none / direct / progressive |
| pruning | 프루닝: prune_none / prune_10x5 / prune_5x5 / prune_3x3 |
| ptq | PTQ 양자화 여부 (True/False) |
| fl_acc | FL 단계 정확도 |
| fl_f1 | FL 단계 F1 |
| prune_acc | 프루닝 직후 정확도 |
| prune_f1 | 프루닝 직후 F1 |
| final_acc | 최종 TFLite 정확도 |
| final_f1 | 최종 TFLite F1 (클수록 좋음) |
| final_size_kb | 최종 TFLite 크기 (KB, 작을수록 좋음) |
| pgd_adv_acc | PGD 공격 후 방어 정확도 |
| pgd_success_rate | PGD 공격 성공률 |

## final_f1 상위 5개

| 순위 | tag | final_f1 | final_size_kb | pgd_adv_acc |
|------|-----|----------|---------------|-------------|
| 1 | yes_qat__distill_progressive__prune_none__ptq_yes | 0.8341232227488151 | 1.8281 | 0.89875 |
| 2 | yes_qat__distill_progressive__prune_10x2__ptq_yes | 0.8262910798122066 | 1.8281 | 0.89875 |
| 3 | yes_qat__distill_progressive__prune_10x5__ptq_yes | 0.8243559718969555 | 1.8281 | 0.89875 |
| 4 | yes_qat__distill_progressive__prune_5x10__ptq_yes | 0.8243559718969555 | 1.8281 | 0.89875 |
| 5 | yes_qat__distill_direct__prune_5x10__ptq_yes | 0.821515892420538 | 1.8281 | 0.89895 |

## final_size_kb 하위 5개 (가장 작은 모델)

| 순위 | tag | final_size_kb | final_f1 | pgd_adv_acc |
|------|-----|---------------|----------|-------------|
| 1 | yes_qat__distill_direct__prune_none__ptq_no | 1.5508 | 0.8063063063063064 | 0.8988 |
| 2 | yes_qat__distill_direct__prune_10x5__ptq_no | 1.5508 | 0.7894736842105263 | 0.89895 |
| 3 | yes_qat__distill_direct__prune_10x2__ptq_no | 1.5508 | 0.7921225382932167 | 0.89895 |
| 4 | yes_qat__distill_direct__prune_5x10__ptq_no | 1.5508 | 0.7929515418502202 | 0.89895 |
| 5 | yes_qat__distill_progressive__prune_none__ptq_no | 1.5508 | 0.8035320088300221 | 0.89925 |

## 추천 조합 (final_f1 >= 0.9 & final_size_kb <= 200.0)

조건을 만족하는 행이 없습니다.
