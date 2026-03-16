# v1_PGD 두 Run 비교 (보고서용 추천)

| 구분 | Run 1 (11-37-42) | Run 2 (16-52-20) |
|------|------------------|------------------|
| **실험 설정** | 동일 (experiment_record 동일) | 동일 |
| **PGD 결과** | ❌ 없음 (pgd/ 폴더 없음) | ✅ 있음 (5개 모델 비교, epsilon sweep) |
| **Compression 분석** | ✅ 5 stages | ✅ 5 stages |
| **Ratio sweep** | ✅ 8 models × 11 ratios | ✅ 8 models × 11 ratios |

---

## 1. Compression 분석 (5 stages) 비교

| Stage | Run 1 Acc | Run 1 F1 | Run 2 Acc | Run 2 F1 | 비고 |
|-------|------------|----------|------------|----------|------|
| Baseline | 0.1707 | 0.2912 | 0.1760 | 0.2904 | 비슷 |
| Traditional+PTQ | 0.2157 | 0.3029 | 0.2121 | 0.3019 | 비슷 |
| **Traditional+QAT** | **0.9682** | **0.9106** | 0.9683 | 0.9092 | Run1이 F1 소폭 상승 |
| **QAT+PTQ** | **0.9490** | **0.8649** | 0.9477 | 0.8624 | Run1이 소폭 상승 |
| **QAT+QAT** | **0.9639** | **0.8968** | 0.9532 | 0.8743 | Run1이 F1 약 2.3%p 높음 |

→ Run 1이 압축 단계 F1 수치는 소폭 더 좋음 (차이 ~2%p 수준).

---

## 2. Run 2에만 있는 것: PGD 평가

Run 2에는 **동일 adversarial example**로 5개 모델을 평가한 결과가 있음.

| Model | Original Acc | Adv Acc | Attack Success Rate |
|-------|--------------|---------|---------------------|
| Keras (global_model.h5) | 0.9274 | 0.9001 | 2.7% |
| Traditional+QAT | 0.9236 | **0.1816** | **74.2%** (매우 취약) |
| Compressed (QAT) | 0.9226 | 0.8814 | 4.1% |
| QAT+PTQ | 0.9446 | 0.9059 | 3.9% |
| noQAT+PTQ | 0.2206 | 0.18 | 4.1% |

→ 논문/보고서에서 “압축 후 PGD robustness”를 넣으려면 **Run 2 데이터가 필수**.

---

## 3. Ratio sweep (대표: 50% normal) F1 비교

| Model | Run 1 F1 @50% | Run 2 F1 @50% |
|-------|----------------|----------------|
| Traditional+QAT | 0.9610 | 0.9529 |
| QAT+PTQ | 0.9532 | 0.9536 |
| Compressed (QAT) | 0.9449 | 0.9541 |
| pruned_10x2 | 0.9593 | 0.9555 |
| pruned_5x10 | 0.9569 | 0.9562 |

→ 전반적으로 비슷; Run 1이 Traditional+QAT에서만 약간 높음.

---

## 4. 보고서에 넣기 좋은 run 추천

**추천: Run 2 (2026-03-14_16-52-20)**

이유:
1. **PGD 결과가 Run 2에만 있음** — 보고서에서 “압축 + adversarial robustness”를 함께 쓰려면 Run 2의 `pgd/pgd_report.md`, `pgd_results.json`이 필요함.
2. **Compression 수치** — Run 1이 F1 소폭 우세하지만, 차이가 2%p 내외라 Run 2만 써도 정확도/압축률/지연 시간 스토리는 동일하게 전달 가능함.
3. **일관성** — 압축 테이블, ratio sweep, **PGD 테이블**을 한 run에서 통일하면 “같은 모델들에 대한 결과”로 서술하기 좋음.

**선택 사항:**
- “Compression만 강조하고 PGD는 빼는” 보고서라면 Run 1 수치(특히 QAT+QAT F1 0.8968)를 인용해도 됨.
- “PGD까지 포함한 전체 실험”을 쓰는 경우에는 **반드시 Run 2**를 기준으로 하고, 필요하면 문장에서 “Compression 성능은 두 run 모두 유사하며, 아래 수치는 Run 2 (16-52-20) 기준이다”라고 한 줄만 명시하면 됨.

---

## 5. 참고 경로

- **Run 1:** `v1_PGD/2026-03-14_11-37-42/` — compression_analysis.md, ratio_sweep_report.md
- **Run 2:** `v1_PGD/2026-03-14_16-52-20/` — compression_analysis.md, ratio_sweep_report.md, **pgd/pgd_report.md**, **pgd/pgd_results.json**
