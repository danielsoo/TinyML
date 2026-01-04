# Project Tasks Summary - Phase 4 & 5

이 문서는 Phase 4와 Phase 5의 태스크들을 백로그에 추가하기 위한 요약입니다.

## Phase 4: Adversarial Hardening & Deployment (Weeks 12-14)

### 태스크 목록

1. **[Phase 4] Complete FGSM Attack Implementation** - Week 12
   - 기본 FGSM 유틸리티를 완전한 공격 구현으로 확장
   - 관련 파일: `src/adversarial/fgsm_hook.py`, `src/models/nets.py`

2. **[Phase 4] FGSM Integration into FL Training Loop** - Week 12-13
   - FL 클라이언트 학습 루프에 adversarial example 생성 통합
   - 관련 파일: `src/federated/client.py`, `src/adversarial/fgsm_hook.py`

3. **[Phase 4] Adversarial Training in FL** - Week 13
   - Adversarial training을 포함한 전체 FL 프로세스 실행
   - 관련 파일: `src/federated/client.py`, `src/federated/server.py`

4. **[Phase 4] Re-compression of Robust Model** - Week 13-14
   - 강화된 모델에 대해 압축 파이프라인 재실행
   - 관련 파일: `src/tinyml/export_tflite.py`, `scripts/analyze_compression.py`

5. **[Phase 4] Microcontroller Deployment** - Week 14
   - 실제 하드웨어 배포 (ESP32/Raspberry Pi Pico) 및 성능 측정
   - 관련 파일: `data/processed/` (TFLite models)

## Phase 5: Final Evaluation & Reporting (Week 15)

### 태스크 목록

1. **[Phase 5] Comprehensive Experiments** - Week 15
   - 모든 모델 변형에 대한 종합 실험 수행
   - 관련 파일: `scripts/analyze_compression.py`, `scripts/visualize_results.py`

2. **[Phase 5] Performance Metrics Collection** - Week 15
   - 성능 지표 수집 및 분석 (이미 완료된 인프라 활용)
   - 관련 파일: `scripts/analyze_compression.py`

3. **[Phase 5] Efficiency Metrics Collection** - Week 15
   - 효율성 지표 수집 및 분석 (이미 완료된 인프라 활용)
   - 관련 파일: `scripts/analyze_compression.py`

4. **[Phase 5] Analysis Reports & Visualizations** - Week 15
   - 종합 분석 보고서 및 시각화 생성 (이미 완료된 인프라 활용)
   - 관련 파일: `scripts/analyze_compression.py`, `scripts/visualize_results.py`

5. **[Phase 5] Final Project Report** - Week 15
   - 최종 프로젝트 보고서 작성
   - 관련 파일: 프로젝트 문서, 분석 결과

6. **[Phase 5] Final Presentation & Demonstration** - Week 15
   - 최종 발표 및 데모 준비
   - 관련 파일: `PSU_Capstone.pptx`, 데모 스크립트

---

## 타임라인 요약

| Phase | 기간 | 완료율 |
|-------|------|--------|
| Phase 1 | Weeks 1-3 | ~95% |
| Phase 2 | Weeks 4-7 | 100% |
| Phase 3 | Weeks 8-11 | ~50% |
| **Phase 4** | **Weeks 12-14** | **~5%** |
| **Phase 5** | **Week 15** | **~40%** |

**전체 프로젝트 완료율: ~58%**

---

## 상세 태스크 정보

각 태스크의 상세 정보는 다음 파일들을 참조하세요:
- Phase 4 상세 태스크: `docs/PHASE4_TASKS.md`
- Phase 5 상세 태스크: `docs/PHASE5_TASKS.md`

