# 양자화/압축 기법 문헌 조사 (2026-10-05)

목적: QAT·PTQ 순서와 조합, 새 기법들을 정리하고 우리 결과(jobs l–w)와 어떻게 연결되는지 기록.
각 항목의 출처는 학회 페이지나 arXiv 기록으로 확인했다. 아래 "우리와의 관계"는 우리 실험을 근거로 한 판단이다.

## 1. PTQ를 더 잘하기 (학습 없이 변환만)

| 기법 | 핵심 | 출처 | 우리와의 관계 |
|---|---|---|---|
| Calibration: max / percentile / entropy(KL) / MSE | 활성값 범위를 최댓값 대신 분위수·KL·MSE 기준으로 자름 | Wu et al. 2020, arXiv:2004.09602 | job l의 PTQ 불안정(F1 29–85)은 max 범위가 꼬리값에 끌려가서 생김. 우리 ±5 SD 클리핑(job m)은 이 계열의 단순한 형태라서 **새 방법이라고 주장하면 안 되고 인용해야 함** |
| ACIQ | 가우시안/라플라스 가정으로 MSE 최소 클리핑 값을 해석적으로 계산 | Banner, Nahshan, Soudry, NeurIPS 2019, arXiv:1810.05723 | ±5 SD 대신 이론값으로 자르는 대안 |
| OCS (outlier channel splitting) | 이상치 채널을 복제·반감해 범위를 줄임 | Zhao et al., ICML 2019, PMLR 97:7543–7552 | 클리핑 없이 꼬리를 다루는 대안. TFLite에선 직접 지원 안 함 |
| DFQ (data-free) | 가중치 equalization + bias correction, 데이터 없이 PTQ | Nagel et al., ICCV 2019, arXiv:1906.04721 | **FL에 잘 맞음**: 서버는 데이터를 못 보니 데이터 없이 보정하는 방법은 프라이버시 측면 장점 |
| AdaRound | 반올림 방향을 소량 데이터로 학습 | Nagel et al., ICML 2020, arXiv:2004.10568 | 주로 4비트 이하에서 효과. INT8 소형 MLP에선 이득이 작을 가능성 |
| BRECQ | 블록 단위 재구성으로 PTQ | Li et al., ICLR 2021, arXiv:2102.05426 | 위와 같음 (저비트용) |
| QDrop | PTQ 재구성 중 양자화를 무작위로 끔 | Wei et al., ICLR 2022, arXiv:2203.05740 | 위와 같음 (저비트용) |

## 2. QAT를 더 잘하기

| 기법 | 핵심 | 출처 | 우리와의 관계 |
|---|---|---|---|
| Integer-only QAT (fake quant) | TFLite/tfmot 표준 QAT | Jacob et al., CVPR 2018 (이미 인용) | 우리 기본 레시피 |
| PACT | 활성 클리핑 값 α를 학습 | Choi et al. 2018, arXiv:1805.06085 | 고정 ±5 대신 클리핑을 학습하는 대안 |
| LSQ | 양자화 step size를 학습 | Esser et al., ICLR 2020, arXiv:1902.08153 | 위와 같음 |
| Oscillation dampening / freezing | QAT 중 가중치가 반올림 경계에서 진동하는 문제 해결 | Nagel et al., ICML 2022 (이미 인용) | QAT 결과 편차 설명에 사용 중 |
| HAWQ (mixed precision) | Hessian으로 층별 비트 수를 정함 | Dong et al., ICCV 2019, arXiv:1905.03696 | 우리는 int16x8을 모델 전체에만 적용. 첫/마지막 층만 고정밀로 두는 실험이 가능 |

## 3. 순서와 조합

| 조합 | 핵심 | 출처 | 우리와의 관계 |
|---|---|---|---|
| PTQ 보정 → QAT 미세조정 | 먼저 PTQ로 범위를 잡고, 정확도가 부족할 때 QAT로 짧게 미세조정 | Wu et al. 2020; Nagel white paper 2021 (이미 인용) | 우리 QAT는 tfmot 이동평균으로 범위를 새로 잡음. "PTQ 범위로 초기화한 QAT"는 안 해봄 |
| Pruning → 양자화 | 가지치기 후 양자화(+Huffman) | Han et al., ICLR 2016 (이미 인용) | 우리 기본 순서와 같음 |
| Pruning 보존 QAT (PQAT/CQAT) | 일반 QAT가 0으로 만든 가중치를 되살리는 문제를 막음 | TF Model Optimization "Collaborative Optimization" 문서 | job n에서 "일반 QAT는 희소성을 되살리고 PQAT는 유지" 관찰 = 이 API가 존재하는 이유. 인용 필요 |
| Pruning + 양자화 동시 (학습 중) | 두 압축을 한 번에 학습 | DJPQ, Wang, Lu, Blankevoort, ECCV 2020, arXiv:2007.10463 | 관련 연구 |
| Pruning + 양자화 동시 (학습 후) | OBS 기반 post-training 압축 | OBC, Frantar & Alistarh, NeurIPS 2022, arXiv:2208.11580 | 관련 연구 |
| 지식 증류 + 양자화 | FP32 teacher로 양자화 student 학습 | Polino, Pascanu, Alistarh, ICLR 2018, arXiv:1802.05668 | 우리 KD+QAT(5.10, recall 레시피)와 같은 계열. 인용 필요 |

## 4. 연합학습 + 양자화

| 기법 | 핵심 | 출처 | 우리와의 관계 |
|---|---|---|---|
| FedPAQ | 통신량을 줄이려고 업데이트를 양자화 | Reisizadeh et al., AISTATS 2020, PMLR 108:2021–2031 | 목적이 다름(통신). 우리는 추론용 모델 양자화 |
| Quantization-robust FL | 여러 비트폭에서 양자화에 강한 모델을 FL로 학습 | arXiv:2206.10844 | 우리 "federated FT/QAT"와 가까움. 인용하고 차이를 밝혀야 함 |
| FedAQT | 기기에서 양자화 변수로 직접 학습 | Google Research, "FedAQT: Accurate Quantized Training with Federated Learning" | 위와 같음 |

## 5. 논문에 반영할 점
1. ±5 SD 클리핑 보정은 percentile/ACIQ 계열이라고 쓰고 인용한다. 우리 기여는 "IDS 표 데이터에서 max 보정이 draw마다 크게 흔들린다는 관찰 + 단순 클리핑으로 해결" 정도로 표현한다.
2. KD+QAT는 Polino et al., PQAT 관찰은 TF collaborative optimization, federated FT/QAT는 quantization-robust FL / FedAQT를 인용한다.
3. 순서 비교(PTQ→QAT)와 데이터 없는 보정(DFQ)은 future work 또는 추가 실험 후보.

## 6. 추가 실험 후보 (PC 실행, 사용자 승인 필요)
- A. 보정 방법 비교: max / percentile(99.9, 99.99) / MSE(ACIQ식) / ±5 SD, job l·m과 같은 draw. 비용 작음.
- B. PTQ 범위로 초기화한 QAT vs 현재 QAT. 비용 중간.
- C. 데이터 없는 보정: 접힌 BN 통계로 합성 입력을 만들어 서버에서 보정 (FL 프라이버시와 연결). 비용 중간.
- D. 학습되는 클리핑(PACT/LSQ식) QAT. 비용 중간~큼.
- E. AdaRound식 반올림. INT8에선 이득이 작을 가능성이 커 우선순위 낮음.

## 7. 실행 (2026-10-05): jobs x1–x4, `scripts/quant_lit_methods.py`
| job | 실험 | 모델 |
|---|---|---|
| `2026-10-05_x1_calib_methods` | A. 보정 방법: tflite max / ±5 clip / per-tensor max, fq_{max, pct99.9, pct99.99, MSE, KL, clip5_max} × 15 calibration sets | CIC near-IID (job b), TON v2 (job s) |
| `2026-10-05_x2_ptq_init_qat` | B. QAT 시작 범위: tfmot 기본 vs PTQ 보정값에서 시작(init) vs 보정값 고정(fixed), 5 draws | 같음 |
| `2026-10-05_x3_data_free_calib` | C. 데이터 없는 보정: N(0,1) 합성 입력, BN 통계(k=3,4,6), CLE 유무, 실데이터 기준 | 같음 |
| `2026-10-05_x4_learned_clip_qat` | D. 학습되는 클리핑(PACT식) QAT, 3가지 시작값, 5 draws | 같음 |

로컬 확인(가짜 데이터, 실제 모델 구조)에서 드러난 사실 — 결과 해석에 필요:
- `fq_` 경로(범위를 직접 지정한 fake-quant 모델)는 per-tensor TFLite PTQ와 결과가 정확히 같다(F1·FAR·놓친 수 동일). 즉 추정 방법만 바꿔 비교할 수 있다.
- TFLite PTQ의 Dense 가중치는 **채널별(per-channel) scale**, tfmot QAT export는 **텐서당 하나(per-tensor)**. 그래서 예전 "PTQ vs QAT" 비교에는 가중치 양자화 방식 차이도 섞여 있다. `tflite_max_per_tensor` 행이 이것을 분리한다.
- tfmot QAT의 활성 범위는 ±6에서 시작해 EMA(0.999)로 움직인다. 우리 QAT(약 140 step)는 끝나도 시작값의 약 87%가 남는다(예: ReLU 출력 최소값이 0이 아니라 −5.2). 실험 B가 이 점을 겨냥한다.
- tfmot 입력 QuantizeLayer는 학습 중 본 전체 입력의 min/max를 따라간다(AllValuesQuantizer). 그래서 "init" 방식에서는 입력 클리핑 효과가 사라지고, "fixed"만 유지된다.
