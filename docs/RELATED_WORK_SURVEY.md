# 관련 논문 조사 (2026-10-05): 대형 학회 22편 + 꼭 비교할 3편

목적: 우리 논문("Reliable Federated TinyML Deployment for IoT Security")과 가까운 연구를 주요 학회
(ACM/IEEE/USENIX/NeurIPS/ICML/ICLR 등)에서 추려, 무엇을 했고 우리와 무엇이 다른지 정리.
확인 방법: 학회 프로그램 페이지·출판사 페이지·DOI가 나온 검색 결과로 존재와 학회를 확인했다
(arXiv/dblp 직접 접속은 막혀 있었음). **[DOI 확인 필요]** 표시는 제출 전 doi.org로 확인할 것.
수치는 초록·공식 요약에서 가져왔고, 못 찾은 것은 적지 않았다.

## 한눈에 보기

| # | 논문 | 학회 | 분야 | 우리와 가장 가까운 점 / 다른 점 |
|---|---|---|---|---|
| 1 | DÏoT (Nguyen et al.) | ICDCS 2019 | 연합학습 IDS | 최초의 FL 기반 IoT 이상탐지. 게이트웨이에서 실행, 압축·MCU 없음 |
| 2 | FLAME (Nguyen et al.) | USENIX Security 2022 | FL 보안 | FL IDS에 대한 백도어/오염 공격 방어. 우리는 학습 단계 공격을 다루지 않음 |
| 3 | FedIoT / FedDetect (Zhang et al.) | SenSys 2021 | FL on IoT 하드웨어 | 라즈베리파이에서 FL 이상탐지. 양자화·MCU 없음 |
| 4 | Belarbi et al. | GLOBECOM 2023 | FL IDS | **같은 TON_IoT**, IP 주소별 클라이언트(실제 non-IID). 압축 없음 |
| 5 | Kitsune (Mirsky et al.) | NDSS 2018 | 경량 NIDS | 비지도 오토인코더, 라즈베리파이. 표준 비교 대상 |
| 6 | Whisper (Fu et al.) | CCS 2021 | 강건한 NIDS | 주파수 영역 특징, 회피 공격에 강함. 위협 모델이 우리보다 강함 |
| 7 | NetBeacon (Zhou et al.) | USENIX Security 2023 | 스위치 내 ML | 트리 모델을 스위치에 올림. 네트워크 안 vs 우리는 단말 |
| 8 | HorusEye (Dong et al.) | USENIX Security 2023 | IoT 스위치 IDS | 2단계(가벼운 1단계 + 무거운 2단계) 구조 |
| 9 | Brain-on-Switch (Yan et al.) | NSDI 2024 | 스위치 내 NN | 이진화 RNN을 스위치에. 극단적 양자화도 탐지 유지 |
| 10 | N3IC (Siracusano et al.) | NSDI 2022 | NIC 내 NN | 이진 NN을 SmartNIC에. 압축 근거 |
| 11 | On-Device Training Under 256KB (Lin et al.) | NeurIPS 2022 | MCU 학습 | MCU에서 직접 학습. 연합 아님 |
| 12 | MCUNet (Lin et al.) | NeurIPS 2020 | MCU 추론 | TinyNAS+TinyEngine, TFLM보다 빠름 |
| 13 | HeteroFL (Diao et al.) | ICLR 2021 | FL + 작은 모델 | 클라이언트별 폭 축소 모델. **BN 통계를 학습 후 서버에서 모음 → 우리 data-free 보정과 가장 가까운 아이디어** |
| 14 | FjORD (Horváth et al.) | NeurIPS 2021 | FL + 압축 | Ordered Dropout + 자기 증류. 학습 중 압축 |
| 15 | FedTiny (Huang et al.) | ICDCS 2023 | FL + 가지치기 | **"모은 데이터 없이 가지치기"** — 우리 주장과 가장 가까움. 양자화·MCU 없음 |
| 16 | ProWD (Yoon et al.) | ICML 2022 | FL + 양자화 | 클라이언트마다 다른 비트폭. 학습 중 양자화 |
| 17 | Hermes (Li et al.) | MobiCom 2021 | FL + 구조적 가지치기 | 클라이언트별 서브네트워크, 휴대폰 |
| 18 | MinUn (Jaiswal et al.) | **LCTES 2023** | MCU 양자화 | 텐서마다 정밀도 다르게. 우리 PTQ 불안정성과 대응 |
| 19 | SeeDot (Gopinath et al.) | PLDI 2019 | MCU 고정소수점 | TFLite PTQ보다 정확. "TFLite PTQ가 상한이 아니다" 근거 |
| 20 | Aster: Sound Mixed Fixed-Point (Lohar et al.) | **EMSOFT 2023** (TECS) | 증명된 양자화 | 오버플로·오차를 증명. 우리 경험적 보정의 원칙적 대안 |
| 21 | Defensive Quantization (Lin et al.) | ICLR 2019 | 양자화 + 견고성 | 양자화가 적대적 오차를 키움 → 우리 FGSM/PGD 결과의 선행 연구 |
| 22 | Deployment faults of DL mobile apps (Chen et al.) | ICSE 2021 | 배포 버그 | TFLite 변환·양자화 결함 분류. 우리 "조용한 실패" 절의 근거 |
| (+) | ONNX converter failures (Jajal et al.) | ISSTA 2024 | 변환기 버그 | 실패의 33%가 **조용히 틀린 모델**을 만듦 |

### 학회는 낮지만 반드시 비교해야 할 것
| 논문 | 출처 | 왜 중요한가 |
|---|---|---|
| **Cognitive IoT and Edge Computing for Intrusion Detection with Federated TinyML** (Li, Laiu, Nichols 외, ORNL) | IEEE AIIoT 2025, pp. 677–684 | **가장 가까운 선행 연구.** FL + TinyML IDS, 3계층(IoT 추론 / 엣지 학습 / 클라우드 FL), N-BaIoT, **이상치에 강한 scaler + 특징 축소 + 양자화**. 우리 "꼬리가 긴 특징 → 클리핑" 발견과 겹침. 인용하고 차이를 분명히 해야 함 |
| TinyFedTL (Kopparapu et al.) | PerCom 2022 Workshop/Demo | MCU에서 FL(마지막 층만 학습)을 처음 구현 |
| Im & Lee, TinyML IDS for in-vehicle CAN | IEEE Embedded Systems Letters 2025 | TFLM + nRF52840의 실제 TinyML IDS. 양자화·FL 없음 |

## 분야별 요약과 우리와의 관계

### A. 연합학습 기반 IoT 침입탐지 (1–4)
- 대부분 **정확도/F1만 보고**하고, 놓친 공격 수, 임계값 정책, 적대적 견고성은 없다.
- 하드웨어는 게이트웨이(DÏoT)나 라즈베리파이(FedIoT)까지만 다룬다. **MCU에서 INT8로 돌린 FL IDS는 없다.**
- Belarbi(TON_IoT)는 IP 주소별로 클라이언트를 나눴다. 리뷰어가 "Dirichlet 분할은 인위적"이라고 지적할 수 있으므로 한계에 적거나 IP별 분할 실험을 추가할 것.
- FLAME과 DIoT 오염 공격: 학습 단계 공격(백도어). 우리는 추론 단계 회피(FGSM/PGD)만 다룬다. 한계·향후 과제로 명시할 것.
- DÏoT의 정확한 페이지(756–767)는 TU Darmstadt 목록 기준. FedIoT는 SenSys **짧은 논문**일 수 있음 → 확인 후 인용.

### B. 경량·네트워크 내 침입탐지 (5–10)
- 보안 학회의 경량 NIDS는 게이트웨이, 라즈베리파이, 스위치·NIC에서 돈다. **MCU 단말에서 도는 NIDS는 대형 학회에 거의 없다.** 이것이 우리 논문의 동기가 된다.
- Kitsune과 HorusEye는 비지도 방식이라 **처음 보는 공격(zero-day)**을 탐지한다. 우리는 지도학습이다 → 한계에 적을 것. 정량 비교 요청이 오면 Kitsune(오픈소스)이 가장 현실적인 비교 대상이다.
- Whisper는 트래픽 수준의 회피 공격을 쓴다. 우리 FGSM/PGD(특징 공간)보다 강한 위협 모델이므로 한계에 적을 것.
- BoS와 N3IC: 이진화까지 해도 탐지가 유지된다 → 우리 INT8 압축의 근거.

### C. 작은 기기에서의 연합학습 + 압축 (11–17)
- 둘로 나뉜다. MCU 쪽 연구(256KB 학습, MCUNet, TinyFedTL)는 연합이 아니거나 마지막 층만 연합한다. FL 압축 쪽 연구(HeteroFL, FjORD, FedTiny, ProWD, Hermes)는 **학습 중에** 모델을 줄이고, 휴대폰·라즈베리파이·시뮬레이션에서 멈춘다.
- 우리만 하는 것:
  - **이미 학습이 끝난** 연합 모델을 클라이언트 데이터만으로 압축한다(가지치기 → 로컬/연합 미세조정 → QAT → INT8 → 증류).
  - **연합 BN 통계로 데이터 없이 INT8 보정**한다. HeteroFL의 sBN과 FedTiny의 BN 선택이 가장 가깝지만, 둘 다 양자화 보정에 쓰지는 않는다.
  - 실제 ESP32 + TFLM, 놓친 공격 수로 신뢰성을 측정한다.
- 비교 후보: FedTiny, 그리고 HeteroFL식 폭 축소 baseline(Flower baselines에 있음).

### D. 임베디드 양자화·견고성·도구 신뢰성 (18–22, +)
- LCTES/EMSOFT 선행 연구: MinUn(LCTES 2023), Aster(EMSOFT 2023), SeeDot(PLDI 2019). LCTES 리뷰어에게 익숙한 맥락이므로 꼭 인용할 것.
- Defensive Quantization: "양자화가 적대적 오차를 증폭한다"는 결과. 우리 표 데이터 MLP에서는 INT8 PTQ가 견고성을 유지했다(CIC PGD 55.6→60.4%). 이것은 그 결과와 대비되는 관찰이다.
- Chen et al. ICSE 2021과 Jajal et al. ISSTA 2024: 변환기가 조용히 틀린 모델을 만든다는 실증 근거. 우리 "네 가지 조용한 실패" 절을 뒷받침한다.

## 논문에 반영할 일 (제안)
1. 관련 연구를 위 네 분야로 재구성하고, 인용을 약 15–20편 추가한다.
2. **ORNL AIIoT 2025 논문과의 차이를 명시**한다. 우리는 CIC/TON, 압축 단계별 분석, 놓친 공격 기준 배포, data-free 보정, 조용한 실패를 다룬다. 저자와 관계가 있다면 이중맹검 규정도 확인할 것.
3. 한계에 추가할 것: 학습 단계 공격(오염/백도어), 처음 보는 공격(비지도 탐지 아님), 트래픽 수준 회피 공격, 실제 non-IID 분할(IP별).
4. 우리 기여 문장: "FL 기반 IDS를 MCU(INT8, TFLM)까지 내리고, 모은 데이터 없이 압축하며, 놓친 공격 수로 평가한 첫 연구"라는 공백을 조사 결과가 뒷받침한다. 단 ORNL 논문 때문에 "처음"이라는 표현은 조심할 것.

## 주의: 우리 제목의 arXiv 프리프린트
- 검색 결과에 **arXiv:2609.27202 "Reliable Federated TinyML Deployment for IoT Security"**가 나온다.
- 요약에 46.7%→93.85% recall, 12.28× 압축이 언급되는데, 이는 예전 WIP 버전 수치로 보인다.
- 이중맹검 심사라면 학회의 프리프린트 정책을 확인해야 한다. 또 그 버전은 우리가 나중에 틀렸다고 확인한 결과를 담고 있으므로, arXiv 버전 갱신을 고려할 것.

## 출처 (확인한 공식 페이지)
- DÏoT: https://ieeexplore.ieee.org/document/8884802
- FLAME: https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen
- DIoT poisoning (DISS@NDSS 2020 workshop): https://www.ndss-symposium.org/wp-content/uploads/2020/04/diss2020-23003-paper.pdf
- FedIoT: https://doi.org/10.1145/3485730.3493444
- Belarbi et al.: https://doi.org/10.1109/GLOBECOM54140.2023.10437860
- Kitsune: https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-3_Mirsky_paper.pdf
- Whisper: https://doi.org/10.1145/3460120.3484585
- NetBeacon: https://www.usenix.org/conference/usenixsecurity23/presentation/zhouguangmeng
- HorusEye: https://www.usenix.org/conference/usenixsecurity23/presentation/dong-yutao
- Brain-on-Switch: https://www.usenix.org/conference/nsdi24/presentation/yan
- N3IC: https://www.usenix.org/conference/nsdi22/presentation/siracusano
- On-Device Training Under 256KB: https://proceedings.neurips.cc/paper_files/paper/2022 (Lin et al.)
- MCUNet: https://papers.nips.cc/paper/2020/hash/86c51678350f656dcc7f490a43946ee5-Abstract.html
- HeteroFL: https://openreview.net/forum?id=TNkPBBYFkXg
- FjORD: https://proceedings.neurips.cc/paper_files/paper/2021/file/6aed000af86a084f9cb0264161e29dd3-Paper.pdf
- FedTiny: ICDCS 2023, pp. 190–201 (arXiv:2212.01977) [DOI 확인 필요]
- ProWD: https://proceedings.mlr.press/v162/yoon22a.html
- Hermes: https://doi.org/10.1145/3447993.3483278
- MinUn: LCTES 2023, pp. 26–39 (pldi23.sigplan.org 프로그램) [DOI 10.1145/3589610.3596278 확인 필요]
- SeeDot: PLDI 2019 [DOI 10.1145/3314221.3314597 확인 필요]
- Aster: https://doi.org/10.1145/3609118
- Defensive Quantization: https://iclr.cc/virtual/2019/poster/863
- Chen et al. ICSE 2021 (arXiv:2101.04930) [DOI 확인 필요]
- Jajal et al. ISSTA 2024: https://doi.org/10.1145/3650212.3680374
- ORNL AIIoT 2025: https://www.ornl.gov/publication/cognitive-iot-and-edge-computing-intrusion-detection-federated-tinyml
- TinyFedTL: arXiv:2110.01107 (PerCom 2022 workshop/demo)
- Im & Lee: https://doi.org/10.1109/LES.2024.3475470
