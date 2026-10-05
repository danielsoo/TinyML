# ESP32 구매 가이드 (논문 6.1 / 부록 B 실측용)

펌웨어(`esp32_tflite_project`)는 이미 준비돼 있고, PC에서 같은 TFLite Micro 라이브러리로 빌드해 출력이 맞는 것까지 확인했습니다.
보드만 사서 꽂으면 됩니다. 사용법은 `docs/ESP32_BENCHMARK_GUIDE.md`를 보세요.

## 1. 무엇을 살까

| 우선순위 | 보드 | 칩 / CPU | 펌웨어 env | 왜 |
|---|---|---|---|---|
| **필수** | **ESP32-DevKitC V4** (모듈 ESP32-WROOM-32E, **플래시 4MB**)<br>또는 흔한 호환보드 "ESP32 DevKit V1 (WROOM-32, 30핀/38핀)" | ESP32, Xtensa LX6 듀얼코어 240 MHz, SRAM 520 KB | `esp32dev` | 논문의 기준 보드. 가장 흔하고 쌈 |
| **강력 추천** | **ESP32-S3-DevKitC-1** (N8R8 또는 N16R8 등, 플래시 8MB 이상이면 무엇이든) | ESP32-S3, Xtensa LX7 듀얼코어 240 MHz (벡터 명령) | `esp32-s3-devkitc-1` | 리뷰어 81C "기기 이질성(device heterogeneity)" 지적에 답할 수 있음: CPU가 다른 두 번째 보드 |
| 선택 | **ESP32-C3-DevKitM-1** 또는 "ESP32-C3 SuperMini" (**플래시 4MB** 확인) | ESP32-C3, RISC-V 싱글코어 160 MHz | `esp32-c3-devkitm-1` | 세 번째 아키텍처(RISC-V). 가장 느린 보드에서의 latency |

- 최소 1개(ESP32)면 6.1/부록 B를 채울 수 있습니다. S3까지 있으면 "서로 다른 MCU 두 종에서 측정"이라고 쓸 수 있어 리뷰 대응이 훨씬 좋아집니다.
- 펌웨어에 모델 4개(약 1.7 MB)가 들어가서 **플래시 4MB 이상**이 반드시 필요합니다. "2MB" 표기된 보드는 피하세요.

### 사면 안 되는 것
- **ESP8266 / NodeMCU (ESP8266)**: 다른 칩입니다. 지원 안 됨.
- **ESP32-CAM**: USB 포트가 없어 별도 어댑터가 필요합니다.
- **ESP32-S2 / ESP32-C6 / ESP32-H2**: 동작할 수도 있지만 펌웨어에 설정이 없습니다(추가는 가능하나 검증 안 됨).
- 플래시 2MB 보드.

## 2. 같이 필요한 것
- **데이터 전송 가능한 USB 케이블** (충전 전용 케이블이면 포트가 안 잡힙니다)
  - ESP32 DevKitC V4 / DevKit V1: 대부분 **Micro-USB** (요즘 일부는 USB-C — 상품 사진에서 확인)
  - ESP32-S3-DevKitC-1, ESP32-C3 SuperMini: **USB-C**
- 브레드보드, 점퍼선, 센서 등은 **필요 없습니다** (USB 연결만으로 측정).

## 3. 어디서 사나
- 국내(빠름, 1~3일): 디바이스마트, 엘레파츠, 아이씨뱅큐, 쿠팡 등에서 위 보드 이름으로 검색.
- 해외(저렴, 1~3주): 알리익스프레스 등. 호환보드도 측정에는 문제없습니다.
- 가격은 판매처마다 다르지만 대체로 ESP32/C3 보드는 1만 원 안팎, S3 보드는 그보다 조금 비쌉니다.

## 4. 받으면 할 일 (요약)
1. 보드를 USB로 PC에 연결. 장치 관리자에 COM 포트가 안 보이면 보드의 USB 칩에 맞는 드라이버 설치:
   **CP210x** (Silicon Labs) 또는 **CH340/CH343** (WCH). S3/C3의 내장 USB는 보통 드라이버 불필요.
2. PowerShell에서:
   ```powershell
   cd C:\Users\danie\Documents\Projects\TinyML
   git pull
   pip install platformio pyserial
   python scripts/collect_esp32_benchmark.py --list-ports
   cd esp32_tflite_project
   pio run -e esp32dev -t upload --upload-port COM3      # S3면 -e esp32-s3-devkitc-1, C3면 -e esp32-c3-devkitm-1
   cd ..
   python scripts/collect_esp32_benchmark.py --port COM3
   ```
3. 보드가 여러 개면 보드마다 결과 파일 이름을 바꿔서 저장:
   `python scripts/collect_esp32_benchmark.py --port COM4 --output data/processed/ablation/esp32s3_benchmark.json`
4. 결과 파일(`data/processed/ablation/*.json`, `*.log`)을 커밋/푸시하면 논문에 넣습니다.

자세한 문제 해결(업로드가 `Connecting...`에서 멈출 때 BOOT 버튼 등)은 `docs/ESP32_BENCHMARK_GUIDE.md` 참고.

## 참고: 미리 확인한 것 / 못 한 것
- 확인함: 같은 TensorFlowLite_ESP32 1.0.0 라이브러리를 PC용으로 빌드해 펌웨어를 실행 → 4개 모델 모두 PC TFLite 출력과 일치(FP32 정확히, INT8 최대 1/256), 판정 8/8 일치, 메모리(arena) 2.2–4.6 KB.
- 크기: 모델 4개 합계 약 1.74 MB + Arduino/TFLM 코드 → `huge_app` 파티션(앱 3 MB)에 들어갑니다. 그래서 플래시 4MB 이상 보드가 필요합니다.
- 못 함: 실제 ESP32용 빌드(PlatformIO 툴체인 다운로드가 이 클라우드 환경의 네트워크 정책으로 막혀 있음). 처음 PC에서 `pio run`을 하면 툴체인을 받느라 5–10분 걸리고, 그때 실제 빌드가 됩니다. 빌드 에러가 나면 로그를 그대로 보내주세요.
