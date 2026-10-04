# ESP32 실측 가이드 (Windows) — 논문 표 3 / Appendix B / 리뷰 81B-3

보드에 꽂고 명령어 3줄이면 끝나도록 준비돼 있습니다. 30분 정도면 됩니다.

## 무엇을 재나요?
펌웨어 하나에 수정본 표 3의 두 모델이 같이 들어갑니다 (CIC-IDS2017 near-IID 연합학습 모델: FP32 원본과, 50% pruning → 클라이언트 0 미세조정 → QAT → INT8 배포 모델. 12.26× 쌍, run `2026-10-04_e_float_compression_ablation`).

| 이름 | 파일 | 크기 |
|---|---|---|
| `compressed` | `esp32_tflite_project/models/ids_compressed_int8.tflite` (prune 50% → client FT → QAT, INT8) | 67,008 B |
| `baseline` | `esp32_tflite_project/models/ids_baseline_fp32.tflite` (FP32) | 821,792 B |

각 모델마다: ① 고정 입력 8개로 보드 출력 vs PC(TFLite) 출력 비교(parity) → ② warm-up 5회 → ③ 100회 latency 측정.

## 0. 준비물 (한 번만)
1. ESP32 보드 + **데이터 전송 가능한** USB 케이블 (충전 전용 케이블이면 포트가 안 잡힘)
2. Python 3.10+ (이미 `.venv` 있으면 그거 사용)
3. 터미널(PowerShell)에서:
   ```powershell
   cd C:\Users\danie\Documents\Projects\TinyML
   git fetch origin; git checkout claude/eloquent-hypatia-kjfsre; git pull
   pip install platformio pyserial
   ```
4. 보드를 꽂았는데 포트가 안 보이면 USB-시리얼 드라이버 설치:
   보드의 USB 칩 글자를 보고 **CP210x**(Silicon Labs) 또는 **CH340**(WCH) 드라이버.

## 1. 포트 확인
```powershell
python scripts/collect_esp32_benchmark.py --list-ports
```
`COM3  Silicon Labs CP210x ...` 같은 줄이 보이면 그 COM 번호를 씁니다.

## 2. 빌드 + 업로드 (보드 종류에 맞게 하나)
```powershell
cd esp32_tflite_project
pio run -e esp32dev -t upload --upload-port COM3            # 일반 ESP32 (DevKit, WROOM, WROVER)
# pio run -e esp32-s3-devkitc-1 -t upload --upload-port COM3
# pio run -e esp32-c3-devkitm-1 -t upload --upload-port COM3
cd ..
```
- 첫 빌드는 툴체인/라이브러리 다운로드 때문에 5–10분 걸립니다.
- 보드 칩 위 금속 캔에 `ESP32-S3` / `ESP32-C3` 라고 써 있으면 그 env, 아무 표시 없거나 `ESP32-WROOM-32`면 `esp32dev`.
- `Connecting........___` 에서 멈추면: 보드의 **BOOT 버튼을 누른 채로** 다시 실행, "Writing at..." 나오면 손 떼기.

## 3. 결과 수집
```powershell
python scripts/collect_esp32_benchmark.py --port COM3
```
포트를 열면 보드가 리셋되면서 처음부터 측정합니다 (1분 정도). 끝나면 이렇게 요약이 나옵니다:
```
✅ ESP32 benchmark saved: ...\data\processed\ablation\esp32_benchmark.json (raw log: esp32_benchmark.log)
   Device: ESP32-D0WD-V3 @ 240 MHz, SDK v4.4.x
   compressed  67008 B  mean x.xxx ms  median ...  parity max|Δ|=0.0039 labels 8/8
   baseline    821792 B mean x.xxx ms  median ...  parity max|Δ|=0.0 labels 8/8
```
아무것도 안 나오고 타임아웃이면 보드의 **EN(RST) 버튼**을 한 번 누르고 다시 실행.

## 4. 커밋해서 넘겨주기
```powershell
git add data/processed/ablation/esp32_benchmark.json data/processed/ablation/esp32_benchmark.log
git commit -m "ESP32 on-device benchmark results"
git push
```
push하면 Claude가 JSON을 읽어서 논문 표 3 / 6.1절 / Appendix B 를 실측값으로 채웁니다.
보드 이름(예: "ESP32-DevKitC V4")도 같이 알려주세요.

## 문제 해결
| 증상 | 해결 |
|---|---|
| 빌드 중 `TensorFlowLite_ESP32` 를 못 찾음 | `platformio.ini` 의 `lib_deps` 를 `https://github.com/tanakamasayuki/Arduino_TensorFlowLite_ESP32.git` 로 바꾸기 (git 필요) |
| `section .flash.rodata ... will not fit` / 앱이 너무 큼 | 보드 flash가 4MB 미만인 경우. `platformio.ini`에서 `board_build.partitions` 확인, 에러 전체를 Claude에게 붙여넣기 |
| 시리얼에 `ERROR ... AllocateTensors failed` | `src/main.cpp` 의 `kTensorArenaSize` 를 `32 * 1024` 로 올리기 |
| `parity max|Δ|` 가 0.01보다 큼 / labels 8/8 아님 | 결과 그대로 커밋해서 알려주기 (보드 커널 차이 분석 필요) |
| 그 외 | 에러 메시지 전체를 Claude에게 붙여넣기 |

## 참고: 검증된 것 / 안 된 것
- ✅ 이 펌웨어는 같은 TFLM 라이브러리(TensorFlowLite_ESP32 1.0.0)로 PC에서 빌드·실행 검증됨: 두 모델 모두 로드/실행, parity INT8 max|Δ|=1/256, FP32 완전 일치, arena 2–4 KB.
- ⚠️ ESP32 툴체인 빌드는 클라우드 환경에서 PlatformIO 서버 접근이 막혀 확인 못 함 → 첫 빌드에서 에러 나면 위 표 참고.
- 다른 모델을 재고 싶으면: `python scripts/prepare_esp32_benchmark.py --compressed <tflite> --baseline <tflite>` (TensorFlow 필요).
