# 마이크로컨트롤러 / 엣지 배포 가이드

## 개요

이 가이드는 TFLite 모델을 **라즈베리 파이** 또는 **ESP32**에서 돌리는 방법을 설명합니다.

- **라즈베리 파이**: Python + TFLite 런타임으로 바로 추론 (권장: 먼저 여기서 검증)
- **ESP32**: TFLite Micro, C 배열로 변환 후 플래시

## 1. 로컬에서 엣지 동작 확인 (IDS 모델)

학습·압축으로 만든 TFLite 모델(예: `saved_model_qat_ptq.tflite`)이 제대로 로드·추론되는지 먼저 확인합니다.

```bash
# 프로젝트 루트에서
python scripts/run_on_edge_test.py --model models/tflite/saved_model_qat_ptq.tflite
```

- 입력 차원(예: 78)을 자동으로 읽어서 더미 입력으로 추론합니다.
- 평균/최소/최대 지연 시간(ms)을 출력합니다.
- **라즈베리 파이 / ESP32**에서 할 작업 요약도 함께 출력됩니다.

다른 모델을 쓰려면 `--model` 경로만 바꾸면 됩니다.

---

## 2. 라즈베리 파이에서 확인

1. **TFLite 파일 복사**
   ```bash
   scp models/tflite/saved_model_qat_ptq.tflite pi@<라즈베리파이IP>:~/TinyML/
   ```

2. **라즈베리 파이에서 런타임 설치**
   ```bash
   pip3 install tflite-runtime
   # 또는 (무거움): pip3 install tensorflow
   ```

3. **추론 스크립트 예시** (RPi에서 실행)
   ```python
   import numpy as np
   import tflite_runtime.interpreter as tflite

   interp = tflite.Interpreter(model_path="saved_model_qat_ptq.tflite")
   interp.allocate_tensors()
   inp = interp.get_input_details()[0]
   out = interp.get_output_details()[0]
   # 입력: (1, 78) float32 (CIC-IDS2017 특징 벡터)
   x = np.random.randn(1, 78).astype(np.float32)
   interp.set_tensor(inp["index"], x)
   interp.invoke()
   pred = interp.get_tensor(out["index"])
   print("Output:", pred)
   ```

4. **지연 시간 측정**: 위 루프를 여러 번 돌려서 `time.time()`으로 평균 ms를 재면 됩니다.

---

## 3. ESP32 배포 (사전 준비)

### 3.1 테스트 모델 생성

```bash
# 가상 환경 활성화
source .venv/bin/activate

# 테스트 모델 생성
python scripts/create_test_model.py
```

이 스크립트는 다음을 생성합니다:
- `data/processed/microcontroller/hello_world_model.tflite` - 간단한 Hello World 모델

### 3.2 TFLite 모델을 C 배열로 변환

**실제 IDS 모델(78차원)을 ESP32용으로 변환:**
```bash
python scripts/deploy_microcontroller.py --model models/tflite/saved_model_qat_ptq.tflite --output data/processed/microcontroller/ids_model_data.c
```

**테스트용 Hello World 모델:**
```bash
python scripts/deploy_microcontroller.py
```

생성되는 파일:
- `data/processed/microcontroller/model_data.c` - C 소스 파일
- `data/processed/microcontroller/model_data.h` - C 헤더 파일

### 3.3 로컬에서 추론 테스트 (하드웨어 없이 검증)

실제 하드웨어가 없어도 배포 파이프라인이 제대로 작동하는지 검증할 수 있습니다:

```bash
# TFLite 모델 추론 테스트
python scripts/test_tflite_inference.py

# C 배열 파일도 함께 검증
python scripts/test_tflite_inference.py --verify-c-files
```

이 스크립트는:
- ✅ 모델이 제대로 로드되는지 확인
- ✅ 여러 테스트 케이스로 추론 실행
- ✅ 추론 시간 측정
- ✅ C 배열 파일 형식 검증
- ✅ 배포 준비 상태 확인

**예상 출력:**
```
============================================================
TFLite Model Inference Test
============================================================

Model Information:
  Input shape: [1, 2]
  Input type: <class 'numpy.float32'>
  Output shape: [1, 1]
  Output type: <class 'numpy.float32'>

Running Inference Tests
============================================================

Test 1: Low values
  Input: [0.3, 0.3]
  Output: [0.4729]
  Inference time: 0.123 ms

...

✅ All inference tests passed!
✅ Model is ready for ESP32 deployment
```

## ESP32 배포 (하드웨어가 있는 경우)

### 개발 환경 설정

#### PlatformIO 설치

```bash
# 가상 환경 안에서
pip install platformio

# 또는 전역 설치
pip install platformio
```

#### ESP32 프로젝트 구조

프로젝트는 이미 `esp32_tflite_project/` 디렉토리에 생성되어 있습니다:

```
esp32_tflite_project/
├── platformio.ini          # PlatformIO 설정
├── src/
│   ├── main.cpp            # ESP32 메인 코드
│   ├── model_data.c        # 모델 C 배열
│   └── model_data.h        # 모델 헤더
└── lib/                    # 라이브러리 디렉토리
```

### 프로젝트 설정

**platformio.ini** 파일이 이미 설정되어 있습니다:
- 보드: ESP32 Dev Module
- 프레임워크: Arduino
- 라이브러리: TensorFlow Lite Micro

### 빌드 및 업로드

```bash
cd esp32_tflite_project

# 라이브러리 설치 (처음 한 번만)
pio lib install

# 빌드
pio run

# 업로드 (ESP32를 USB로 연결)
pio run --target upload

# 또는 포트 지정
pio run --target upload --upload-port /dev/ttyUSB0  # Linux/Mac
pio run --target upload --upload-port COM3          # Windows
```

### 시리얼 모니터

```bash
# PlatformIO 시리얼 모니터
pio device monitor

# 또는 다른 시리얼 모니터 사용
screen /dev/ttyUSB0 115200  # Linux/Mac
```

## 예상 출력

시리얼 모니터에서 다음과 같은 출력을 확인할 수 있습니다:

```
========================================
ESP32 TensorFlow Lite Micro Test
========================================

Loading TFLite model...
Model loaded successfully!
Model size: 2168 bytes

Model initialized successfully!
Input shape: [1, 2]
Output shape: [1, 1]

Ready for inference!
========================================

Running inference...
Input: [0.50, 0.50]
Output: 0.472900
Inference time: 1234 microseconds
----------------------------------------
```

## 문제 해결

### 모델이 너무 큼

- 모델 크기를 확인: `python scripts/deploy_microcontroller.py --check-only`
- ESP32는 일반적으로 1-2MB 플래시 메모리 사용 가능
- 모델이 너무 크면 양자화(Quantization) 적용 고려

### 메모리 부족 오류

- `kTensorArenaSize`를 증가시키거나 모델을 더 작게 압축
- `platformio.ini`에서 메모리 설정 조정:
  ```ini
  board_build.partitions = huge_app.csv
  ```

### 추론 속도가 느림

- 모델 복잡도 감소
- 양자화 적용
- 하드웨어 가속기 사용 (ESP32의 경우)

### 업로드 실패

- ESP32가 올바르게 연결되었는지 확인
- 포트 권한 확인 (Linux/Mac): `sudo chmod 666 /dev/ttyUSB0`
- 보드 선택 확인: `pio boards esp32`

## 다음 단계

1. 실제 Bot-IoT 모델 배포
2. 하드웨어에서 추론 성능 측정
3. 전력 소비 분석
4. 실시간 추론 검증

## 참고 자료

- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [PlatformIO ESP32 문서](https://docs.platformio.org/en/latest/platforms/espressif32.html)
- [ESP32 개발 보드 정보](https://www.espressif.com/en/products/devkits)

