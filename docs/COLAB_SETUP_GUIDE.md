# Google Colab 완전 가이드

Colab에서 TinyML 프로젝트를 처음부터 끝까지 실행하는 방법을 단계별로 설명합니다.

---

## 📋 목차

1. [Colab 런타임 설정](#1-colab-런타임-설정)
2. [노트북 열기](#2-노트북-열기)
3. [셀 실행 순서](#3-셀-실행-순서)
4. [터미널 사용법](#4-터미널-사용법)
5. [문제 해결](#5-문제-해결)

---

## 1. Colab 런타임 설정

### 1.1 Colab 접속

1. **Google Colab 열기**
   - 브라우저에서 https://colab.research.google.com 접속
   - 또는 Google Drive에서 "새로 만들기" → "더보기" → "Google Colaboratory"

### 1.2 노트북 업로드

**방법 A: GitHub에서 직접 열기 (권장)**
1. Colab 메인 페이지에서 "GitHub" 탭 클릭
2. GitHub URL 입력: `https://github.com/danielsoo/TinyML`
3. `colab/train_colab.ipynb` 선택

**방법 B: 파일 업로드**
1. Colab 메인 페이지에서 "파일" → "노트북 업로드"
2. 로컬의 `colab/train_colab.ipynb` 파일 선택

### 1.3 런타임 타입 설정 (중요!)

**GPU 활성화:**
1. 상단 메뉴: **런타임** → **런타임 유형 변경**
2. **하드웨어 가속기**: "GPU" 선택
   - T4 GPU (무료) 또는 V100/A100 (Pro)
3. **런타임 유형**: "Python 3" 선택
4. **저장** 클릭

**확인 방법:**
```python
# 첫 번째 셀에서 실행
!nvidia-smi
```
GPU 정보가 표시되면 성공!

---

## 2. 노트북 열기

### 2.1 노트북 구조 확인

노트북은 다음 섹션으로 구성되어 있습니다:

1. **구글 드라이브 연결** (셀 1)
2. **Runtime & GPU 체크** (셀 2-3)
3. **저장소 클론/업데이트** (셀 4-5)
4. **설정 파일 업데이트** (셀 6-7)
5. **의존성 설치** (셀 8-12)
6. **데이터셋 준비** (셀 13-14)
7. **학습 실행** (셀 15-17)
8. **모델 저장** (셀 18-19)
9. **압축 분석** (셀 20-24) ⭐ 새로 추가됨

---

## 3. 셀 실행 순서

### 3.1 순차 실행 (권장)

**방법 1: 전체 실행**
- 상단 메뉴: **런타임** → **모두 실행**
- 또는 `Ctrl+F9` (Windows) / `Cmd+F9` (Mac)

**방법 2: 셀별 실행**
- 각 셀에서 `Shift+Enter` 또는 셀 왼쪽의 ▶ 버튼 클릭

### 3.2 필수 실행 순서

```
셀 1: 구글 드라이브 연결
  ↓
셀 2-3: GPU 확인
  ↓
셀 4-5: 저장소 클론
  ↓
셀 6-7: 설정 파일 업데이트
  ↓
셀 8-12: 의존성 설치
  ↓
셀 13-14: 데이터셋 확인 (선택)
  ↓
셀 15-17: 학습 실행
  ↓
셀 18-19: 모델 저장
  ↓
셀 20-24: 압축 분석 (선택)
```

---

## 4. 터미널 사용법

### 4.1 Colab 터미널 열기

**방법 1: 셀에서 터미널 명령 실행**
```python
# Python 셀에서
!ls -la
!pwd
!python --version
```

**방법 2: 터미널 모드 사용**
- 셀 타입을 "코드" → "텍스트"로 변경하면 마크다운
- 코드 셀에서 `!` 접두사 사용

### 4.2 자주 사용하는 터미널 명령어

#### 디렉토리 확인
```python
!pwd  # 현재 경로
!ls -la  # 파일 목록
!cd /content/TinyML && ls  # 프로젝트 디렉토리로 이동 후 목록
```

#### 파일 확인
```python
!cat config/federated_colab.yaml  # 파일 내용 보기
!head -20 src/federated/client.py  # 파일 앞부분 보기
```

#### Python 스크립트 실행
```python
# 방법 1: ! 사용
!python scripts/analyze_compression.py --help

# 방법 2: %system 사용 (출력 캡처)
%system python scripts/analyze_compression.py --models "Baseline:src/models/global_model.h5"
```

#### 패키지 설치
```python
!pip install pandas matplotlib seaborn
!pip list | grep tensorflow  # 특정 패키지 확인
```

#### Git 명령어
```python
!git status
!git log --oneline -5  # 최근 5개 커밋
!git pull  # 저장소 업데이트
```

### 4.3 파일 다운로드/업로드

#### 파일 다운로드
```python
from google.colab import files

# 단일 파일
files.download('data/processed/analysis/compression_analysis.csv')

# 여러 파일 (zip으로)
!zip -r results.zip data/processed/analysis/
files.download('results.zip')
```

#### 파일 업로드
```python
from google.colab import files

uploaded = files.upload()  # 파일 선택 창이 열림
# 업로드된 파일은 현재 디렉토리에 저장됨
```

### 4.4 환경 변수 설정
```python
import os

# 환경 변수 설정
os.environ['FEDERATED_CONFIG'] = 'config/federated_colab.yaml'

# 확인
print(os.getenv('FEDERATED_CONFIG'))
```

---

## 5. 문제 해결

### 5.1 런타임 연결 끊김

**증상**: "런타임에 연결할 수 없습니다"

**해결 방법:**
1. **런타임** → **런타임 다시 시작**
2. 또는 **런타임** → **런타임 다시 시작하고 모두 실행**

**주의**: 런타임 재시작 시 변수와 설치된 패키지가 초기화됩니다!

### 5.2 GPU 메모리 부족

**증상**: "OOM (Out of Memory)" 에러

**해결 방법:**
```python
# GPU 메모리 정리
import tensorflow as tf
tf.keras.backend.clear_session()

# 또는 런타임 재시작
```

**설정 파일 수정:**
```yaml
# config/federated_colab.yaml
data:
  max_samples: 50000  # 줄이기 (기본값: 200000)
```

### 5.3 패키지 설치 실패

**증상**: `pip install` 실패

**해결 방법:**
```python
# pip 업그레이드
!pip install --upgrade pip

# 특정 버전 설치
!pip install tensorflow==2.12.0

# 강제 재설치
!pip install --force-reinstall package_name
```

### 5.7 Protobuf 호환성 오류

**증상**: `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`

**원인**: TensorFlow와 protobuf 버전 간 호환성 문제

**해결 방법:**

**방법 1: Protobuf 버전 다운그레이드 (권장)**
```python
# Protobuf 3.20.x로 다운그레이드
!pip install protobuf==3.20.3

# 런타임 재시작 필요
# Runtime → Restart runtime
```

**방법 2: Protobuf 4.x로 업그레이드**
```python
# Protobuf 4.x로 업그레이드
!pip install --upgrade protobuf>=4.21.0

# 런타임 재시작
# Runtime → Restart runtime
```

**방법 3: TensorFlow 버전 다운그레이드**
```python
# TensorFlow 2.12.0 사용 (더 안정적)
!pip install tensorflow==2.12.0

# 런타임 재시작
# Runtime → Restart runtime
```

**방법 4: 경고 무시 (임시 해결책)**
```python
# 경고를 무시하고 계속 진행
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
# TensorFlow는 정상 작동하지만 경고가 계속 표시됨
```

**확인:**
```python
import tensorflow as tf
import google.protobuf
print(f"TensorFlow: {tf.__version__}")
print(f"Protobuf: {google.protobuf.__version__}")

# 정상 작동 확인
print("GPU devices:", tf.config.list_physical_devices('GPU'))
```

**참고**: 이 경고는 TensorFlow가 정상 작동하는데도 나타날 수 있습니다. GPU가 인식되고 학습이 진행되면 무시해도 됩니다.

### 5.4 파일 경로 오류

**증상**: "FileNotFoundError"

**해결 방법:**
```python
# 현재 디렉토리 확인
!pwd

# 파일 존재 확인
import os
print(os.path.exists('src/models/global_model.h5'))

# 절대 경로 사용
path = '/content/TinyML/src/models/global_model.h5'
```

### 5.5 Google Drive 마운트 실패

**증상**: "Drive mount failed"

**해결 방법:**
```python
# 기존 마운트 해제 후 재마운트
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive')
```

### 5.6 저장소 클론 실패

**증상**: "git clone" 실패

**해결 방법:**
```python
# 기존 디렉토리 삭제 후 재클론
!rm -rf /content/TinyML
!git clone https://github.com/danielsoo/TinyML.git /content/TinyML
```

### 5.8 ModuleNotFoundError: No module named 'src'

**증상**: `ModuleNotFoundError: No module named 'src'` 또는 `ModuleNotFoundError: No module named 'src.data'`

**원인**: Python이 프로젝트 디렉토리를 모듈 경로에서 찾지 못함

**해결 방법:**

**방법 1: sys.path에 프로젝트 디렉토리 추가 (권장)**
```python
import os
import sys

PROJECT_DIR = "/content/TinyML"
os.chdir(PROJECT_DIR)

# 프로젝트 디렉토리를 Python 경로에 추가
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# 이제 스크립트 실행 가능
!python scripts/analyze_compression.py --help
```

**방법 2: PYTHONPATH 환경 변수 설정**
```python
import os
os.environ['PYTHONPATH'] = '/content/TinyML'

# 또는 셸에서
!export PYTHONPATH=/content/TinyML:$PYTHONPATH
!python scripts/analyze_compression.py --help
```

**방법 3: -m 옵션 사용 (프로젝트 루트에서)**
```python
import os
os.chdir('/content/TinyML')

# Python 모듈로 실행
!python -m scripts.analyze_compression --help
```

**방법 4: 절대 경로로 실행**
```python
# 프로젝트 루트에서 실행
!cd /content/TinyML && python scripts/analyze_compression.py --help
```

**확인:**
```python
import sys
print("Python path:")
for p in sys.path[:5]:  # 처음 5개만 출력
    print(f"  {p}")
```

---

## 6. 실전 예제

### 6.1 전체 워크플로우 (터미널 명령어로)

```python
# 1. 디렉토리 확인
!pwd
!ls -la

# 2. 저장소 클론
!git clone https://github.com/danielsoo/TinyML.git /content/TinyML
!cd /content/TinyML && pwd

# 3. 의존성 설치
!cd /content/TinyML && pip install -r requirements.txt
!cd /content/TinyML && pip install flwr[simulation]

# 4. 설정 확인
!cd /content/TinyML && cat config/federated_colab.yaml

# 5. 학습 실행
!cd /content/TinyML && python -m src.federated.client \
    --config config/federated_colab.yaml \
    --save-model src/models/global_model.h5

# 6. 모델 확인
!cd /content/TinyML && ls -lh src/models/

# 7. 분석 실행
!cd /content/TinyML && python scripts/analyze_compression.py \
    --models "Baseline:src/models/global_model.h5" \
    --config config/federated_colab.yaml

# 8. 결과 확인
!cd /content/TinyML && ls -la data/processed/analysis/
!cd /content/TinyML && cat data/processed/analysis/compression_analysis.md
```

### 6.2 단계별 디버깅

```python
# 각 단계마다 확인
import os
from pathlib import Path

# 1. 프로젝트 디렉토리 확인
project_dir = Path("/content/TinyML")
print(f"Project exists: {project_dir.exists()}")
if project_dir.exists():
    print(f"Files: {list(project_dir.iterdir())[:10]}")

# 2. 모델 파일 확인
model_path = project_dir / "src/models/global_model.h5"
print(f"Model exists: {model_path.exists()}")
if model_path.exists():
    print(f"Model size: {model_path.stat().st_size / 1024:.2f} KB")

# 3. 설정 파일 확인
config_path = project_dir / "config/federated_colab.yaml"
print(f"Config exists: {config_path.exists()}")

# 4. 데이터 경로 확인
import yaml
if config_path.exists():
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_path = cfg.get("data", {}).get("path")
    print(f"Data path: {data_path}")
    print(f"Data path exists: {Path(data_path).exists() if data_path else False}")
```

---

## 7. 유용한 팁

### 7.1 세션 관리

**세션 시간 연장:**
- 무료 버전: 12시간 (비활성 시 끊김)
- Pro 버전: 24시간

**세션 유지:**
```python
# 주기적으로 실행 (자동화)
import time
while True:
    time.sleep(300)  # 5분마다
    print("Session alive")
```

### 7.2 출력 저장

```python
# 출력을 파일로 저장
!python script.py > output.txt 2>&1

# 또는 Python에서
import subprocess
result = subprocess.run(['python', 'script.py'], 
                       capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
```

### 7.3 진행 상황 모니터링

```python
# tqdm 사용
from tqdm import tqdm
import time

for i in tqdm(range(100)):
    time.sleep(0.1)
```

### 7.4 메모리 사용량 확인

```python
# 메모리 사용량
!free -h

# GPU 메모리
!nvidia-smi

# Python 메모리
import psutil
print(f"RAM: {psutil.virtual_memory().percent}%")
```

---

## 8. 체크리스트

실행 전 확인사항:

- [ ] GPU 런타임 활성화됨
- [ ] Google Drive 마운트됨
- [ ] 저장소 클론됨
- [ ] 의존성 설치됨
- [ ] 설정 파일 경로 확인됨
- [ ] 데이터 파일 경로 확인됨
- [ ] 충분한 디스크 공간 (최소 5GB)

---

## 9. 빠른 참조

### 필수 명령어

```python
# 현재 위치
!pwd

# 파일 목록
!ls -la

# Python 버전
!python --version

# TensorFlow 버전
!python -c "import tensorflow as tf; print(tf.__version__)"

# GPU 확인
!nvidia-smi

# 프로젝트 디렉토리로 이동
import os
os.chdir('/content/TinyML')
```

### 자주 사용하는 경로

```
/content/TinyML/                    # 프로젝트 루트
/content/TinyML/src/models/         # 모델 저장 위치
/content/drive/MyDrive/TinyML_models/   # 데이터 위치 (CSV 파일들이 직접 있음)
/content/TinyML/config/            # 설정 파일
/content/drive/MyDrive/             # Google Drive
```

---

이제 Colab에서 모든 작업을 수행할 수 있습니다! 🚀

