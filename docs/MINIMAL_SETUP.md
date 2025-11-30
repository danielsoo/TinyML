# 최소 설정 가이드: 학습 시작에 필요한 파일/폴더

학습을 시작하기 위해 필요한 최소한의 파일과 폴더 구조를 설명합니다.

---

## ❌ 데이터만으로는 부족합니다!

**데이터 파일 4개만으로는 학습을 시작할 수 없습니다.** 다음이 모두 필요합니다:

---

## ✅ 필수 파일/폴더 구조

### 1. 데이터 파일 (4개 CSV)
```
data/raw/Bot-IoT/
  ├── reduced_data_1.csv
  ├── reduced_data_2.csv
  ├── reduced_data_3.csv
  └── reduced_data_4.csv
```

**위치:**
- **로컬**: `TinyML/data/raw/Bot-IoT/` 폴더 안
- **Colab**: Google Drive 또는 `/content/TinyML/data/raw/Bot-IoT/`

---

### 2. 설정 파일 (필수)
```
config/
  ├── federated_local.yaml    # 로컬 환경용
  └── federated_colab.yaml     # Colab 환경용
```

**설정 파일 내용:**
- 데이터 경로
- 모델 설정
- 학습 하이퍼파라미터

---

### 3. 소스 코드 (필수)
```
src/
  ├── data/
  │   └── loader.py           # 데이터 로더
  ├── federated/
  │   └── client.py           # FL 클라이언트 (학습 실행)
  └── models/
      └── nets.py              # 모델 정의
```

**필요한 이유:**
- `loader.py`: CSV 파일을 읽고 전처리
- `client.py`: 실제 학습 실행 코드
- `nets.py`: 모델 구조 정의

---

### 4. 의존성 파일 (필수)
```
requirements.txt               # Python 패키지 목록
```

**필요한 패키지:**
- tensorflow
- flwr (Flower)
- pandas, numpy
- scikit-learn
- yaml

---

## 📁 최소 프로젝트 구조

```
TinyML/
├── config/
│   └── federated_local.yaml      # 또는 federated_colab.yaml
├── data/
│   └── raw/
│       └── Bot-IoT/
│           ├── reduced_data_1.csv
│           ├── reduced_data_2.csv
│           ├── reduced_data_3.csv
│           └── reduced_data_4.csv
├── src/
│   ├── data/
│   │   └── loader.py
│   ├── federated/
│   │   └── client.py
│   └── models/
│       └── nets.py
├── requirements.txt
└── (선택) scripts/
    └── run_fl_sim.sh
```

---

## 🚀 학습 시작 방법

### 로컬 환경

**1. 프로젝트 구조 확인**
```bash
# 필수 폴더/파일 확인
ls -la data/raw/Bot-IoT/          # CSV 4개 확인
ls -la config/federated_local.yaml # 설정 파일 확인
ls -la src/data/loader.py          # 코드 확인
```

**2. 의존성 설치**
```bash
pip install -r requirements.txt
pip install flwr[simulation]
```

**3. 학습 실행**
```bash
python -m src.federated.client \
    --config config/federated_local.yaml \
    --save-model src/models/global_model.h5
```

---

### Colab 환경

**1. 데이터 업로드**
- Google Drive에 CSV 4개 업로드
- 또는 Colab에 직접 업로드

**2. 저장소 클론**
```python
!git clone https://github.com/danielsoo/TinyML.git /content/TinyML
```

**3. 설정 파일 수정**
- `config/federated_colab.yaml`에서 데이터 경로 확인

**4. 학습 실행**
```python
!python -m src.federated.client \
    --config config/federated_colab.yaml
```

---

## ❓ 자주 묻는 질문

### Q: 데이터 파일만 있으면 되나요?
**A: 아니요.** 데이터 + 코드 + 설정 파일이 모두 필요합니다.

### Q: `src/models/global_model.h5` 파일이 필요하나요?
**A: 아니요.** 이 파일은 학습 결과로 생성됩니다. 학습 전에는 필요 없습니다.

### Q: `data/processed/` 폴더가 필요하나요?
**A: 아니요.** 학습 시 자동으로 생성됩니다.

### Q: 최소한으로 뭐가 필요한가요?
**A:**
1. 데이터 CSV 4개
2. 설정 파일 (`federated_*.yaml`)
3. 소스 코드 (`src/` 폴더)
4. `requirements.txt`

### Q: GitHub에서 클론하면 다 있나요?
**A: 네!** 하지만 데이터 파일은 별도로 다운로드해야 합니다:
```bash
git clone https://github.com/danielsoo/TinyML.git
cd TinyML
make download-data  # 데이터 다운로드
```

---

## ✅ 체크리스트

학습 시작 전 확인:

- [ ] 데이터 CSV 4개가 `data/raw/Bot-IoT/`에 있음
- [ ] 설정 파일이 `config/`에 있음 (`federated_local.yaml` 또는 `federated_colab.yaml`)
- [ ] 소스 코드가 `src/`에 있음
  - [ ] `src/data/loader.py`
  - [ ] `src/federated/client.py`
  - [ ] `src/models/nets.py`
- [ ] `requirements.txt`가 있음
- [ ] Python 패키지가 설치됨 (`pip install -r requirements.txt`)

---

## 📝 요약

**데이터 4개만으로는 학습할 수 없습니다!**

필요한 것:
1. ✅ 데이터 CSV 4개
2. ✅ 설정 파일 (YAML)
3. ✅ 소스 코드 (Python 파일들)
4. ✅ 의존성 패키지

**가장 쉬운 방법:**
```bash
# GitHub에서 클론 (코드 + 설정)
git clone https://github.com/danielsoo/TinyML.git

# 데이터 다운로드
cd TinyML
make download-data

# 의존성 설치
pip install -r requirements.txt

# 학습 시작
make run-fl
```

이제 모든 것이 준비되었습니다! 🚀

