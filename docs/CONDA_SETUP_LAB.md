# 랩 PC Conda 환경 설정 (Research / TinyML)

학교 랩 컴퓨터에서 리서치용 conda 환경을 만들 때 순서대로 진행하면 됩니다.

---

## 1. Conda 설치 (없을 때만)

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 또는 Anaconda 설치
- 터미널에서 `conda --version` 으로 확인

---

## 2. 환경 만들기

```bash
# Python 3.10 또는 3.11 (TensorFlow 호환)
conda create -n research python=3.10 -y
conda activate research
```

---

## 3. 프로젝트 폴더로 이동

```bash
cd /경로/프로젝트/TinyML-main
# 예: cd ~/Documents/Privacy/Research/TinyML-main
```

---

## 4. 패키지 설치

```bash
# requirements.txt 기준 (랩 PC는 보통 x86/Windows 또는 Linux)
pip install -r requirements.txt

# YAML 설정 파일 읽기용 (없으면 config 로드 시 에러)
pip install pyyaml
```

**참고:** 랩 PC가 **Apple Silicon Mac**이면 `requirements.txt` 그대로 사용. **Windows/Linux**여도 동일한 명령으로 설치 가능.

---

## 5. 동작 확인

```bash
python -c "import tensorflow as tf; import flwr; import yaml; print('OK')"
```

에러 없이 `OK` 가 나오면 환경 설정 완료.

---

## 6. 학습 실행

```bash
conda activate research
cd /경로/TinyML-main
python scripts/train.py
```

config는 `config/federated_local.yaml` 이 기본입니다. 데이터 경로(`data.path`)만 랩 PC에 맞게 수정하면 됩니다.

---

## 요약 체크리스트

| 단계 | 명령 |
|------|------|
| 1 | `conda create -n research python=3.10 -y` |
| 2 | `conda activate research` |
| 3 | `cd TinyML-main` |
| 4 | `pip install -r requirements.txt` |
| 5 | `pip install pyyaml` |
| 6 | `python -c "import tensorflow; import flwr; import yaml; print('OK')"` |

데이터(Bot-IoT 또는 CIC-IDS2017 CSV)는 `data/raw/Bot-IoT/` 등 config에 적힌 경로에 두면 됩니다.
