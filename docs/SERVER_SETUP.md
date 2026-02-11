# 서버(Vast.ai) 설정: TinyML 사용

**서버에는 TinyML 폴더만 있으면 됩니다.** (TinyML-decayinglr은 사용하지 않음)

- **TinyML** 폴더 하나만 `/workspace/TinyML` 에 두고 실행하면 됩니다.
- Git 연결 없이 로컬에서 폴더 자체를 서버로 넘기는 방식입니다.

---

## 1. 워크스페이스 경로

| 이전 | 현재 |
|------|------|
| `/workspace/TinyML-decayinglr` | `/workspace/TinyML` |

---

## 2. 폴더 넘기기 (기본 방식)

- **로컬:** `Research/TinyML` 폴더 전체를 서버로 복사/동기화합니다.
- **서버 경로:** `/workspace/TinyML` 에 두면 됩니다.

**넘기는 방법 예시:**
- **Syncthing:** 로컬 TinyML 폴더를 서버 `/workspace/TinyML` 와 동기화
- **scp:** `scp -P 51586 -r /path/to/TinyML root@84.0.210.73:/workspace/`
- **zip 업로드:** 로컬에서 TinyML 압축 → 서버에 업로드 → 서버에서 `/workspace/` 에 압축 해제

---

## 3. 다시 싱크할 때

코드가 바뀌었으면 **로컬 TinyML 폴더를 다시 서버로 넘기면** 됩니다.

- 서버에 기존 `TinyML-decayinglr` 만 있는 경우:
  - 그 폴더를 삭제하거나 빈 폴더로 두고,
  - 로컬 **TinyML** 폴더를 `/workspace/TinyML` 로 복사/동기화합니다.
- 이미 `/workspace/TinyML` 이 있으면: 같은 방식으로 덮어쓰거나 동기화하면 됩니다.

실행 경로는 항상:

```bash
cd /workspace/TinyML
```

---

## 4. 실행 예시

```bash
cd /workspace/TinyML

# 전체 파이프라인
python run.py --config config/federated.yaml

# 학습 스킵 (기존 모델 사용)
python run.py --config config/federated.yaml --skip-train
```

`RUN_DIR` 등은 `run.py`가 자동으로 `data/processed/runs/<version>/<run_id>/` 를 사용합니다.

---

## 5. 요약

- **서버에 필요한 것:** **TinyML** 폴더만 (`/workspace/TinyML`).
- **방식:** Git 없이 로컬 TinyML 폴더를 서버로 넘김 (Syncthing, scp, zip 등).
- **다시 싱크:** 로컬 TinyML 폴더를 다시 `/workspace/TinyML` 로 복사/동기화.

---

## (참고) Git 사용하는 경우

서버에서 직접 clone 하려면:

```bash
cd /workspace
rm -rf TinyML-decayinglr   # 필요 시
git clone https://github.com/danielsoo/TinyML.git
cd TinyML
```

폴더 넘기기 대신 Git으로 맞추고 싶을 때만 사용하면 됩니다.
