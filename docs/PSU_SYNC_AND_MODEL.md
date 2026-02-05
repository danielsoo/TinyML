# PSU 서버 동기화 & 서버에서 모델 찾기

## 서버 → 로컬 (결과만 받기)

**TinyML-main 하나만 사용.** 서버에서 가져온 결과(분석·모델·eval·outputs)는 **전부 TinyML-main 안**에 들어갑니다.

```bash
cd /path/to/TinyML-main
bash scripts/pull_results_to_tinyml_results.sh
```

또는 Research 폴더에서:

```bash
cd /path/to/Research
bash TinyML-main/scripts/pull_results_to_tinyml_results.sh
```

- **받는 곳**: **TinyML-main** (`data/processed/runs/`, `models/`, `outputs/`)
- **폴더 구조**: 한 run = 한 폴더  
  `TinyML-main/data/processed/runs/<version>/<datetime>/`  
  - `run_config.yaml` — 해당 run에 사용한 전체 설정 (라운드, 에폭, balance_ratio 등)  
  - `analysis/` — compression_analysis.md (내부에 **Run / Training Configuration** 섹션 포함), 시각화  
  - `models/` — tflite, h5 등  
  - `outputs/`  
  - `eval/` — ratio_sweep_report.md (동일하게 학습 설정 요약 포함)  
- **예**: `TinyML-main/data/processed/runs/v11/2026-02-02_23-28-45/models/tflite/saved_model_original.tflite`

**기존 run 보고서에 학습 설정 넣기 (한 번만 실행):**  
이전 버전 run들 `compression_analysis.md` / `ratio_sweep_report.md`에 "Run / Training Configuration" 섹션이 없으면 한 번에 채우려면:

```bash
cd /path/to/TinyML-main
# conda/venv 활성화 후 (pyyaml 필요)
python scripts/backfill_run_config_to_reports.py
```

- `--dry-run`: 실제 수정 없이 어떤 파일이 바뀔지만 출력  
- 각 run 폴더에 `run_config.yaml`이 없으면 버전별로 추정한 config로 저장  
- `analysis/compression_analysis.md`, `eval/ratio_sweep_report.md`에 섹션이 없으면 삽입

rsync 직접 쓰려면:

```bash
cd /path/to/TinyML-main
rsync -avz yqp5187@e5-cse-135-01.cse.psu.edu:/scratch/yqp5187/TinyML-main/data/processed/runs/ data/processed/runs/
rsync -avz yqp5187@e5-cse-135-01.cse.psu.edu:/scratch/yqp5187/TinyML-main/models/ models/
rsync -avz yqp5187@e5-cse-135-01.cse.psu.edu:/scratch/yqp5187/TinyML-main/outputs/ outputs/
```

**옵션:** `sync_results_from_psu.sh`는 `data/processed/` (analysis, runs)만 동기화합니다. run별 모델·eval 전부가 필요하면 위 `pull_results_to_tinyml_results.sh`를 쓰면 됩니다.

**`command not found` / `invalid option` / `hostname contains invalid characters` 나올 때:** 스크립트가 CRLF(Windows 줄바꿈)로 저장됐을 수 있습니다. TinyML-main 폴더에서 한 번만 실행:

```bash
sed -i '' 's/\r$//' scripts/pull_results_to_tinyml_results.sh
```

그 다음 다시 `bash scripts/pull_results_to_tinyml_results.sh` 실행.

---

## 로컬 → 서버 (코드 올리기)

변경된 로컬 파일을 서버로 올릴 때:

```bash
cd /path/to/TinyML-main
bash scripts/sync_to_psu.sh
```

- 올라가는 것: `config/`, `scripts/`, `src/` (코드), `docs/` 등
- 제외: `data/raw`, `data/processed`, `models/`, `src/models/*.h5`, `outputs`, `TinyML-results`, `.git` 등 (대용량·생성물은 제외)

---

## 서버에서 모델 찾는 방법

`train.py`로 학습을 끝내면 모델이 아래 위치에 저장됩니다.

| 파일 | 설명 |
|------|------|
| `src/models/global_model.h5` | 최신 복사본 (항상 마지막 학습 결과로 덮어씀) |
| `src/models/global_model_YYYYMMDD_HHMMSS.h5` | 타임스탬프 파일 (예: v11 실행 시 생성된 파일) |

**서버 접속 후 확인:**

```bash
ssh yqp5187@e5-cse-135-01.cse.psu.edu
cd /scratch/yqp5187/TinyML-main
ls -la src/models/
```

**9:1 테스트만 할 때 (학습 건너뛰고):**

```bash
cd /scratch/yqp5187/TinyML-main
conda activate /scratch/yqp5187/conda_envs/research   # 또는 사용 중인 env
python scripts/evaluate_9to1.py --config config/federated_local.yaml --model src/models/global_model.h5 --ratio 9
```

특정 타임스탬프 모델로 테스트하려면:

```bash
python scripts/evaluate_9to1.py --config config/federated_local.yaml --model src/models/global_model_20260202_232845.h5 --ratio 9
```

(실제 파일명은 `ls src/models/`로 확인)

---

## 비율에 맞는 임계값 추천 (tune_threshold.py)

특정 비율(예: 9:1)에서 **어떤 threshold를 쓰면 좋을지** 추천받으려면:

```bash
cd /path/to/TinyML-main
python scripts/tune_threshold.py --config config/federated_scratch.yaml --model models/tflite/saved_model_original.tflite --ratio 90
```

- `--ratio 90`: 정상 90% : 공격 10% 테스트셋으로 threshold 탐색
- `--metric f1`: F1 최대화로 추천 (기본)
- `--metric attack_recall`: 공격 Recall 최대화 (공격을 더 많이 잡고 싶을 때)
- `--metric balanced`: 0.5*F1 + 0.5*attack_recall 로 추천
- `--out threshold_90.csv`: 구간별 지표 CSV 저장

출력 예: 여러 threshold에서 Accuracy, F1, Normal Recall, Normal Precision, Attack Recall을 보여주고, 선택한 metric 기준 **추천 threshold**를 한 줄로 출력합니다.

---

## 모델 지정해서 평가만 돌리기 (폴더 구조)

학습/압축 없이 **지정한 모델**로 ratio sweep + threshold tuning만 돌리고, 그 결과를 보고서로 남기려면 `run.py --model <경로>` 를 쓰면 됩니다. 보고서가 저장되는 폴더는 아래처럼 정해집니다.

| 상황 | 보고서 저장 위치 |
|------|------------------|
| `--model` 이 **runs 아래** 모델일 때<br/>예: `data/processed/runs/v11/2026-02-02_23-28-45/models/tflite/saved_model_original.tflite` | **그 run의 `eval/`**<br/>→ `data/processed/runs/v11/2026-02-02_23-28-45/eval/ratio_sweep_report.md` |
| `--model` 이 지정됐지만 runs 밖 경로일 때<br/>+ `.last_run_id` 있음 (이전에 분석 실행함) | **현재 분석 폴더**<br/>→ `data/processed/analysis/<version>/<datetime>/ratio_sweep_report.md` |
| `--model` 만 지정하고 분석 폴더/run 없음 | **eval 전용 폴더**<br/>→ `data/processed/eval/<YYYY-MM-DD_HH-MM-SS>/ratio_sweep_report.md` |
| `--model` 없이 파이프라인 실행 (기본) | **현재 분석 폴더**<br/>→ `data/processed/analysis/<version>/<datetime>/ratio_sweep_report.md` |

**예시 (학습/압축/분석 생략, 특정 run 모델로 평가만):**

```bash
python run.py --skip-train --skip-compression --skip-analysis \
  --model data/processed/runs/v11/2026-02-02_23-28-45/models/tflite/saved_model_original.tflite
```

→ `data/processed/runs/v11/2026-02-02_23-28-45/eval/ratio_sweep_report.md` 에 ratio sweep + threshold tuning 결과가 쌓입니다. 같은 run의 모델을 여러 번 평가해도 같은 `eval/` 폴더에 덮어쓰게 됩니다.

**폴더 구조 요약**

- **한 번의 학습 run** → `data/processed/runs/<version>/<datetime>/`  
  - `models/`, `outputs/`, `analysis/` (스냅샷)  
  - **지정 모델로 평가만 돌리면** → 같은 run 아래 `eval/ratio_sweep_report.md`
- **학습 없이 임의 모델로만 평가** → `data/processed/eval/<timestamp>/ratio_sweep_report.md`

---

## compression_analysis vs ratio_sweep 같은 데이터인지 확인

두 리포트가 **같은 테스트 데이터**로 나왔는지 확인하려면:

1. **실행 로그에서 확인**
   - `run.py` 또는 `analyze_compression.py` 실행 시: `[load_cicids2017] data_path=...` 와 `Test set loaded: N samples` 가 찍힘.
   - `evaluate_ratio_sweep.py` 실행 시: 같은 `[load_cicids2017] data_path=...` 와 `Test: N (Normal=..., Attack=...)` 가 찍힘.
   - **data_path**가 같고 **Test set 크기(N)**가 같으면 같은 데이터로 평가한 것.

2. **서버에서 한 번에 돌릴 때**
   - `run.py` 한 번에 돌리면 compression 분석과 ratio sweep이 **같은 환경**에서 순서대로 실행되므로, `CICIDS2017_DATA_PATH` 를 설정하지 않으면 둘 다 config의 `data.path` (예: `data/raw/Bot-IoT`) 를 씀.
   - 서버에서 `export CICIDS2017_DATA_PATH=/scratch/yqp5187/Bot-IoT` 한 뒤 `run.py` 를 돌리면 둘 다 그 경로를 씀.
   - 따라서 **같은 셸에서 한 번만** `run.py` 를 실행했다면, 두 단계는 같은 데이터를 쓴다.

3. **다른 결과가 나온다면**
   - compression은 높은데 ratio_sweep은 Normal Recall 0이면: 예전에 **다른 환경**(예: 로컬 vs 서버, 또는 env 설정 전/후)에서 각각 돌렸을 가능성이 있음.
   - **모델이 다르면** 결과도 다름: compression_analysis는 **TFLite** (Original/PTQ), ratio_sweep은 **Keras .h5** 를 쓰면 서로 다른 예측이 나올 수 있음. `run.py` 는 ratio_sweep에 **같은 Original TFLite** (`models/tflite/saved_model_original.tflite`) 를 쓰도록 되어 있어서, 같은 run 이면 두 리포트가 같은 모델로 평가됨.
   - 다음부터는 위 1번처럼 로그에 찍힌 `data_path` 와 테스트 샘플 수를 비교하면 됨.

---

## 서버 안에서 데이터 찾는 방법

SSH로 서버 접속한 뒤, CIC-IDS2017 CSV가 어디 있는지 찾을 때:

```bash
# 1) 내 스크래치 아래에서 *.pcap_ISCX.csv 검색 (시간 걸릴 수 있음)
find /scratch/yqp5187 -name "*.pcap_ISCX.csv" -type f 2>/dev/null

# 2) 파일명에 CIC 또는 ISCX 포함된 CSV만 검색
find /scratch/yqp5187 -name "*CIC*" -o -name "*ISCX*" 2>/dev/null | head -50

# 3) 먼저 상위 디렉터리만 보고, 데이터 폴더 후보 확인
ls -la /scratch/yqp5187/
ls -la /scratch/yqp5187/TinyML-main/data/raw/ 2>/dev/null
```

찾은 디렉터리 경로(예: `/scratch/yqp5187/datasets/cicids2017`)를 `CICIDS2017_DATA_PATH`로 쓰면 됩니다.

---

## 서버에서 데이터 경로가 다를 때 (CIC-IDS2017)

9:1 평가나 학습 시 `No CIC-IDS2017 CSV files (*.pcap_ISCX.csv) found in data/raw/Bot-IoT` 가 나오면, 서버에 데이터가 다른 폴더에 있다는 뜻입니다.

**방법 1: 환경 변수로 경로 지정**

서버에서 CIC-IDS2017 CSV 파일들이 있는 디렉터리로 경로를 지정한 뒤 실행:

```bash
export CICIDS2017_DATA_PATH=/scratch/yqp5187/경로/  # 실제 *.pcap_ISCX.csv 가 있는 폴더
python scripts/evaluate_9to1.py --config config/federated_local.yaml --model src/models/global_model.h5 --ratio 9
```

`tune_threshold_all_ratios.py`, `evaluate_ratio_sweep.py`, `run.py` 등 **데이터를 읽는 모든 스크립트**도 같은 방식으로 `CICIDS2017_DATA_PATH`를 설정한 뒤 실행하면 됩니다.

**방법 2: 데이터를 config 경로에 두기**

`data/raw/Bot-IoT` (또는 config의 `data.path`) 아래에 `*.pcap_ISCX.csv` 파일들을 넣거나, 그 경로를 해당 폴더로 심볼릭 링크:

```bash
mkdir -p /scratch/yqp5187/TinyML-main/data/raw/Bot-IoT
# 여기에 CIC-IDS2017 CSV 복사 또는 ln -s /실제경로/*.pcap_ISCX.csv .
```

데이터를 아직 서버에 두지 않았다면, [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)에서 받아 서버의 한 폴더에 풀어 둔 다음, 그 폴더 경로를 `CICIDS2017_DATA_PATH`로 쓰면 됩니다.
