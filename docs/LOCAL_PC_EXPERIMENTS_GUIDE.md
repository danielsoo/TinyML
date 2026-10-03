# 내 PC(Windows)에서 재학습 실험 돌리기 — centralized / non-IID

대상 PC: RTX 5070 / RAM 32 GB / Windows. 리뷰 81B-1(centralized baseline), 81B-2(non-IID)를 채우는 실험입니다.

## 왜 WSL2 + CPU 인가요?
- **GPU 설정은 필요 없습니다.** FL 시뮬레이션 코드(`src/federated/client.py`)는 GPU 메모리 부족 문제 때문에
  원래부터 클라이언트를 **CPU로만** 돌리게 되어 있습니다(예전 Vast.ai 실행도 마찬가지). 모델도 작은 MLP라 GPU 이득이 작습니다.
- **WSL2(Windows 안의 Ubuntu)** 를 쓰는 이유: FL 시뮬레이터(Flower + Ray)가 Windows에서는 불안정합니다
  (레포의 `train_windows.py`가 "no Ray dependency"로 따로 있는 이유). 리눅스에서는 기존 서버들과 똑같이 돌아갑니다.

## 1. WSL2 설치 (한 번만, 10분)
PowerShell을 **관리자 권한**으로 열고:
```powershell
wsl --install -d Ubuntu-24.04
```
재부팅 → Ubuntu 창이 뜨면 사용자 이름/비밀번호 만들기.

## 2. WSL 메모리 늘리기 (한 번만)
기본값은 RAM의 절반(16 GB)이라 전체 데이터셋 + SMOTE에 빠듯할 수 있습니다.
메모장으로 `C:\Users\danie\.wslconfig` 파일을 만들고:
```ini
[wsl2]
memory=24GB
```
저장 후 PowerShell에서 `wsl --shutdown` (다음에 Ubuntu 열면 적용).

## 3. 데이터 넣기
`C:\Users\danie\Documents\Projects\TinyML\data\raw\CIC-IDS2017\` 폴더에 CIC-IDS2017 CSV **8개**
(`Monday-WorkingHours.pcap_ISCX.csv` … `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`, 약 865 MB)를 넣습니다.
- 팀원 Mac의 `drive-download-20260129T185431Z-3-001/` 폴더에 있는 8개 파일, 또는
- 공식 배포본(UNB CIC "CIC-IDS2017" → `MachineLearningCSV.zip`, 압축 풀면 `MachineLearningCVE/` 안에 8개).
파일 이름은 그대로 두세요 (로더가 `*.pcap_ISCX.csv`를 찾습니다).

## 4. 코드 받기 + 환경 설치 (한 번만, 10–20분)
**Windows** PowerShell(평소 쓰던 git)에서:
```powershell
cd C:\Users\danie\Documents\Projects\TinyML
git fetch origin; git checkout claude/eloquent-hypatia-kjfsre; git pull
```
**Ubuntu** 창에서:
```bash
cd /mnt/c/Users/danie/Documents/Projects/TinyML
bash scripts/wsl_setup.sh
```
마지막에 `CIC-IDS2017: 8 CSV files found ✅` 가 나오면 준비 끝.
(git 명령은 계속 **Windows 쪽에서** 하세요. WSL에서 같은 폴더에 git을 쓰면 줄바꿈/권한 차이로 파일이 전부 "수정됨"으로 보일 수 있습니다.)

## 5. 빠른 테스트 먼저 (30분 내외)
```bash
bash scripts/run_revision_experiments.sh --quick
```
5 rounds × 1 epoch로 전체 과정(학습 → 압축 → 평가)이 돌아가는지만 확인합니다. 숫자는 의미 없습니다.
`All steps OK.` 가 나오면 성공. **걸린 시간을 알려주시면 본 실험 시간을 계산해 드립니다.**

## 6. 본 실험 (몇 시간 ~ 하루, 밤새 돌리기)
```bash
nohup bash scripts/run_revision_experiments.sh > revision.out 2>&1 &
tail -f revision.out        # 진행 상황 보기 (Ctrl+C 해도 실험은 계속됨)
```
돌아가는 것:
| 단계 | 내용 | 논문 |
|---|---|---|
| `baseline_ablation` | (a) centralized 180 epoch, (b) FL near-IID 60×3 + 압축 → (c)(d) | 표 2 (a)행, 비교용 (b)~(d) |
| `non_iid` | FL Dirichlet(0.3), 4 clients, 60×3 | 81B-2 |

- 모두 논문 recipe(`config/paper_v12/`: α=0.35, lr 1e-3 cosine, FedAvgM, 전체 데이터) 그대로입니다.
- **절전 모드 끄기**: 설정 → 시스템 → 전원 → 화면/절전 "안 함". Ubuntu 창은 닫지 말고 최소화.
- 중간에 끊겼으면 같은 폴더로 이어서: `bash scripts/run_revision_experiments.sh --out data/processed/revision/<그 폴더>`
  (끝난 단계는 건너뜀)
- 여유가 있으면 추가: `--with-failed` (표 2 (b)행 fixed-LR 재현), `--with-scaling` (클라이언트 20/50개, 81D). 시간이 크게 늘어납니다.

## 7. 결과 넘겨주기
끝나면 `data/processed/revision/<날짜>_full/SUMMARY.md` 에 표가 정리됩니다. **Windows** PowerShell에서:
```powershell
git add data/processed/revision
git commit -m "Revision experiments: centralized + non-IID results"
git push
```
push하면 Claude가 SUMMARY를 읽어서 논문 표 2 (a)행, non-IID 절, 7절 한계 목록을 실제 숫자로 고칩니다.

## 문제 해결
| 증상 | 해결 |
|---|---|
| 로그에 `Killed` / `MemoryError` | 2단계 `.wslconfig` 메모리를 28GB로 올리고 `wsl --shutdown` 후 같은 `--out` 으로 재실행 |
| `bash: $'\r': command not found` | Windows에서 `git pull` 다시 (`.sh`는 LF로 받도록 설정돼 있음). 그래도 나면 `sed -i 's/\r$//' scripts/*.sh` |
| `❌ Need the 8 CIC-IDS2017 ...` | 3단계 파일 위치/이름 확인 |
| `❌ <step> failed` | 출력된 로그 마지막 25줄을 Claude에게 붙여넣기 |

## 8. 자동 반복 모드 (Claude가 결과 확인 → 수정 → 재실행)
PC에서 worker를 한 번 켜두면, 사람이 명령어를 칠 필요 없이 반복됩니다.

```bash
cd /mnt/c/Users/danie/Documents/Projects/TinyML
nohup bash scripts/pc_worker.sh > worker.out 2>&1 &
tail -f worker.out          # 확인용 (Ctrl+C 해도 worker는 계속)
```
- 10분마다 이 브랜치를 pull → `experiments/queue/`에 Claude가 넣어둔 새 작업이 있으면 실행 → 결과를 자동 commit/push.
- 이미 돌고 있는 실험이 있으면 끝날 때까지 기다렸다가, 끝난 결과를 먼저 push합니다.
- git은 Windows의 `git.exe`로 실행되므로 Windows에서 평소 쓰는 GitHub 로그인으로 push됩니다.
- 끄기: `pkill -f "^bash scripts/pc_worker.sh"`
- worker는 `scripts/run_revision_experiments.sh`만, 정해진 옵션으로만 실행합니다 (작업 파일은 설정값일 뿐 명령어가 아님).

### 튜닝 규칙 (논문 신뢰성)
- **고장 난 실행**(에러, NaN, 한 클래스로만 예측하는 붕괴)은 원인을 고쳐서 바로 재실행합니다.
- **정상이지만 숫자가 낮은 결과**(예: non-IID에서 recall 하락)는 그 자체가 리뷰어 질문에 대한 답이므로 그대로 보고합니다.
- 성능을 끌어올리는 튜닝은 **검증 데이터**(`eval_split: val`, 학습 데이터의 10%)로만 고르고, 고른 설정 하나만 마지막에 테스트셋으로 평가합니다. 시도한 설정 수는 논문에 적습니다.
  테스트셋 점수를 보면서 고르면 리뷰어가 "test set에 과적합"이라고 지적할 수 있습니다.
