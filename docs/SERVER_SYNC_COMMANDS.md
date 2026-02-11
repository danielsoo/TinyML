# 서버로 TinyML 폴더 넘기기 (실행 명령어)

**서버에는 TinyML 폴더만 있으면 됩니다.** 아래 명령으로 TinyML만 넘기면 됩니다.

로컬 Mac에서 **터미널**로 실행하세요. 포트(`51586`)와 주소(`84.0.210.73`)는 Vast.ai 인스턴스에 맞게 바꾸세요.

---

## 1. scp로 한 번에 복사

```bash
# Research 폴더가 있는 위치에서 실행
cd /Users/younsoopark/Documents/Privacy/Research

# TinyML 폴더 전체를 서버 /workspace/ 로 복사
scp -P 51586 -r TinyML root@84.0.210.73:/workspace/
```

- 서버에는 `/workspace/TinyML` 로 들어갑니다.
- `-r`: 폴더 전체 복사  
- `-P 51586`: Vast.ai SSH 포트 (대시보드에서 확인)

---

## 2. rsync로 동기화 (변경된 파일만 넘길 때)

```bash
cd /Users/younsoopark/Documents/Privacy/Research

# 로컬 TinyML → 서버 /workspace/TinyML 동기화 (차이만 전송)
rsync -avz -e "ssh -p 51586" \
  --exclude '.git' \
  --exclude '.cursor' \
  --exclude '.stfolder' \
  --exclude 'data/processed/runs' \
  --exclude '.venv' \
  TinyML/ root@84.0.210.73:/workspace/TinyML/
```

- **-a** 보존, **-v** 진행 상황, **-z** 압축
- `--exclude`: 제외할 폴더 (필요 없으면 줄 삭제)
- 마지막에 `/` 있음: `TinyML/` → `TinyML/` 내용만 동기화

---

## 3. 서버에서 확인

```bash
ssh -p 51586 root@84.0.210.73 "ls -la /workspace/TinyML"
```

---

## 4. 한 줄 요약

```bash
scp -P 51586 -r /Users/younsoopark/Documents/Privacy/Research/TinyML root@84.0.210.73:/workspace/
```

포트/IP만 Vast.ai 대시보드에 나온 값으로 바꿔 쓰면 됩니다.
