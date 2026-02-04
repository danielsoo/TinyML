# TinyML-results vs TinyML-main 비교 (확인 일자: 2026-02-04)

## 요약

| 구분 | TinyML-results에만 있음 | TinyML-main에만 있음 |
|------|-------------------------|----------------------|
| **runs** | `v1/2026-01-30_14-34-39` (run 전체) | `v11/2026-02-04_00-55-09`, `v11/2026-02-04_02-49-15`, `v11/2026-02-04_03-25-48` (run 전체) |

- **TinyML-main**에는 서버에서 pull한 **runs**가 들어 있음: `data/processed/runs/<version>/<datetime>/` (analysis, models, outputs, eval 포함).
- **TinyML-results**의 `processed/analysis/` 내용은  
  - 같은 run이 있으면 → TinyML-main에서는 **`data/processed/runs/<version>/<datetime>/analysis/`** 에 있음 (경로만 다름).  
  - run이 없고 analysis만 있던 경우 → TinyML-main의 **`data/processed/analysis/`** 에 있음 (main에도 analysis만 따로 받아 둔 적 있으면).

## 1. Runs (run별 스냅샷: analysis + models + outputs + eval)

- **TinyML-results** `processed/runs/`: v1(1개), v11(4개: 02-02 16,18,22,23시), v3,v4,v5,v6,v7,v8,v8_centralized,v9
- **TinyML-main** `data/processed/runs/`: v11(7개: 위 4개 + **02-04 00-55, 02-49, 03-25**), v2,v3,v4,v5,v6,v7,v8,v8_centralized,v9  
  → **main에만 있는 run**: `v11/2026-02-04_00-55-09`, `v11/2026-02-04_02-49-15`, `v11/2026-02-04_03-25-48`  
  → **results에만 있는 run**: `v1/2026-01-30_14-34-39` (서버 runs에는 없어서 pull 시 안 들어감)

## 2. Analysis만 있는 폴더 (예전 구조)

- **TinyML-results** `processed/analysis/v11/`: 16-13-48, 18-17-38, 19-10-25, 19-11-40, 19-16-09, 22-31-06, 23-28-45, **02-04 00-55-09, 02-49-15, 03-25-48**
- **TinyML-main**  
  - `data/processed/analysis/v11/`: 16-13-48, 18-17-38, 19-10-25, 19-11-40, 19-16-09, 22-31-06, 23-28-45 (02-04 세 개는 **analysis/** 쪽에는 없음)  
  - `data/processed/runs/v11/2026-02-04_*/analysis/`: 02-04 세 run의 분석 결과는 **여기**에 있음 (runs 안에 포함)

→ **결론**: results의 analysis 내용 중 02-04 세 개는 main에 **runs/v11/2026-02-04_*/analysis/** 로 들어가 있음. 경로만 다름.

## 3. 결론: “TinyML-results 전체가 TinyML-main에 다 들어가 있는지”

- **거의 다 들어가 있음.**  
  - results에 있는 run들은 (v1 제외) main의 `data/processed/runs/`에 있고,  
  - results의 `processed/analysis/`에 있는 버전/날짜는 main에  
    - `data/processed/analysis/` 또는  
    - `data/processed/runs/<version>/<datetime>/analysis/`  
  둘 중 한 곳에 있음.
- **TinyML-main에만 있는 것**: v11 2026-02-04 run 3개 (서버에서 pull한 것).
- **TinyML-results에만 있는 것**:  
  - **run 전체**: `v1/2026-01-30_14-34-39`  
  - 이 run을 main에도 두고 싶으면 TinyML-results에서 main으로 복사하면 됨.

## 4. v1 run을 TinyML-main에 넣고 싶을 때

```bash
cp -R /Users/younsoopark/Documents/Privacy/Research/TinyML-results/processed/runs/v1 \
      /Users/younsoopark/Documents/Privacy/Research/TinyML-main/data/processed/runs/
```

이후 main만 써도 results에 있던 v1 run과 동일한 구조가 main에 있음.
