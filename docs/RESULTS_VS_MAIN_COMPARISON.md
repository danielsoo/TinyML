# TinyML-results vs TinyML-main comparison (checked 2026-02-04)

## Summary

| Category | Only in TinyML-results | Only in TinyML-main |
|----------|------------------------|----------------------|
| **runs** | `v1/2026-01-30_14-34-39` (full run) | `v11/2026-02-04_00-55-09`, `v11/2026-02-04_02-49-15`, `v11/2026-02-04_03-25-48` (full runs) |

- **TinyML-main** contains **runs** pulled from the server: `data/processed/runs/<version>/<datetime>/` (includes analysis, models, outputs, eval).
- **TinyML-results** `processed/analysis/` content:
  - If the same run exists → in TinyML-main it is under **`data/processed/runs/<version>/<datetime>/analysis/`** (same content, different path).
  - If there is no run and only analysis → in TinyML-main it is under **`data/processed/analysis/`** (when analysis was pulled separately into main).

## 1. Runs (per-run snapshot: analysis + models + outputs + eval)

- **TinyML-results** `processed/runs/`: v1(1), v11(4: 02-02 16,18,22,23h), v3,v4,v5,v6,v7,v8,v8_centralized,v9
- **TinyML-main** `data/processed/runs/`: v11(7: above 4 + **02-04 00-55, 02-49, 03-25**), v2,v3,v4,v5,v6,v7,v8,v8_centralized,v9  
  → **Runs only in main**: `v11/2026-02-04_00-55-09`, `v11/2026-02-04_02-49-15`, `v11/2026-02-04_03-25-48`  
  → **Run only in results**: `v1/2026-01-30_14-34-39` (not on server runs, so not pulled)

## 2. Analysis-only folders (legacy layout)

- **TinyML-results** `processed/analysis/v11/`: 16-13-48, 18-17-38, 19-10-25, 19-11-40, 19-16-09, 22-31-06, 23-28-45, **02-04 00-55-09, 02-49-15, 03-25-48**
- **TinyML-main**  
  - `data/processed/analysis/v11/`: 16-13-48, 18-17-38, 19-10-25, 19-11-40, 19-16-09, 22-31-06, 23-28-45 (the three 02-04 runs are **not** under **analysis/**)  
  - `data/processed/runs/v11/2026-02-04_*/analysis/`: analysis for the three 02-04 runs is **here** (inside runs).

→ **Conclusion**: Of the analysis in results, the three 02-04 runs are in main under **runs/v11/2026-02-04_*/analysis/**. Same content, different path.

## 3. Conclusion: “Is all of TinyML-results contained in TinyML-main?”

- **Almost all of it.**  
  - Runs in results (except v1) are in main’s `data/processed/runs/`, and  
  - Versions/dates under results’ `processed/analysis/` are in main at either  
    - `data/processed/analysis/` or  
    - `data/processed/runs/<version>/<datetime>/analysis/`  
  one or the other.
- **Only in TinyML-main**: v11 2026-02-04 runs (3), pulled from the server.
- **Only in TinyML-results**:  
  - **Full run**: `v1/2026-01-30_14-34-39`  
  - To have this run in main too, copy it from TinyML-results to main.

## 4. Adding the v1 run to TinyML-main

```bash
cp -R /Users/younsoopark/Documents/Privacy/Research/TinyML-results/processed/runs/v1 \
      /Users/younsoopark/Documents/Privacy/Research/TinyML-main/data/processed/runs/
```

After that, using only main gives you the same v1 run structure as in results.
