#!/usr/bin/env python3
"""
Write sweep_summary.md from sweep_compression_grid_with_pgd.csv.
One-page summary: run description, column meanings, top 5 by final_f1,
bottom 5 by final_size_kb (smallest models), optional recommended rows.

Usage:
  python scripts/write_sweep_summary.py --run-dir data/processed/runs/sweep_pgd/2026-03-14_12-00-00
"""
import argparse
import csv
import sys
from pathlib import Path


COLUMN_DESCRIPTIONS = [
    ("tag", "조합 식별자 (fl_qat__distill_*__pruning__ptq_*)"),
    ("tflite_path", "TFLite 모델 파일 경로"),
    ("fl_qat", "FL 후 QAT 적용 여부 (True/False)"),
    ("distillation", "지식 증류 방식: none / direct / progressive"),
    ("pruning", "프루닝: prune_none / prune_10x5 / prune_5x5 / prune_3x3"),
    ("ptq", "PTQ 양자화 여부 (True/False)"),
    ("fl_acc", "FL 단계 정확도"),
    ("fl_f1", "FL 단계 F1"),
    ("prune_acc", "프루닝 직후 정확도"),
    ("prune_f1", "프루닝 직후 F1"),
    ("final_acc", "최종 TFLite 정확도"),
    ("final_f1", "최종 TFLite F1 (클수록 좋음)"),
    ("final_size_kb", "최종 TFLite 크기 (KB, 작을수록 좋음)"),
    ("pgd_adv_acc", "PGD 공격 후 방어 정확도"),
    ("pgd_success_rate", "PGD 공격 성공률"),
]


def _safe_float(val, default=None):
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def main():
    parser = argparse.ArgumentParser(description="Write sweep_summary.md from sweep CSV")
    parser.add_argument("--run-dir", required=True, help="Run directory containing sweep_compression_grid_with_pgd.csv")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top rows by final_f1 and by smallest size (default: 5)")
    parser.add_argument("--recommend-f1", type=float, default=0.9, help="Recommend rows with final_f1 >= this (default: 0.9)")
    parser.add_argument("--recommend-size-kb", type=float, default=200.0, help="Recommend rows with final_size_kb <= this (default: 200)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    csv_path = run_dir / "sweep_compression_grid_with_pgd.csv"
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    rows = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)

    if not rows:
        print("CSV has no data rows.", file=sys.stderr)
        return 1

    # Sort by final_f1 desc for top-N; by final_size_kb asc for smallest
    def f1_key(r):
        v = _safe_float(r.get("final_f1"))
        return (v if v is not None else -1.0)

    def size_key(r):
        v = _safe_float(r.get("final_size_kb"))
        return (v if v is not None else float("inf"))

    top_f1 = sorted(rows, key=f1_key, reverse=True)[: args.top_n]
    top_small = sorted(rows, key=size_key)[: args.top_n]

    recommended = [
        r for r in rows
        if _safe_float(r.get("final_f1"), 0) >= args.recommend_f1
        and _safe_float(r.get("final_size_kb"), float("inf")) <= args.recommend_size_kb
    ]

    lines = [
        "# 스윕 요약 (Sweep Summary)",
        "",
        "이 run은 **압축 그리드 스윕(fl_qat × distillation × pruning × ptq) 48조합** 실행 후, 모든 모델에 대해 **PGD**를 수행하고 결과를 병합한 결과입니다.",
        "",
        "## CSV 컬럼 설명",
        "",
        "| 컬럼 | 설명 |",
        "|------|------|",
    ]
    for col, desc in COLUMN_DESCRIPTIONS:
        lines.append(f"| {col} | {desc} |")

    lines.extend([
        "",
        f"## final_f1 상위 {args.top_n}개",
        "",
        "| 순위 | tag | final_f1 | final_size_kb | pgd_adv_acc |",
        "|------|-----|----------|---------------|-------------|",
    ])
    for i, r in enumerate(top_f1, 1):
        tag = (r.get("tag") or "").replace("|", "\\|")
        f1 = r.get("final_f1", "")
        size = r.get("final_size_kb", "")
        pgd = r.get("pgd_adv_acc", "")
        lines.append(f"| {i} | {tag} | {f1} | {size} | {pgd} |")

    lines.extend([
        "",
        f"## final_size_kb 하위 {args.top_n}개 (가장 작은 모델)",
        "",
        "| 순위 | tag | final_size_kb | final_f1 | pgd_adv_acc |",
        "|------|-----|---------------|----------|-------------|",
    ])
    for i, r in enumerate(top_small, 1):
        tag = (r.get("tag") or "").replace("|", "\\|")
        size = r.get("final_size_kb", "")
        f1 = r.get("final_f1", "")
        pgd = r.get("pgd_adv_acc", "")
        lines.append(f"| {i} | {tag} | {size} | {f1} | {pgd} |")

    lines.extend([
        "",
        f"## 추천 조합 (final_f1 >= {args.recommend_f1} & final_size_kb <= {args.recommend_size_kb})",
        "",
    ])
    if not recommended:
        lines.append("조건을 만족하는 행이 없습니다.")
    else:
        lines.append("| tag | final_f1 | final_size_kb | pgd_adv_acc |")
        lines.append("|-----|----------|---------------|-------------|")
        for r in recommended:
            tag = (r.get("tag") or "").replace("|", "\\|")
            f1 = r.get("final_f1", "")
            size = r.get("final_size_kb", "")
            pgd = r.get("pgd_adv_acc", "")
            lines.append(f"| {tag} | {f1} | {size} | {pgd} |")
    lines.append("")

    out_path = run_dir / "sweep_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"   📋 Saved sweep summary: {out_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
