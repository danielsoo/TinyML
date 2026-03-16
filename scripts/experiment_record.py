"""
Shared experiment record writer for run.py and run_sweep_and_pgd.py.
Writes experiment_record.md with Data/Model/Evaluation/AT/Compression/Federated tables.
"""
from datetime import datetime
from pathlib import Path


def write_experiment_record(run_cfg: dict, runs_dir: Path) -> None:
    """Write experiment_record.md with all config used in this run (실험에 사용된 모든 요소 기록)."""
    data_cfg = run_cfg.get("data", {})
    model_cfg = run_cfg.get("model", {})
    eval_cfg = run_cfg.get("evaluation", {})
    at_cfg = run_cfg.get("adversarial_training", {})
    comp_cfg = run_cfg.get("compression", {})
    fed_cfg = run_cfg.get("federated", {})

    def row(key: str, val) -> str:
        v = val if val is not None else "null"
        return f"| {key} | {v} |\n"

    lines = [
        "# 실험 설정 기록 (Experiment Record)",
        "",
        f"- **생성 시각:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- **전체 설정 파일:** 이 디렉터리의 `run_config.yaml`",
        "",
        "## Data",
        "| 항목 | 값 |",
        "|------|-----|",
    ]
    for k, v in data_cfg.items():
        lines.append(row(k, v))
    lines.extend(["", "## Model", "| 항목 | 값 |", "|------|-----|"])
    for k, v in model_cfg.items():
        lines.append(row(k, v))
    lines.extend(["", "## Evaluation", "| 항목 | 값 |", "|------|-----|"])
    for k, v in eval_cfg.items():
        if isinstance(v, list):
            lines.append(row(k, ", ".join(str(x) for x in v)))
        else:
            lines.append(row(k, v))
    lines.extend(["", "## Adversarial training (학습 직후·압축 직전 AT)", "| 항목 | 값 |", "|------|-----|"])
    for k, v in at_cfg.items():
        lines.append(row(k, v))
    lines.extend(["", "## Compression", "| 항목 | 값 |", "|------|-----|"])
    for k, v in comp_cfg.items():
        lines.append(row(k, v))
    lines.extend(["", "## Federated", "| 항목 | 값 |", "|------|-----|"])
    for k, v in fed_cfg.items():
        lines.append(row(k, v))
    lines.append("")

    out = runs_dir / "experiment_record.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"   📋 Saved experiment record: {out}\n")
