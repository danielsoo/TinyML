#!/usr/bin/env python3
"""Generate FL evaluation report in Markdown.

Reads outputs/fl_evaluation_history.json and creates outputs/fl_evaluation_report.md.
- Per-round mean and per-device accuracy (%), loss
- For identifying problem segments (accuracy drops)
"""
import json
from pathlib import Path
from datetime import datetime


def generate_fl_report(json_path: Path, output_dir: Path, report_name: str = "fl_evaluation_report.md") -> Path:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rounds_data = data.get("rounds", [])
    num_clients = int(data.get("num_clients", 0))
    num_rounds = int(data.get("num_rounds", 0))

    def _safe_float(v, default=0.0):
        if v is None or (isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf"))):
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    if not rounds_data:
        raise ValueError("No rounds in history. Run FL training first.")

    out_path = output_dir / report_name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# FL Evaluation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Total rounds**: {num_rounds}\n")
        f.write(f"- **Number of devices (clients)**: {num_clients}\n")
        if rounds_data:
            last = rounds_data[-1]
            f.write(f"- **Final mean accuracy**: {_safe_float(last.get('accuracy_pct'), 0):.2f}%\n")
            f.write(f"- **Final loss**: {_safe_float(last.get('loss'), 0):.4f}\n")
        f.write("\n")

        # Per-round table (all in percent)
        f.write("## Round-by-Round Metrics (Accuracy %)\n\n")
        header = "| Round | Mean (%) | Loss |"
        for i in range(num_clients):
            header += f" Device {i} (%) |"
        header += "\n"
        sep = "|-------|---------|------|"
        for _ in range(num_clients):
            sep += "------------|"
        sep += "\n"
        f.write(header)
        f.write(sep)

        for r in rounds_data:
            row = f"| {r['round']} | {_safe_float(r.get('accuracy_pct'), 0):.2f} | {_safe_float(r.get('loss'), 0):.4f} |"
            for i in range(num_clients):
                pct = r.get("client_accuracies_pct") or []
                val = _safe_float(pct[i] if i < len(pct) else 0, 0)
                row += f" {val:.2f} |"
            row += "\n"
            f.write(row)

        f.write("\n## Detailed Metrics (Last Round)\n\n")
        last = rounds_data[-1]
        f.write(f"- **Accuracy**: {_safe_float(last.get('accuracy'), 0):.4f} ({_safe_float(last.get('accuracy_pct'), 0):.2f}%)\n")
        f.write(f"- **Loss**: {_safe_float(last.get('loss'), 0):.4f}\n")
        f.write(f"- **Precision**: {_safe_float(last.get('precision'), 0):.4f} ({_safe_float(last.get('precision_pct'), 0):.2f}%)\n")
        f.write(f"- **Recall**: {_safe_float(last.get('recall'), 0):.4f} ({_safe_float(last.get('recall_pct'), 0):.2f}%)\n")
        f.write(f"- **F1-Score**: {_safe_float(last.get('f1_score'), 0):.4f} ({_safe_float(last.get('f1_pct'), 0):.2f}%)\n")
        if last.get("total_samples"):
            f.write(f"- **Total samples (eval)**: {last['total_samples']}\n")
        f.write("\n")

        f.write("## Finding Problem Rounds / Devices\n\n")
        f.write("- **CSV**: `outputs/fl_evaluation_history.csv` — sort by `accuracy_pct` or `client_*_accuracy_pct` to see drops.\n")
        f.write("- **Graph**: `outputs/fl_evaluation_plot.png` — which round or which device line drops.\n")
        f.write("- **JSON**: `outputs/fl_evaluation_history.json` — same data for scripts.\n")

    print(f"Saved: {out_path}")
    return out_path


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate FL evaluation Markdown report from JSON history.")
    p.add_argument("--input", type=Path, default=Path("outputs/fl_evaluation_history.json"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--output", type=str, default="fl_evaluation_report.md")
    args = p.parse_args()
    if not args.input.exists():
        print(f"Not found: {args.input}. Run FL training first.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_fl_report(args.input, args.output_dir, args.output)


if __name__ == "__main__":
    main()
