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
            acc_pct = last.get('accuracy_pct')
            loss_val = last.get('loss')
            if acc_pct is not None:
                f.write(f"- **Final mean accuracy**: {acc_pct:.2f}%\n")
            if loss_val is not None:
                f.write(f"- **Final loss**: {loss_val:.4f}\n")
        f.write("\n")

        # Per-round table (all in percent)
        f.write("## Round-by-Round Metrics\n\n")
        header = "| Round | Mean Acc (%) | Loss |"
        for i in range(num_clients):
            header += f" Device {i} (%) |"
        header += "\n"
        sep = "|-------|-------------|------|"
        for _ in range(num_clients):
            sep += "------------|"
        sep += "\n"
        f.write(header)
        f.write(sep)

        for r in rounds_data:
            acc_pct = r.get('accuracy_pct')
            loss_val = r.get('loss')
            row = f"| {r['round']} | "
            row += f"{acc_pct:.2f}" if acc_pct is not None else "N/A"
            row += " | "
            row += f"{loss_val:.4f}" if loss_val is not None else "N/A"
            row += " |"
            for i in range(num_clients):
                pct = r.get("client_accuracies_pct") or []
                val = pct[i] if i < len(pct) else None
                if val is not None:
                    row += f" {val:.2f} |"
                else:
                    row += " N/A |"
            row += "\n"
            f.write(row)

        f.write("\n## Detailed Metrics (Last Round)\n\n")
        last = rounds_data[-1]
        acc = last.get('accuracy')
        acc_pct = last.get('accuracy_pct')
        loss_val = last.get('loss')
        prec = last.get('precision')
        prec_pct = last.get('precision_pct')
        rec = last.get('recall')
        rec_pct = last.get('recall_pct')
        f1 = last.get('f1_score')
        f1_pct = last.get('f1_pct')
        
        if acc is not None:
            f.write(f"- **Accuracy**: {acc:.4f}" + (f" ({acc_pct:.2f}%)" if acc_pct is not None else "") + "\n")
        if loss_val is not None:
            f.write(f"- **Loss**: {loss_val:.4f}\n")
        if prec is not None:
            f.write(f"- **Precision**: {prec:.4f}" + (f" ({prec_pct:.2f}%)" if prec_pct is not None else "") + "\n")
        if rec is not None:
            f.write(f"- **Recall**: {rec:.4f}" + (f" ({rec_pct:.2f}%)" if rec_pct is not None else "") + "\n")
        if f1 is not None:
            f.write(f"- **F1-Score**: {f1:.4f}" + (f" ({f1_pct:.2f}%)" if f1_pct is not None else "") + "\n")
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
