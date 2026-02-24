#!/usr/bin/env python3
"""FL history analysis: round-by-round and per-device accuracy plot.

Reads outputs/fl_evaluation_history.json and plots:
- Mean accuracy per round
- Per-device (client) accuracy
as time series, saves to outputs/fl_evaluation_plot.png.
For reports/presentations: identify which round/device has issues.
"""
from pathlib import Path
import argparse
import json


def load_history(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_fl_history(
    json_path: Path,
    output_dir: Path,
    save_name: str = "fl_evaluation_plot.png",
) -> Path:
    data = load_history(json_path)
    rounds_data = data.get("rounds", [])
    num_clients = int(data.get("num_clients", 0))

    if not rounds_data:
        raise ValueError("No rounds in history. Run FL training first.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")

    rounds = [r["round"] for r in rounds_data]
    mean_acc_pct = [r.get("accuracy_pct") for r in rounds_data]
    
    # Filter out None values for plotting
    valid_indices = [i for i, val in enumerate(mean_acc_pct) if val is not None]
    if not valid_indices:
        print("No accuracy data available for plotting")
        return output_dir / save_name
    
    valid_rounds = [rounds[i] for i in valid_indices]
    valid_mean_acc_pct = [mean_acc_pct[i] for i in valid_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(valid_rounds, valid_mean_acc_pct, "b-o", linewidth=2, markersize=8, label="Mean accuracy (%)")

    for i in range(num_clients):
        key_pct = "client_accuracies_pct"
        if key_pct in rounds_data[0] and rounds_data[0][key_pct] and i < len(rounds_data[0][key_pct]):
            device_pct = [r.get(key_pct, [])[i] if r.get(key_pct) and i < len(r.get(key_pct, [])) else None for r in rounds_data]
            # Filter out None values
            device_valid_indices = [j for j, val in enumerate(device_pct) if val is not None]
            if device_valid_indices:
                device_valid_rounds = [rounds[j] for j in device_valid_indices]
                device_valid_pct = [device_pct[j] for j in device_valid_indices]
                plt.plot(device_valid_rounds, device_valid_pct, "-o", linewidth=1.5, markersize=5, label=f"Device {i} (%)")

    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("FL Evaluation: Accuracy per Round (Mean + Per-Device)", fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / save_name
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot FL evaluation history (round + per-device accuracy).")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/fl_evaluation_history.json"),
        help="Path to fl_evaluation_history.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save plot",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fl_evaluation_plot.png",
        help="Output filename",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Not found: {args.input}. Run FL training first (e.g. python scripts/train.py).")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_fl_history(args.input, args.output_dir, args.output)


if __name__ == "__main__":
    main()
