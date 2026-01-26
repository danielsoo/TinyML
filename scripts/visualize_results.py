#!/usr/bin/env python3
"""
Visualization Script for Compression Analysis

Creates size vs accuracy trade-off visualizations and other analysis plots.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


class CompressionVisualizer:
    """Visualize compression analysis results."""

    def __init__(self, results_path: str, output_dir: str = "data/processed/analysis"):
        """Initialize visualizer with results path."""
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        if self.results_path.suffix == ".json":
            with open(self.results_path) as f:
                self.results = json.load(f)
            self.df = pd.DataFrame(self.results)
        else:
            self.df = pd.read_csv(self.results_path)
            self.results = self.df.to_dict("records")

    def plot_size_vs_accuracy(self, save_path: Optional[str] = None):
        """Create size vs accuracy trade-off plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot
        scatter = ax.scatter(
            self.df["file_size_mb"],
            self.df["accuracy"],
            s=200,
            alpha=0.6,
            c=range(len(self.df)),
            cmap="viridis",
            edgecolors="black",
            linewidths=2,
        )

        # Add labels for each point
        for idx, row in self.df.iterrows():
            ax.annotate(
                row["stage"],
                (row["file_size_mb"], row["accuracy"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

        # Trend line
        if len(self.df) > 1:
            z = np.polyfit(self.df["file_size_mb"], self.df["accuracy"], 1)
            p = np.poly1d(z)
            ax.plot(
                self.df["file_size_mb"],
                p(self.df["file_size_mb"]),
                "r--",
                alpha=0.5,
                label="Trend",
            )

        ax.set_xlabel("Model Size (MB)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
        ax.set_title("Size vs Accuracy Trade-off", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Saved plot: {save_path}")
        else:
            save_path = self.output_dir / "size_vs_accuracy.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Saved plot: {save_path}")

        plt.close()

    def plot_compression_metrics(self, save_path: Optional[str] = None):
        """Create comprehensive metrics comparison plot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Size comparison
        axes[0, 0].bar(self.df["stage"], self.df["file_size_mb"], color="steelblue")
        axes[0, 0].set_ylabel("Size (MB)", fontweight="bold")
        axes[0, 0].set_title("Model Size by Stage", fontweight="bold")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. Accuracy comparison
        axes[0, 1].bar(self.df["stage"], self.df["accuracy"], color="forestgreen")
        axes[0, 1].set_ylabel("Accuracy", fontweight="bold")
        axes[0, 1].set_title("Accuracy by Stage", fontweight="bold")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].set_ylim([0, 1])

        # 3. F1-Score comparison
        axes[1, 0].bar(self.df["stage"], self.df["f1_score"], color="coral")
        axes[1, 0].set_ylabel("F1-Score", fontweight="bold")
        axes[1, 0].set_title("F1-Score by Stage", fontweight="bold")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].set_ylim([0, 1])

        # 4. Latency comparison
        axes[1, 1].bar(self.df["stage"], self.df["avg_latency_ms"], color="purple")
        axes[1, 1].set_ylabel("Latency (ms)", fontweight="bold")
        axes[1, 1].set_title("Inference Latency by Stage", fontweight="bold")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Saved plot: {save_path}")
        else:
            save_path = self.output_dir / "compression_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Saved plot: {save_path}")

        plt.close()

    def plot_compression_ratio(self, save_path: Optional[str] = None):
        """Plot compression ratios if available."""
        if "compression_ratio" not in self.df.columns:
            print("No compression ratio data available.")
            return

        df_with_ratio = self.df[self.df["compression_ratio"].notna()]

        if len(df_with_ratio) == 0:
            print("No compression ratio data available.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(
            df_with_ratio["stage"],
            df_with_ratio["compression_ratio"],
            color="teal",
            edgecolor="black",
            linewidth=1.5,
        )

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}x",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_ylabel("Compression Ratio", fontsize=12, fontweight="bold")
        ax.set_title("Compression Ratio by Stage", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Saved plot: {save_path}")
        else:
            save_path = self.output_dir / "compression_ratio.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Saved plot: {save_path}")

        plt.close()

    def plot_all(self):
        """Generate all visualizations."""
        print("Generating visualizations...")
        self.plot_size_vs_accuracy()
        self.plot_compression_metrics()
        self.plot_compression_ratio()
        print("\n✅ All visualizations generated!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize compression analysis results"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=False,
        default="data/processed/analysis/compression_analysis.json",
        help="Path to results file (CSV or JSON). Default: data/processed/analysis/compression_analysis.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/analysis",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--plot",
        type=str,
        choices=["size-accuracy", "metrics", "compression-ratio", "all"],
        default="all",
        help="Which plot to generate",
    )

    args = parser.parse_args()

    visualizer = CompressionVisualizer(
        results_path=args.results, output_dir=args.output_dir
    )

    if args.plot == "all":
        visualizer.plot_all()
    elif args.plot == "size-accuracy":
        visualizer.plot_size_vs_accuracy()
    elif args.plot == "metrics":
        visualizer.plot_compression_metrics()
    elif args.plot == "compression-ratio":
        visualizer.plot_compression_ratio()


if __name__ == "__main__":
    from typing import Optional
    import numpy as np
    main()

