#!/usr/bin/env python3
"""
Compression Analysis Script

Measures model size, accuracy, and inference speed at each compression stage.
Generates reports in CSV/JSON format and visualizations.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to Python path (for Colab compatibility)
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data.loader import load_dataset
from src.models import nets


class CompressionAnalyzer:
    """Analyze model compression stages: size, accuracy, and inference speed."""

    def __init__(
        self,
        config_path: str = "config/federated_local.yaml",
        output_dir: str = "data/processed/analysis",
    ):
        """Initialize analyzer with config and output directory."""
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict] = []

        # Load config
        import yaml
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Load test dataset
        data_cfg = self.config.get("data", {})
        dataset_name = data_cfg.get("name", "bot_iot")
        dataset_kwargs = {
            k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}
        }

        print(f"Loading dataset: {dataset_name}")
        _, _, self.x_test, self.y_test = load_dataset(dataset_name, **dataset_kwargs)
        print(f"Test set size: {len(self.x_test)} samples")

        # Get model config
        model_cfg = self.config.get("model", {})
        self.model_name = model_cfg.get("name", "mlp")
        self.input_shape = (self.x_test.shape[1],) if self.x_test.ndim == 2 else self.x_test.shape[1:]
        self.num_classes = len(np.unique(self.y_test))

    def measure_model_size(self, model_path: str) -> Dict[str, float]:
        """Measure model file size and calculate compression metrics."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        file_size_bytes = path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        # Load model to count parameters
        if model_path.endswith(".h5"):
            model = tf.keras.models.load_model(model_path)
            param_count = model.count_params()
        elif model_path.endswith(".tflite"):
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            # get_tensor_details() returns a list of dicts, each with 'shape' key
            param_count = sum(
                np.prod(tensor['shape']) for tensor in interpreter.get_tensor_details()
                if 'shape' in tensor and tensor['shape'] is not None
            )
        else:
            param_count = 0

        return {
            "file_size_bytes": file_size_bytes,
            "file_size_mb": file_size_mb,
            "parameter_count": param_count,
        }

    def evaluate_model(
        self, model_path: str, stage_name: str
    ) -> Dict[str, float]:
        """Evaluate model accuracy and other metrics on test set."""
        print(f"\nEvaluating {stage_name}...")

        # Load model
        if model_path.endswith(".h5"):
            model = tf.keras.models.load_model(model_path)
            y_pred_proba = model.predict(self.x_test, verbose=0)

            # Convert probabilities to predictions
            if self.num_classes <= 2:
                if y_pred_proba.ndim > 1:
                    y_pred_proba = y_pred_proba[:, 0]
                y_pred = (y_pred_proba >= 0.5).astype(int)
            else:
                y_pred = np.argmax(y_pred_proba, axis=1)

        elif model_path.endswith(".tflite"):
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # TFLite models typically support batch size 1 only
            # Process samples one at a time
            y_pred_list = []
            batch_size = 1

            for i in range(0, len(self.x_test), batch_size):
                batch_x = self.x_test[i : i + batch_size]
                interpreter.set_tensor(input_details[0]["index"], batch_x.astype(np.float32))
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]["index"])
                y_pred_list.append(output)

            y_pred_proba = np.concatenate(y_pred_list, axis=0)

            # Convert probabilities to predictions
            if self.num_classes <= 2:
                if y_pred_proba.ndim > 1:
                    y_pred_proba = y_pred_proba[:, 0]
                y_pred = (y_pred_proba >= 0.5).astype(int)
            else:
                y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average="binary" if self.num_classes <= 2 else "weighted", zero_division=0)
        recall = recall_score(self.y_test, y_pred, average="binary" if self.num_classes <= 2 else "weighted", zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average="binary" if self.num_classes <= 2 else "weighted", zero_division=0)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

    def measure_inference_speed(
        self, model_path: str, num_runs: int = 10
    ) -> Dict[str, float]:
        """Measure inference speed (latency) in milliseconds."""
        print(f"Measuring inference speed ({num_runs} runs)...")

        # Load model
        if model_path.endswith(".h5"):
            model = tf.keras.models.load_model(model_path)
            times = []
            for _ in range(num_runs):
                start = time.time()
                _ = model.predict(self.x_test[:100], verbose=0)  # Use subset for speed
                times.append((time.time() - start) * 1000)  # Convert to ms

        elif model_path.endswith(".tflite"):
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            times = []
            test_subset = self.x_test[:100]
            for _ in range(num_runs):
                start = time.time()
                for i in range(len(test_subset)):
                    interpreter.set_tensor(
                        input_details[0]["index"], test_subset[i : i + 1].astype(np.float32)
                    )
                    interpreter.invoke()
                    _ = interpreter.get_tensor(output_details[0]["index"])
                times.append((time.time() - start) * 1000)  # Convert to ms
        else:
            return {"avg_latency_ms": 0.0, "min_latency_ms": 0.0, "max_latency_ms": 0.0}

        avg_latency = np.mean(times)
        min_latency = np.min(times)
        max_latency = np.max(times)

        return {
            "avg_latency_ms": float(avg_latency),
            "min_latency_ms": float(min_latency),
            "max_latency_ms": float(max_latency),
            "samples_per_second": float(100 / avg_latency * 1000),  # Approximate
        }

    def analyze_stage(
        self,
        model_path: str,
        stage_name: str,
        baseline_path: Optional[str] = None,
    ) -> Dict:
        """Analyze a single compression stage."""
        print(f"\n{'='*60}")
        print(f"Analyzing stage: {stage_name}")
        print(f"Model: {model_path}")
        print(f"{'='*60}")

        # Measure size
        size_metrics = self.measure_model_size(model_path)
        print(f"File size: {size_metrics['file_size_mb']:.4f} MB")
        print(f"Parameters: {size_metrics['parameter_count']:,}")

        # Evaluate accuracy
        accuracy_metrics = self.evaluate_model(model_path, stage_name)
        print(f"Accuracy: {accuracy_metrics['accuracy']:.4f}")
        print(f"F1-Score: {accuracy_metrics['f1_score']:.4f}")

        # Measure inference speed
        speed_metrics = self.measure_inference_speed(model_path)

        # Calculate compression ratio if baseline provided
        compression_ratio = None
        if baseline_path and Path(baseline_path).exists():
            baseline_size = self.measure_model_size(baseline_path)
            compression_ratio = baseline_size["file_size_mb"] / size_metrics["file_size_mb"]
            print(f"Compression ratio: {compression_ratio:.2f}x")

        # Combine all metrics
        result = {
            "stage": stage_name,
            "model_path": model_path,
            **size_metrics,
            **accuracy_metrics,
            **speed_metrics,
        }

        if compression_ratio:
            result["compression_ratio"] = compression_ratio

        self.results.append(result)
        return result

    def save_results(self, format: str = "all"):
        """Save analysis results to CSV, JSON, and/or Markdown."""
        if not self.results:
            print("No results to save.")
            return

        df = pd.DataFrame(self.results)

        if format in ["csv", "all"]:
            csv_path = self.output_dir / "compression_analysis.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n✅ Saved CSV: {csv_path}")

        if format in ["json", "all"]:
            json_path = self.output_dir / "compression_analysis.json"
            with open(json_path, "w") as f:
                json.dump(self.results, f, indent=2)
            print(f"✅ Saved JSON: {json_path}")

        if format in ["markdown", "all"]:
            md_path = self.output_dir / "compression_analysis.md"
            self._generate_markdown_report(df, md_path)
            print(f"✅ Saved Markdown: {md_path}")

    def _generate_markdown_report(self, df: pd.DataFrame, output_path: Path):
        """Generate markdown report with comparison tables."""
        with open(output_path, "w") as f:
            f.write("# Compression Analysis Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")

            f.write("## Summary\n\n")
            f.write(f"Total stages analyzed: {len(df)}\n\n")

            # Comparison table
            f.write("## Comparison Table\n\n")
            f.write("| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |\n")
            f.write("|-------|-----------|------------|----------|----------|--------------|\n")

            for _, row in df.iterrows():
                f.write(
                    f"| {row['stage']} | {row['file_size_mb']:.4f} | "
                    f"{row['parameter_count']:,} | {row['accuracy']:.4f} | "
                    f"{row['f1_score']:.4f} | {row['avg_latency_ms']:.2f} |\n"
                )

            # Compression ratios if available
            if "compression_ratio" in df.columns:
                f.write("\n## Compression Ratios\n\n")
                f.write("| Stage | Compression Ratio |\n")
                f.write("|-------|------------------|\n")
                for _, row in df.iterrows():
                    if pd.notna(row.get("compression_ratio")):
                        f.write(f"| {row['stage']} | {row['compression_ratio']:.2f}x |\n")

            # Detailed metrics
            f.write("\n## Detailed Metrics\n\n")
            for _, row in df.iterrows():
                f.write(f"### {row['stage']}\n\n")
                f.write(f"- **Model Path**: `{row['model_path']}`\n")
                f.write(f"- **File Size**: {row['file_size_mb']:.4f} MB ({row['file_size_bytes']:,} bytes)\n")
                f.write(f"- **Parameters**: {row['parameter_count']:,}\n")
                f.write(f"- **Accuracy**: {row['accuracy']:.4f}\n")
                f.write(f"- **Precision**: {row['precision']:.4f}\n")
                f.write(f"- **Recall**: {row['recall']:.4f}\n")
                f.write(f"- **F1-Score**: {row['f1_score']:.4f}\n")
                f.write(f"- **Avg Latency**: {row['avg_latency_ms']:.2f} ms\n")
                f.write(f"- **Samples/sec**: {row.get('samples_per_second', 0):.2f}\n")
                if "compression_ratio" in row and pd.notna(row.get("compression_ratio")):
                    f.write(f"- **Compression Ratio**: {row['compression_ratio']:.2f}x\n")
                f.write("\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze model compression stages"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/federated_local.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Model file paths to analyze (format: stage_name:path)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline model path for compression ratio calculation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/analysis",
        help="Output directory for results",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json", "markdown", "all"],
        default="all",
        help="Output format",
    )

    args = parser.parse_args()

    analyzer = CompressionAnalyzer(
        config_path=args.config, output_dir=args.output_dir
    )

    # Parse model paths
    baseline_path = args.baseline
    if baseline_path and not Path(baseline_path).exists():
        print(f"Warning: Baseline model not found: {baseline_path}")
        baseline_path = None

    for model_spec in args.models:
        if ":" in model_spec:
            stage_name, model_path = model_spec.split(":", 1)
        else:
            stage_name = Path(model_spec).stem
            model_path = model_spec

        if not Path(model_path).exists():
            print(f"Warning: Model not found: {model_path}, skipping...")
            continue

        # Use first model as baseline if not specified
        if baseline_path is None and len(analyzer.results) == 0:
            baseline_path = model_path

        analyzer.analyze_stage(
            model_path=model_path,
            stage_name=stage_name,
            baseline_path=baseline_path,
        )

    analyzer.save_results(format=args.format)
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()

