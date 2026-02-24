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
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Suppress unnecessary warnings before importing libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*DtypeWarning.*')

# Add project root to Python path (for Colab compatibility)
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Suppress TensorFlow warnings after import
tf.get_logger().setLevel('ERROR')

from src.data.loader import load_dataset
from src.models import nets


class CompressionAnalyzer:
    """Analyze model compression stages: size, accuracy, and inference speed."""

    def __init__(
        self,
        config_path: str = None,
        output_dir: str = "data/processed/analysis",
        output_dir_final: str = None,
        version: str = None,
        run_id: str = None,
    ):
        """Initialize analyzer with config and output directory."""
        # Use federated_colab.yaml as default if exists, otherwise federated_local.yaml
        if config_path is None:
            if os.path.exists("config/federated_colab.yaml"):
                config_path = "config/federated_colab.yaml"
            else:
                config_path = "config/federated_local.yaml"
        
        self.config_path = config_path
        self.results: List[Dict] = []

        # Load config
        import yaml
        with open(config_path, encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Version: CLI override > config > "run"
        self.version = version or self.config.get("version") or "run"
        # run_id: datetime detail (2026-01-30_14-50-09)
        self.run_id = run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Data version (summary: which data was used for training)
        data_cfg = self.config.get("data", {})
        data_parts = [data_cfg.get("name", "unknown")]
        if data_cfg.get("max_samples"):
            data_parts.append(f"max{data_cfg['max_samples']//1000}k")
        if data_cfg.get("balance_ratio"):
            data_parts.append(f"bal{data_cfg['balance_ratio']}")
        self.data_version = "_".join(data_parts)

        # Output: either direct path (runs/.../analysis) or version/run_id under base
        if output_dir_final:
            self.output_dir = Path(output_dir_final)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            base_dir = Path(output_dir)
            self.output_dir = base_dir / self.version / self.run_id
            self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📌 Version: {self.version} | Run: {self.run_id} | Data: {self.data_version}")

        # Load test dataset
        dataset_name = data_cfg.get("name", "bot_iot")
        dataset_kwargs = {
            k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}
        }

        print(f"\n{'='*60}")
        print(f"📊 COMPRESSION ANALYSIS")
        print(f"{'='*60}")
        print(f"\n📁 Loading dataset: {dataset_name}")
        print(f"📋 Dataset config: {dataset_kwargs}")
        _, _, self.x_test, self.y_test = load_dataset(dataset_name, **dataset_kwargs)
        print(f"✅ Test set loaded: {len(self.x_test):,} samples")

        # Prediction threshold for binary (prob >= threshold → Attack). Same as ratio_sweep/eval.
        self.prediction_threshold = float(self.config.get("evaluation", {}).get("prediction_threshold", 0.5))

        # Get model config
        model_cfg = self.config.get("model", {})
        self.model_name = model_cfg.get("name", "mlp")
        self.input_shape = (self.x_test.shape[1],) if self.x_test.ndim == 2 else self.x_test.shape[1:]
        self.num_classes = len(np.unique(self.y_test))

    def _load_keras_h5(self, model_path: str):
        """Load Keras .h5 model; use quantize_scope when config has use_qat (QAT models)."""
        use_qat = self.config.get("federated", {}).get("use_qat", False)
        if use_qat:
            try:
                import tensorflow_model_optimization as tfmot
                with tfmot.quantization.keras.quantize_scope():
                    return tf.keras.models.load_model(model_path, compile=False)
            except Exception:
                pass
        return tf.keras.models.load_model(model_path, compile=False)

    def measure_model_size(self, model_path: str) -> Dict[str, float]:
        """Measure model file size and calculate compression metrics."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        file_size_bytes = path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        # Load model to count parameters (compile=False for custom loss compatibility)
        if model_path.endswith(".h5"):
            model = self._load_keras_h5(model_path)
            param_count = model.count_params()
        elif model_path.endswith(".tflite"):
            interpreter = tf.lite.Interpreter(
                model_path=model_path,
                experimental_preserve_all_tensors=True,
            )
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
        # Suppress output during evaluation
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

        # Load model (compile=False for custom loss e.g. focal loss)
        if model_path.endswith(".h5"):
            model = self._load_keras_h5(model_path)

            # Check input shape compatibility
            model_input_shape = model.input_shape
            data_feature_count = self.x_test.shape[1]
            expected_feature_count = model_input_shape[1] if model_input_shape and len(model_input_shape) > 1 else None
            
            if expected_feature_count and expected_feature_count != data_feature_count:
                raise ValueError(
                    f"Model input shape mismatch: "
                    f"Model expects {expected_feature_count} features, "
                    f"but data has {data_feature_count} features. "
                    f"Please retrain the model with the updated data loader that includes "
                    f"IP addresses and categorical features, or use a model trained with "
                    f"the same feature set as the current data."
                )
            
            y_pred_proba = model.predict(self.x_test, verbose=0)

            # Convert probabilities to predictions (use config threshold)
            if self.num_classes <= 2:
                if y_pred_proba.ndim > 1:
                    y_pred_proba = y_pred_proba[:, 0]
                y_pred = (y_pred_proba >= self.prediction_threshold).astype(int)
            else:
                y_pred = np.argmax(y_pred_proba, axis=1)

        elif model_path.endswith(".tflite"):
            interpreter = tf.lite.Interpreter(
                model_path=model_path,
                experimental_preserve_all_tensors=True,
            )
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Check input shape compatibility
            tflite_input_shape = input_details[0]["shape"]
            data_feature_count = self.x_test.shape[1]
            expected_feature_count = tflite_input_shape[1] if len(tflite_input_shape) > 1 else None
            
            if expected_feature_count and expected_feature_count != data_feature_count:
                raise ValueError(
                    f"TFLite model input shape mismatch: "
                    f"Model expects {expected_feature_count} features, "
                    f"but data has {data_feature_count} features. "
                    f"Please retrain the model with the updated data loader that includes "
                    f"IP addresses and categorical features, or use a model trained with "
                    f"the same feature set as the current data."
                )

            # TFLite models typically support batch size 1 only
            # Process samples one at a time
            y_pred_list = []
            batch_size = 1

            # Check input type (FLOAT32 or INT8)
            input_dtype = input_details[0]["dtype"]
            
            for i in range(0, len(self.x_test), batch_size):
                batch_x = self.x_test[i : i + batch_size]
                
                # Convert to appropriate type
                if input_dtype == np.int8:
                    # For INT8 quantized models, apply quantization
                    input_scale = input_details[0].get("quantization_parameters", {}).get("scales", [1.0])[0]
                    input_zero_point = input_details[0].get("quantization_parameters", {}).get("zero_points", [0])[0]
                    batch_x_quantized = (batch_x / input_scale + input_zero_point).astype(np.int8)
                    interpreter.set_tensor(input_details[0]["index"], batch_x_quantized)
                else:
                    # For FLOAT32 models
                    interpreter.set_tensor(input_details[0]["index"], batch_x.astype(np.float32))
                
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]["index"])
                
                # Dequantize output if needed
                output_dtype = output_details[0]["dtype"]
                if output_dtype == np.int8:
                    output_scale = output_details[0].get("quantization_parameters", {}).get("scales", [1.0])[0]
                    output_zero_point = output_details[0].get("quantization_parameters", {}).get("zero_points", [0])[0]
                    output = output_scale * (output.astype(np.float32) - output_zero_point)
                
                y_pred_list.append(output)

            y_pred_proba = np.concatenate(y_pred_list, axis=0)

            # Convert probabilities to predictions (use config threshold)
            if self.num_classes <= 2:
                if y_pred_proba.ndim > 1:
                    y_pred_proba = y_pred_proba[:, 0]
                y_pred = (y_pred_proba >= self.prediction_threshold).astype(int)
            else:
                y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average="binary" if self.num_classes <= 2 else "weighted", zero_division=0)
        recall = recall_score(self.y_test, y_pred, average="binary" if self.num_classes <= 2 else "weighted", zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average="binary" if self.num_classes <= 2 else "weighted", zero_division=0)

        out = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }
        # Normal class (0) metrics: Recall = of actual normal, predicted normal; Precision = of predicted normal, actual normal
        if self.num_classes <= 2:
            prec_per = precision_score(self.y_test, y_pred, average=None, zero_division=0)
            rec_per = recall_score(self.y_test, y_pred, average=None, zero_division=0)
            out["precision_normal"] = float(prec_per[0])  # of predicted normal, fraction actually normal
            out["recall_normal"] = float(rec_per[0])       # of actual normal, fraction predicted normal
        return out

    def measure_inference_speed(
        self, model_path: str, num_runs: int = 10
    ) -> Dict[str, float]:
        """Measure inference speed (latency) in milliseconds."""
        # Suppress output during speed measurement
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

        # Load model (compile=False for custom loss compatibility)
        if model_path.endswith(".h5"):
            model = self._load_keras_h5(model_path)
            times = []
            for _ in range(num_runs):
                start = time.time()
                _ = model.predict(self.x_test[:100], verbose=0)  # Use subset for speed
                times.append((time.time() - start) * 1000)  # Convert to ms

        elif model_path.endswith(".tflite"):
            interpreter = tf.lite.Interpreter(
                model_path=model_path,
                experimental_preserve_all_tensors=True,
            )
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Check input type (FLOAT32 or INT8)
            input_dtype = input_details[0]["dtype"]

            times = []
            test_subset = self.x_test[:100]
            for _ in range(num_runs):
                start = time.time()
                for i in range(len(test_subset)):
                    batch_x = test_subset[i : i + 1]
                    
                    # Convert to appropriate type
                    if input_dtype == np.int8:
                        # For INT8 quantized models, apply quantization
                        input_scale = input_details[0].get("quantization_parameters", {}).get("scales", [1.0])[0]
                        input_zero_point = input_details[0].get("quantization_parameters", {}).get("zero_points", [0])[0]
                        batch_x_quantized = (batch_x / input_scale + input_zero_point).astype(np.int8)
                        interpreter.set_tensor(input_details[0]["index"], batch_x_quantized)
                    else:
                        # For FLOAT32 models
                        interpreter.set_tensor(input_details[0]["index"], batch_x.astype(np.float32))
                    
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
        print(f"🔍 Analyzing: {stage_name}")
        print(f"📦 Model: {model_path}")
        print(f"{'='*60}")

        # Measure size
        size_metrics = self.measure_model_size(model_path)
        print(f"\n📏 Model Size:")
        print(f"   • File size: {size_metrics['file_size_mb']:.4f} MB")
        print(f"   • Parameters: {size_metrics['parameter_count']:,}")

        # Evaluate accuracy
        accuracy_metrics = self.evaluate_model(model_path, stage_name)
        print(f"\n🎯 Performance:")
        print(f"   • Accuracy: {accuracy_metrics['accuracy']:.4f} ({accuracy_metrics['accuracy']*100:.2f}%)")
        print(f"   • F1-Score: {accuracy_metrics['f1_score']:.4f} ({accuracy_metrics['f1_score']*100:.2f}%)")
        if "recall_normal" in accuracy_metrics and "precision_normal" in accuracy_metrics:
            print(f"   • Normal Recall (of actual normal, % predicted as normal): {accuracy_metrics['recall_normal']:.4f} ({accuracy_metrics['recall_normal']*100:.2f}%)")
            print(f"   • Normal Precision (of predicted normal, % actually normal): {accuracy_metrics['precision_normal']:.4f} ({accuracy_metrics['precision_normal']*100:.2f}%)")

        # Measure inference speed
        speed_metrics = self.measure_inference_speed(model_path)
        print(f"\n⚡ Inference Speed:")
        print(f"   • Avg latency: {speed_metrics['avg_latency_ms']:.2f} ms")
        print(f"   • Min latency: {speed_metrics['min_latency_ms']:.2f} ms")
        print(f"   • Max latency: {speed_metrics['max_latency_ms']:.2f} ms")

        # Calculate compression ratio if baseline provided
        compression_ratio = None
        if baseline_path and Path(baseline_path).exists():
            baseline_size = self.measure_model_size(baseline_path)
            compression_ratio = baseline_size["file_size_mb"] / size_metrics["file_size_mb"]
            print(f"\n📊 Compression:")
            print(f"   • Ratio: {compression_ratio:.2f}x")
            print(f"   • Size reduction: {(1 - 1/compression_ratio)*100:.1f}%")

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

    def _convert_to_serializable(self, obj):
        """Convert NumPy types to Python native types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj

    def save_results(self, format: str = "all"):
        """Save analysis results to CSV, JSON, and/or Markdown."""
        if not self.results:
            print("No results to save.")
            return

        df = pd.DataFrame(self.results)

        print(f"\n{'='*60}")
        print(f"💾 Saving Results")
        print(f"{'='*60}")
        
        if format in ["csv", "all"]:
            csv_path = self.output_dir / "compression_analysis.csv"
            df.to_csv(csv_path, index=False)
            print(f"✅ CSV: {csv_path}")

        if format in ["json", "all"]:
            json_path = self.output_dir / "compression_analysis.json"
            # Convert NumPy types to Python native types for JSON serialization
            serializable_results = self._convert_to_serializable(self.results)
            output_data = {
                "version": self.version,
                "data_version": self.data_version,
                "generated_at": datetime.now().isoformat(),
                "results": serializable_results,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"✅ JSON: {json_path}")

        if format in ["markdown", "all"]:
            md_path = self.output_dir / "compression_analysis.md"
            self._generate_markdown_report(df, md_path)
            print(f"✅ Markdown: {md_path}")

        # Version summary (for version comparison)
        self._append_version_summary(df)

        # Path for next step (viz etc): version/run_id
        rel_path = f"{self.version}/{self.run_id}"
        # When writing under runs/.../analysis, .last_run_id lives in runs/; else in analysis base
        if "runs" in str(self.output_dir):
            last_run_dir = self.output_dir.parent.parent.parent
        else:
            last_run_dir = self.output_dir.parent.parent
        (last_run_dir / ".last_run_id").write_text(rel_path, encoding="utf-8")

    def _append_version_summary(self, df: pd.DataFrame):
        """Append one-line summary to VERSIONS.md for cross-version comparison."""
        if "runs" in str(self.output_dir):
            summary_dir = self.output_dir.parent.parent.parent
        else:
            summary_dir = self.output_dir.parent.parent
        summary_path = summary_dir / "VERSIONS.md"
        best_acc = df["accuracy"].max()
        best_f1 = df["f1_score"].max()
        orig_row = df.iloc[0]
        comp_mask = df["stage"].str.lower().str.contains("compress", na=False)
        comp_row = df[comp_mask].iloc[0] if comp_mask.any() else None
        size_orig = orig_row["file_size_mb"]
        size_comp = comp_row["file_size_mb"] if comp_row is not None else None
        ratio = f"{size_orig/size_comp:.1f}x" if size_comp is not None and size_comp > 0 else "-"

        header = "| Version | Run (datetime) | Data | Best Acc | Best F1 | Orig (MB) | Comp (MB) | Ratio |\n"
        header += "|---------|----------------|------|----------|---------|-----------|-----------|-------|\n"
        line = f"| {self.version} | {self.run_id} | {self.data_version} | {best_acc:.4f} | {best_f1:.4f} | {size_orig:.3f} | {size_comp or 0:.3f} | {ratio} |\n"

        if not summary_path.exists():
            summary_path.write_text("# Version Comparison\n\n" + header + line, encoding="utf-8")
        else:
            content = summary_path.read_text(encoding="utf-8")
            if "| Version |" not in content:
                content = content.rstrip() + "\n\n" + header
            content = content.rstrip() + "\n" + line
            summary_path.write_text(content, encoding="utf-8")
        print(f"✅ Version summary: {summary_path}")

    def _generate_markdown_report(self, df: pd.DataFrame, output_path: Path):
        """Generate markdown report with comparison tables."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Compression Analysis Report\n\n")
            f.write(f"| Item | Value |\n")
            f.write(f"|------|----|\n")
            f.write(f"| **Version** | {self.version} |\n")
            f.write(f"| **Run (datetime)** | {self.run_id} |\n")
            f.write(f"| **Data Version** | {self.data_version} |\n")
            f.write(f"| **Generated** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |\n\n")

            # Run / Training Configuration (for reports and reproducibility)
            f.write("## Run / Training Configuration\n\n")
            cfg = self.config
            data_cfg = cfg.get("data", {})
            fed_cfg = cfg.get("federated", {})
            model_cfg = cfg.get("model", {})
            eval_cfg = cfg.get("evaluation", {})
            comp_cfg = cfg.get("compression", {})
            f.write("| Item | Value |\n|------|-------|\n")
            f.write(f"| **Data** | {data_cfg.get('name', '-')} |\n")
            f.write(f"| **Data path** | {data_cfg.get('path', '-')} |\n")
            f.write(f"| **Max samples** | {data_cfg.get('max_samples', '-')} |\n")
            br = data_cfg.get('balance_ratio')
            br_desc = {1.0: "50:50", 4.0: "normal:attack 8:2", 9.0: "9:1", 19.0: "19:1"}.get(br) if br is not None else None
            br_str = f"{br} ({br_desc})" if br_desc else (str(br) if br is not None else "-")
            f.write(f"| **Balance ratio** (normal:attack) | {br_str} |\n")
            f.write(f"| **Num clients** | {data_cfg.get('num_clients', '-')} |\n")
            f.write(f"| **Binary** | {data_cfg.get('binary', '-')} |\n")
            f.write(f"| **Use SMOTE** | {data_cfg.get('use_smote', '-')} |\n")
            f.write(f"| **Model** | {model_cfg.get('name', '-')} |\n")
            f.write(f"| **FL rounds** | {fed_cfg.get('num_rounds', '-')} |\n")
            f.write(f"| **Local epochs** | {fed_cfg.get('local_epochs', '-')} |\n")
            f.write(f"| **Batch size** | {fed_cfg.get('batch_size', '-')} |\n")
            f.write(f"| **Learning rate** | {fed_cfg.get('learning_rate', '-')} |\n")
            f.write(f"| **Fraction fit** | {fed_cfg.get('fraction_fit', '-')} |\n")
            f.write(f"| **Fraction evaluate** | {fed_cfg.get('fraction_evaluate', '-')} |\n")
            f.write(f"| **Use class weights** | {fed_cfg.get('use_class_weights', '-')} |\n")
            f.write(f"| **Use focal loss** | {fed_cfg.get('use_focal_loss', '-')} |\n")
            f.write(f"| **Focal loss alpha** | {fed_cfg.get('focal_loss_alpha', '-')} |\n")
            f.write(f"| **Use distillation** | {fed_cfg.get('use_distillation', '-')} |\n")
            f.write(f"| **Use QAT** | {fed_cfg.get('use_qat', '-')} |\n")
            f.write(f"| **Server momentum** | {fed_cfg.get('server_momentum', '-')} |\n")
            f.write(f"| **Server learning rate** | {fed_cfg.get('server_learning_rate', '-')} |\n")
            f.write(f"| **LR decay type** | {fed_cfg.get('lr_decay_type', '-')} |\n")
            f.write(f"| **LR decay rate** | {fed_cfg.get('lr_decay_rate', '-')} |\n")
            f.write(f"| **LR drop rate** | {fed_cfg.get('lr_drop_rate', '-')} |\n")
            f.write(f"| **LR epochs drop** | {fed_cfg.get('lr_epochs_drop', '-')} |\n")
            f.write(f"| **LR min** | {fed_cfg.get('lr_min', '-')} |\n")
            f.write(f"| **Min fit clients** | {fed_cfg.get('min_fit_clients', '-')} |\n")
            f.write(f"| **Min evaluate clients** | {fed_cfg.get('min_evaluate_clients', '-')} |\n")
            f.write(f"| **Min available clients** | {fed_cfg.get('min_available_clients', '-')} |\n")
            f.write(f"| **Prediction threshold** | {eval_cfg.get('prediction_threshold', self.prediction_threshold)} |\n")
            rs_models = eval_cfg.get('ratio_sweep_models')
            rs_str = f"{len(rs_models)} models" if isinstance(rs_models, list) else (str(rs_models) if rs_models is not None else "-")
            f.write(f"| **Ratio sweep models** | {rs_str} |\n")
            f.write(f"| **Always build traditional** | {comp_cfg.get('always_build_traditional', '-')} |\n")
            trad_path = comp_cfg.get('traditional_model_path')
            f.write(f"| **Traditional model path** | {trad_path if trad_path is not None else 'null'} |\n")
            f.write("\n")

            f.write("## Summary\n\n")
            f.write(f"Total stages analyzed: {len(df)}\n\n")

            # Find baseline (first row or row with stage name "Baseline")
            baseline_row = None
            if len(df) > 0:
                baseline_candidates = df[df['stage'].str.lower() == 'baseline']
                if len(baseline_candidates) > 0:
                    baseline_row = baseline_candidates.iloc[0]
                else:
                    baseline_row = df.iloc[0]

            # Comparison table (include normal Recall/Precision if present)
            has_normal_metrics = "recall_normal" in df.columns and df["recall_normal"].notna().any()
            if has_normal_metrics:
                f.write("| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Normal Recall | Normal Precision | Latency (ms) |\n")
                f.write("|-------|-----------|------------|----------|----------|---------------|------------------|--------------|\n")
                for _, row in df.iterrows():
                    rec_n = f"{row['recall_normal']:.4f}" if pd.notna(row.get("recall_normal")) else "-"
                    prec_n = f"{row['precision_normal']:.4f}" if pd.notna(row.get("precision_normal")) else "-"
                    f.write(
                        f"| {row['stage']} | {row['file_size_mb']:.4f} | "
                        f"{row['parameter_count']:,} | {row['accuracy']:.4f} | "
                        f"{row['f1_score']:.4f} | {rec_n} | {prec_n} | {row['avg_latency_ms']:.2f} |\n"
                    )
            else:
                f.write("| Stage | Size (MB) | Parameters | Accuracy | F1-Score | Latency (ms) |\n")
                f.write("|-------|-----------|------------|----------|----------|--------------|\n")
                for _, row in df.iterrows():
                    f.write(
                        f"| {row['stage']} | {row['file_size_mb']:.4f} | "
                        f"{row['parameter_count']:,} | {row['accuracy']:.4f} | "
                        f"{row['f1_score']:.4f} | {row['avg_latency_ms']:.2f} |\n"
                    )

            # Improvements vs Baseline section
            if baseline_row is not None and len(df) > 1:
                f.write("\n## 📊 Improvements vs Baseline\n\n")
                f.write(f"**Baseline:** {baseline_row['stage']}\n\n")
                f.write("| Stage | Size Change | Accuracy Change | F1-Score Change | Latency Change | Overall Status |\n")
                f.write("|-------|-------------|-----------------|-----------------|----------------|----------------|\n")
                
                for _, row in df.iterrows():
                    if row['stage'] == baseline_row['stage']:
                        f.write(f"| {row['stage']} | - | - | - | - | **Baseline** |\n")
                        continue
                    
                    # Calculate changes
                    size_change = ((row['file_size_mb'] - baseline_row['file_size_mb']) / baseline_row['file_size_mb']) * 100
                    acc_change = (row['accuracy'] - baseline_row['accuracy']) * 100
                    f1_change = (row['f1_score'] - baseline_row['f1_score']) * 100
                    latency_change = ((row['avg_latency_ms'] - baseline_row['avg_latency_ms']) / baseline_row['avg_latency_ms']) * 100 if baseline_row['avg_latency_ms'] > 0 else 0
                    
                    # Format changes with arrows (↓ = good, ↑ = bad for size/latency, but good for accuracy/f1)
                    size_str = f"{size_change:+.2f}%" + (" ↓" if size_change < 0 else " ↑" if size_change > 0 else " →")
                    acc_str = f"{acc_change:+.4f}%" + (" ↑" if acc_change > 0 else " ↓" if acc_change < 0 else " →")
                    f1_str = f"{f1_change:+.4f}%" + (" ↑" if f1_change > 0 else " ↓" if f1_change < 0 else " →")
                    # Latency: negative change is good (faster), positive is bad (slower)
                    latency_str = f"{latency_change:+.2f}%" + (" ↓" if latency_change < 0 else " ↑" if latency_change > 0 else " →")
                    
                    # Overall status (count improvements)
                    improvements = 0
                    if size_change < 0:  # Smaller is better
                        improvements += 1
                    if acc_change >= 0:  # Higher or same is better
                        improvements += 1
                    if f1_change >= 0:  # Higher or same is better
                        improvements += 1
                    if latency_change < 0:  # Lower latency is better
                        improvements += 1
                    
                    if improvements == 4:
                        status = "✅ All Improved"
                    elif improvements >= 2:
                        status = "✅ Mostly Better"
                    elif improvements == 1:
                        status = "⚠️ Mixed Results"
                    else:
                        status = "❌ Degraded"
                    
                    f.write(f"| {row['stage']} | {size_str} | {acc_str} | {f1_str} | {latency_str} | {status} |\n")

            # Pipeline overview: how each stage is produced
            f.write("\n## Pipeline overview (How each stage is produced)\n\n")
            f.write("| Stage | Input | Processing | Output file |\n")
            f.write("|-------|-------|------------|-------------|\n")
            f.write("| **Keras** | FL/central training done | Use as-is (no QAT strip) or load .h5 | `models/global_model.h5` |\n")
            f.write("### 2×2 Experimental Design\n\n")
            f.write("| Model | Training Method | Compression Pipeline | Filename |\n")
            f.write("|-------|----------------|---------------------|----------|\n")
            f.write("| **Baseline** | QAT-trained | No compression (float32 TFLite) | `saved_model_original.tflite` |\n")
            f.write("| **Traditional + PTQ** | Traditional (no QAT) | 50% prune → **PTQ** → int8 TFLite | `saved_model_no_qat_ptq.tflite` |\n")
            f.write("| **Traditional + QAT** | Traditional (no QAT) | 50% prune → **QAT fine-tune (2 epochs)** → int8 TFLite | `saved_model_traditional_qat.tflite` |\n")
            f.write("| **QAT + PTQ** | QAT-trained | 50% prune → **PTQ** → int8 TFLite | `saved_model_qat_ptq.tflite` |\n")
            f.write("| **QAT + QAT** | QAT-trained | 50% prune → **QAT fine-tune (2 epochs)** → int8 TFLite | `saved_model_pruned_qat.tflite` |\n")

            # Compression ratios if available
            if "compression_ratio" in df.columns:
                f.write("\n## Compression Ratios\n\n")
                f.write("| Stage | Compression Ratio | Size Reduction |\n")
                f.write("|-------|------------------|----------------|\n")
                for _, row in df.iterrows():
                    if pd.notna(row.get("compression_ratio")):
                        reduction = (1 - 1/row['compression_ratio']) * 100
                        f.write(f"| {row['stage']} | {row['compression_ratio']:.2f}x | {reduction:.1f}% |\n")

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
                if "recall_normal" in row and pd.notna(row.get("recall_normal")):
                    f.write(f"- **Normal Recall** (of actual normal, % predicted as normal): {row['recall_normal']:.4f}\n")
                if "precision_normal" in row and pd.notna(row.get("precision_normal")):
                    f.write(f"- **Normal Precision** (of predicted normal, % actually normal): {row['precision_normal']:.4f}\n")
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
        required=False,
        default=None,
        help="Model file paths to analyze (format: stage_name:path). If not specified, analyzes train.py output models automatically.",
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
        help="Base output directory (results go to {output-dir}/{version}/)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Run version (overrides config). Default: from config or timestamp",
    )
    parser.add_argument(
        "--output-dir-final",
        type=str,
        default=None,
        help="Write directly to this path (e.g. data/processed/runs/v11/2026-02-04_12-00-00/analysis). No version/run_id appended.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run datetime (e.g. 2026-02-04_12-00-00). Used with --output-dir-final for .last_run_id.",
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
        config_path=args.config,
        output_dir=args.output_dir,
        output_dir_final=args.output_dir_final,
        version=args.version,
        run_id=args.run_id,
    )

    # Use default model paths if not specified
    if args.models is None:
        print("No models specified. Using default train.py output models...")
        args.models = [
            "Baseline:models/tflite/saved_model_original.tflite",
            "Traditional+PTQ:models/tflite/saved_model_no_qat_ptq.tflite",
            "Traditional+QAT:models/tflite/saved_model_traditional_qat.tflite",
            "QAT+PTQ:models/tflite/saved_model_qat_ptq.tflite",
            "QAT+QAT:models/tflite/saved_model_pruned_qat.tflite",
        ]

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
    print(f"\n{'='*60}")
    print(f"✨ Analysis Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

