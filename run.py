"""
Complete TinyML Pipeline Runner

Executes the full workflow:
1. Train model with Federated Learning (train.py)
2. Compress model (compression.py)
3. Analyze compression results (analyze_compression.py)
4. Ratio sweep (evaluate_ratio_sweep.py) → ratio_sweep_report.md
4b. Threshold tuning (tune_threshold_all_ratios.py) → appended to ratio_sweep_report.md (full sweep + best per ratio)
5. Visualize results (visualize_results.py)

Usage:
    python run.py --config config/federated_scratch.yaml
    python run.py --skip-train   # Skip FL training, use existing model
    python run.py --skip-viz     # Skip visualization
    python run.py --skip-ratio-sweep  # Skip ratio sweep report
    python run.py --model path/to/model.tflite  # Use specified model for ratio sweep + threshold tuning (no train)
    python run.py --skip-train --skip-compression --skip-analysis --model data/processed/runs/v11/2026-02-02_23-28-45/models/tflite/saved_model_original.tflite  # Eval-only; report → runs/.../eval/ratio_sweep_report.md
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime

import yaml
from pathlib import Path


def run_command(command: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"  {description}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ {description} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}\n")
        return False


def get_eval_dir_for_model(model_path: str) -> Path | None:
    """If model is under data/processed/runs/<version>/<datetime>/models/..., return runs/<version>/<datetime>/eval."""
    p = Path(model_path).resolve()
    # p = .../runs/v11/datetime/models/tflite/saved_model.tflite
    if "runs" not in p.parts:
        return None
    run_dir = p.parent.parent.parent  # .../runs/v11/datetime
    if (run_dir / "models").exists():
        return run_dir / "eval"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run complete TinyML pipeline"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip federated learning training (use existing model)"
    )
    parser.add_argument(
        "--skip-compression",
        action="store_true",
        help="Skip compression step"
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip compression analysis"
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization"
    )
    parser.add_argument(
        "--skip-ratio-sweep",
        action="store_true",
        help="Skip ratio sweep (100%%–0%% normal:attack report)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/federated_local.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--centralized",
        action="store_true",
        help="Use centralized training (no FedAvg) instead of FL – for baseline comparison",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model path for evaluation (ratio sweep + threshold tuning). If not set, uses default TFLite/Keras from project root.",
    )

    args = parser.parse_args()
    
    # Track overall success, eval report dir, and current run dir (when we run analysis)
    all_success = True
    eval_report_dir = None
    current_runs_dir = None  # data/processed/runs/<version>/<run_id> when analysis runs

    print(f"\n{'='*80}")
    print(f"  🚀 STARTING TINYML COMPLETE PIPELINE")
    print(f"{'='*80}\n")
    print(f"Configuration: {args.config}")
    print(f"Training mode: {'Centralized (no FedAvg)' if args.centralized else 'Federated Learning'}")
    print(f"Skip train: {args.skip_train}")
    print(f"Skip compression: {args.skip_compression}")
    print(f"Skip analysis: {args.skip_analysis}")
    print(f"Skip ratio sweep: {args.skip_ratio_sweep}")
    print(f"Skip visualization: {args.skip_viz}")
    print(f"Eval model: {args.model or '(default: models/tflite/saved_model_original.tflite or src/models/global_model.h5)'}")
    
    # Step 1: Training (FL or Centralized)
    if not args.skip_train:
        if args.centralized:
            success = run_command(
                [sys.executable, "scripts/train_centralized.py", "--config", args.config],
                "STEP 1: Centralized Training (no FedAvg)"
            )
        else:
            success = run_command(
                [sys.executable, "scripts/train.py", "--config", args.config],
                "STEP 1: Federated Learning Training"
            )
        if not success:
            print("⚠️  Training failed. Stopping pipeline.")
            return 1
        all_success = all_success and success
        # Copy for compression.py which expects models/global_model.h5
        src_model = Path("src/models/global_model.h5")
        dst_model = Path("models/global_model.h5")
        if src_model.exists():
            dst_model.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_model, dst_model)
            print(f"   📋 Copied {src_model} → {dst_model} (for compression)\n")
    else:
        print(f"\n{'='*80}")
        print(f"  ⏭️  SKIPPING STEP 1: Training (using existing model)")
        print(f"{'='*80}\n")

    # Step 2: Model Compression (compress trained model only, skip integration test)
    if not args.skip_compression:
        success = run_command(
            [sys.executable, "compression.py", "--use-trained", "--config", args.config],
            "STEP 2: Model Compression"
        )
        if not success:
            print("⚠️  Compression failed. Stopping pipeline.")
            return 1
        all_success = all_success and success
    else:
        print(f"\n{'='*80}")
        print(f"  ⏭️  SKIPPING STEP 2: Compression")
        print(f"{'='*80}\n")
    
    # Step 3: Compression Analysis (writes directly to runs/<version>/<run_id>/analysis)
    if not args.skip_analysis:
        with open(args.config, encoding="utf-8") as f:
            run_cfg = yaml.safe_load(f)
        version_override = run_cfg.get("version", "run")
        if args.centralized:
            version_override = f"{version_override}_centralized"
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        runs_dir = Path("data/processed/runs") / version_override / run_id
        runs_dir.mkdir(parents=True, exist_ok=True)
        current_runs_dir = runs_dir
        analysis_out = runs_dir / "analysis"
        analysis_cmd = [
            sys.executable, "scripts/analyze_compression.py",
            "--config", args.config,
            "--output-dir-final", str(analysis_out),
            "--version", version_override,
            "--run-id", run_id,
        ]
        success = run_command(analysis_cmd, "STEP 3: Compression Analysis")
        if not success:
            print("⚠️  Analysis failed. Continuing to visualization...")
        all_success = all_success and success
    else:
        print(f"\n{'='*80}")
        print(f"  ⏭️  SKIPPING STEP 3: Analysis")
        print(f"{'='*80}\n")

    # Step 4: Ratio sweep (100:0 → 0:100) + threshold tuning → ratio_sweep_report.md
    if not args.skip_ratio_sweep:
        # Model for evaluation
        ratio_model = args.model.strip() if args.model else ""
        if not ratio_model:
            ratio_model = "models/tflite/saved_model_original.tflite"
            if not Path(ratio_model).exists():
                ratio_model = "src/models/global_model.h5"
        ratio_model_path = Path(ratio_model)
        if not ratio_model_path.exists():
            print(f"⚠️  Model not found: {ratio_model}. Skipping Step 4.")
        else:
            # Where to save the report
            last_run_file = Path("data/processed/runs/.last_run_id")
            eval_dir_from_model = get_eval_dir_for_model(ratio_model) if args.model else None
            if eval_dir_from_model is not None:
                # Model is under data/processed/runs/<version>/<datetime>/models/ → save to .../eval/
                report_dir = eval_dir_from_model
                report_dir.mkdir(parents=True, exist_ok=True)
                report_path = report_dir / "ratio_sweep_report.md"
                eval_report_dir = report_dir  # show in final summary
                print(f"  📁 Eval report dir (from --model): {report_dir}")
            elif last_run_file.exists():
                rel_path = last_run_file.read_text(encoding="utf-8").strip()
                report_dir = Path("data/processed/runs") / rel_path / "eval"
                report_dir.mkdir(parents=True, exist_ok=True)
                report_path = report_dir / "ratio_sweep_report.md"
            else:
                # Eval-only run: no analysis yet, model not under runs/ → data/processed/eval/<timestamp>/
                report_dir = Path("data/processed/eval") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                report_dir.mkdir(parents=True, exist_ok=True)
                report_path = report_dir / "ratio_sweep_report.md"
                print(f"  📁 Eval-only report dir: {report_dir}")

            sweep_cmd = [
                sys.executable, "scripts/evaluate_ratio_sweep.py",
                "--config", args.config,
                "--model", str(ratio_model_path),
                "--report", str(report_path),
            ]
            success = run_command(sweep_cmd, "STEP 4: Ratio Sweep (100%→0% normal:attack report)")
            if not success:
                print("⚠️  Ratio sweep failed. Continuing...")
            all_success = all_success and success
            if success and report_path.exists():
                tune_cmd = [
                    sys.executable, "scripts/tune_threshold_all_ratios.py",
                    "--config", args.config,
                    "--model", str(ratio_model_path),
                    "--append-to", str(report_path),
                ]
                tune_ok = run_command(tune_cmd, "STEP 4b: Threshold Tuning (append to ratio_sweep_report.md)")
                if not tune_ok:
                    print("⚠️  Threshold tuning failed. Ratio sweep report is still complete.")
                all_success = all_success and tune_ok
    else:
        print(f"\n{'='*80}")
        print(f"  ⏭️  SKIPPING STEP 4: Ratio sweep")
        print(f"{'='*80}\n")
    
    # Step 5: Result Visualization
    if not args.skip_viz:
        # Path from analyze (.last_run_id under data/processed/runs/)
        last_run_file = Path("data/processed/runs/.last_run_id")
        if last_run_file.exists():
            rel_path = last_run_file.read_text(encoding="utf-8").strip()
            analysis_dir = Path("data/processed/runs") / rel_path / "analysis"
        else:
            with open(args.config, encoding="utf-8") as f:
                run_cfg = yaml.safe_load(f)
            rel_path = run_cfg.get("version", "latest")
            analysis_dir = Path("data/processed/runs") / rel_path / "analysis"
        results_json = analysis_dir / "compression_analysis.json"
        viz_cmd = [
            sys.executable, "scripts/visualize_results.py",
            "--results", str(results_json),
            "--output-dir", str(analysis_dir),
        ]
        success = run_command(viz_cmd, "STEP 5: Result Visualization")
        if not success:
            print("⚠️  Visualization failed.")
        all_success = all_success and success
    else:
        print(f"\n{'='*80}")
        print(f"  ⏭️  SKIPPING STEP 5: Visualization")
        print(f"{'='*80}\n")
    
    # Final summary
    print(f"\n{'='*80}")
    if all_success:
        print(f"  ✅ PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        print(f"  ⚠️  PIPELINE COMPLETED WITH SOME WARNINGS")
    print(f"{'='*80}\n")
    
    # Run path (version/datetime) — single source: data/processed/runs/.last_run_id
    last_run_file = Path("data/processed/runs/.last_run_id")
    rel_path = last_run_file.read_text(encoding="utf-8").strip() if last_run_file.exists() else "latest"

    # Copy models/outputs into run dir only when we ran analysis (analysis already written there)
    if current_runs_dir is not None:
        runs_dir = current_runs_dir
        for name in ["models", "outputs"]:
            if Path(name).exists():
                dst = runs_dir / name
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(Path(name), dst)
        print(f"📁 Run snapshot: data/processed/runs/{rel_path}/")

    # Run history (for tracking progress)
    runs_index = Path("data/processed/runs/RUNS.md")
    runs_index.parent.mkdir(parents=True, exist_ok=True)
    header = "| Version | Run (datetime) |\n|---------|----------------|\n"
    line = f"| {rel_path.split('/')[0]} | {rel_path.split('/')[-1] if '/' in rel_path else '-'} |\n"
    if not runs_index.exists():
        runs_index.write_text("# Run History\n\n" + header + line, encoding="utf-8")
    else:
        content = runs_index.read_text(encoding="utf-8")
        if "| Version |" not in content:
            content = content.rstrip() + "\n\n" + header
        if line.strip() not in content:
            runs_index.write_text(content.rstrip() + "\n" + line, encoding="utf-8")

    _analysis_dir = f"data/processed/runs/{rel_path}/analysis"
    _eval_dir = f"data/processed/runs/{rel_path}/eval"
    print("Generated files:")
    print("  📦 Models:")
    print("     - src/models/global_model.h5 (FL trained)")
    print("     - models/tflite/saved_model_original.tflite")
    print("     - models/tflite/saved_model_pruned_quantized.tflite")
    print(f"  📊 Analysis ({rel_path}):")
    print(f"     - {_analysis_dir}/compression_analysis.csv")
    print(f"     - {_analysis_dir}/compression_analysis.json")
    print(f"     - {_analysis_dir}/compression_analysis.md")
    print(f"     - {_eval_dir}/ratio_sweep_report.md")
    if eval_report_dir is not None:
        print(f"  📊 Eval report (--model): {eval_report_dir}/ratio_sweep_report.md")
    print(f"  📈 Visualizations:")
    print(f"     - {_analysis_dir}/size_vs_accuracy.png")
    print(f"     - {_analysis_dir}/compression_metrics.png")
    print(f"     - {_analysis_dir}/compression_ratio.png")
    print()
    
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
