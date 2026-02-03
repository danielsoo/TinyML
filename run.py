"""
Complete TinyML Pipeline Runner

Executes the full workflow:
1. Train model with Federated Learning (train.py)
2. Compress model (compression.py)
3. Analyze compression results (analyze_compression.py)
4. Visualize results (visualize_results.py)

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-train  # Skip FL training, use existing model
    python run_pipeline.py --skip-viz    # Skip visualization
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

    args = parser.parse_args()
    
    # Track overall success
    all_success = True
    
    print(f"\n{'='*80}")
    print(f"  🚀 STARTING TINYML COMPLETE PIPELINE")
    print(f"{'='*80}\n")
    print(f"Configuration: {args.config}")
    print(f"Training mode: {'Centralized (no FedAvg)' if args.centralized else 'Federated Learning'}")
    print(f"Skip train: {args.skip_train}")
    print(f"Skip compression: {args.skip_compression}")
    print(f"Skip analysis: {args.skip_analysis}")
    print(f"Skip visualization: {args.skip_viz}")
    
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
    
    # Step 3: Compression Analysis
    if not args.skip_analysis:
        with open(args.config, encoding="utf-8") as f:
            run_cfg = yaml.safe_load(f)
        version_override = None
        if args.centralized:
            base_ver = run_cfg.get("version", "run")
            version_override = f"{base_ver}_centralized"
        analysis_cmd = [
            sys.executable, "scripts/analyze_compression.py",
            "--config", args.config,
        ]
        if version_override:
            analysis_cmd.extend(["--version", version_override])
        success = run_command(analysis_cmd, "STEP 3: Compression Analysis")
        if not success:
            print("⚠️  Analysis failed. Continuing to visualization...")
        all_success = all_success and success
    else:
        print(f"\n{'='*80}")
        print(f"  ⏭️  SKIPPING STEP 3: Analysis")
        print(f"{'='*80}\n")
    
    # Step 4: Result Visualization
    if not args.skip_viz:
        # Path from analyze (.last_run_id: version/datetime)
        last_run_file = Path("data/processed/analysis/.last_run_id")
        if last_run_file.exists():
            rel_path = last_run_file.read_text(encoding="utf-8").strip()
            analysis_dir = Path("data/processed/analysis") / rel_path
        else:
            with open(args.config, encoding="utf-8") as f:
                run_cfg = yaml.safe_load(f)
            rel_path = run_cfg.get("version", "latest")
            analysis_dir = Path("data/processed/analysis") / rel_path
        results_json = analysis_dir / "compression_analysis.json"
        viz_cmd = [
            sys.executable, "scripts/visualize_results.py",
            "--results", str(results_json),
            "--output-dir", str(analysis_dir),
        ]
        success = run_command(viz_cmd, "STEP 4: Result Visualization")
        if not success:
            print("⚠️  Visualization failed.")
        all_success = all_success and success
    else:
        print(f"\n{'='*80}")
        print(f"  ⏭️  SKIPPING STEP 4: Visualization")
        print(f"{'='*80}\n")
    
    # Final summary
    print(f"\n{'='*80}")
    if all_success:
        print(f"  ✅ PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        print(f"  ⚠️  PIPELINE COMPLETED WITH SOME WARNINGS")
    print(f"{'='*80}\n")
    
    # Run path (version/datetime)
    last_run_file = Path("data/processed/analysis/.last_run_id")
    rel_path = last_run_file.read_text(encoding="utf-8").strip() if last_run_file.exists() else "latest"

    # Per-run snapshot archive (version/datetime structure)
    runs_dir = Path("data/processed/runs") / rel_path
    runs_dir.mkdir(parents=True, exist_ok=True)
    if (Path("data/processed/analysis") / rel_path).exists():
        dst = runs_dir / "analysis"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(Path("data/processed/analysis") / rel_path, dst)
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

    _analysis_dir = f"data/processed/analysis/{rel_path}"
    print("Generated files:")
    print("  📦 Models:")
    print("     - src/models/global_model.h5 (FL trained)")
    print("     - models/tflite/saved_model_original.tflite")
    print("     - models/tflite/saved_model_pruned_quantized.tflite")
    print(f"  📊 Analysis ({rel_path}):")
    print(f"     - {_analysis_dir}/compression_analysis.csv")
    print(f"     - {_analysis_dir}/compression_analysis.json")
    print(f"     - {_analysis_dir}/compression_analysis.md")
    print(f"  📈 Visualizations:")
    print(f"     - {_analysis_dir}/size_vs_accuracy.png")
    print(f"     - {_analysis_dir}/compression_metrics.png")
    print(f"     - {_analysis_dir}/compression_ratio.png")
    print()
    
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
