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
import subprocess
import sys
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
    
    args = parser.parse_args()
    
    # Track overall success
    all_success = True
    
    print(f"\n{'='*80}")
    print(f"  🚀 STARTING TINYML COMPLETE PIPELINE")
    print(f"{'='*80}\n")
    print(f"Configuration: {args.config}")
    print(f"Skip train: {args.skip_train}")
    print(f"Skip compression: {args.skip_compression}")
    print(f"Skip analysis: {args.skip_analysis}")
    print(f"Skip visualization: {args.skip_viz}")
    
    # Step 1: Federated Learning Training
    if not args.skip_train:
        success = run_command(
            [sys.executable, "scripts/train.py", "--config", args.config],
            "STEP 1: Federated Learning Training"
        )
        if not success:
            print("⚠️  Training failed. Stopping pipeline.")
            return 1
        all_success = all_success and success
    else:
        print(f"\n{'='*80}")
        print(f"  ⏭️  SKIPPING STEP 1: Training (using existing model)")
        print(f"{'='*80}\n")
    
    # Step 2: Model Compression
    if not args.skip_compression:
        success = run_command(
            [sys.executable, "compression.py"],
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
        success = run_command(
            [sys.executable, "scripts/analyze_compression.py"],
            "STEP 3: Compression Analysis"
        )
        if not success:
            print("⚠️  Analysis failed. Continuing to visualization...")
        all_success = all_success and success
    else:
        print(f"\n{'='*80}")
        print(f"  ⏭️  SKIPPING STEP 3: Analysis")
        print(f"{'='*80}\n")
    
    # Step 4: Result Visualization
    if not args.skip_viz:
        success = run_command(
            [sys.executable, "scripts/visualize_results.py"],
            "STEP 4: Result Visualization"
        )
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
    
    print("Generated files:")
    print("  📦 Models:")
    print("     - src/models/global_model.h5 (FL trained)")
    print("     - models/tflite/saved_model_original.tflite")
    print("     - models/tflite/saved_model_pruned_quantized.tflite")
    print("  📊 Analysis:")
    print("     - data/processed/analysis/compression_analysis.csv")
    print("     - data/processed/analysis/compression_analysis.json")
    print("     - data/processed/analysis/compression_analysis.md")
    print("  📈 Visualizations:")
    print("     - data/processed/analysis/size_vs_accuracy.png")
    print("     - data/processed/analysis/compression_metrics.png")
    print("     - data/processed/analysis/compression_ratio.png")
    print()
    
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
