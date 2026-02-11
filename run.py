"""
Complete TinyML Pipeline Runner

Executes the full workflow:
1. Train model with Federated Learning (train.py)
2. Compress model (compression.py)
3. Analyze compression results (analyze_compression.py)
4. Ratio sweep (evaluate_ratio_sweep.py) → ratio_sweep_report.md
3b. FGSM (run_fgsm.py) → run_dir/fgsm/fgsm_report.md + fgsm_results.json
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
        "--skip-fgsm",
        action="store_true",
        help="Skip FGSM attack step (run_dir/fgsm/)"
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
    parser.add_argument(
        "--log",
        type=str,
        default="",
        help="Tee full stdout/stderr to this file (e.g. run.log).",
    )

    args = parser.parse_args()

    # Optional: tee output to log file
    log_path = getattr(args, "log", "") or ""
    if log_path:
        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
        log_file = open(log_path, "w", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, log_file)
        sys.stderr = Tee(sys.__stderr__, log_file)
        print(f"Logging to {log_path}")
    
    # Track overall success, step results for final report, eval dir, run dir
    all_success = True
    step_results = []  # list of (step_name, success) for final summary
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
        step_results.append(("STEP 1: Training", success))
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

        # Build traditional (no-QAT) model if missing so all three TFLite outputs are produced
        with open(args.config, encoding="utf-8") as f:
            run_cfg = yaml.safe_load(f)
        use_real_qat = run_cfg.get("federated", {}).get("use_qat", False)
        comp_cfg = run_cfg.get("compression", {})
        always_build_traditional = comp_cfg.get("always_build_traditional", True)
        trad_path = Path(comp_cfg.get("traditional_model_path") or "models/global_model_traditional.h5")
        if use_real_qat and always_build_traditional and not trad_path.exists():
            print(f"\n{'='*80}")
            print(f"  📦 Building Traditional model (FL without QAT) for saved_model_no_qat_ptq.tflite")
            print(f"{'='*80}\n")
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                run_cfg["federated"] = {**run_cfg.get("federated", {}), "use_qat": False}
                yaml.dump(run_cfg, tmp, default_flow_style=False, allow_unicode=True, sort_keys=False)
                tmp_path = tmp.name
            try:
                ok = run_command(
                    [sys.executable, "scripts/train.py", "--config", tmp_path],
                    "Traditional FL (no QAT)"
                )
                if ok and Path("src/models/global_model.h5").exists():
                    trad_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy("src/models/global_model.h5", trad_path)
                    print(f"   📋 Traditional model saved to {trad_path}\n")
                    # Restore QAT model for compression (second train overwrote src/models/global_model.h5)
                    if dst_model.exists():
                        shutil.copy(dst_model, src_model)
                        print(f"   📋 Restored QAT model to {src_model}\n")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
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
        step_results.append(("STEP 2: Compression", success))
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
        # Save full run config for reports and reproducibility
        run_config_path = runs_dir / "run_config.yaml"
        with open(run_config_path, "w", encoding="utf-8") as f:
            yaml.dump(run_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"   📋 Saved run config: {run_config_path}\n")
        analysis_out = runs_dir / "analysis"
        analysis_cmd = [
            sys.executable, "scripts/analyze_compression.py",
            "--config", args.config,
            "--output-dir-final", str(analysis_out),
            "--version", version_override,
            "--run-id", run_id,
        ]
        success = run_command(analysis_cmd, "STEP 3: Compression Analysis")
        step_results.append(("STEP 3: Analysis", success))
        if not success:
            print("⚠️  Analysis failed. Continuing to visualization...")
        all_success = all_success and success
    else:
        print(f"\n{'='*80}")
        print(f"  ⏭️  SKIPPING STEP 3: Analysis")
        print(f"{'='*80}\n")

    # Step 3b: FGSM (write run_dir/fgsm/fgsm_report.md + fgsm_results.json) — compare all models
    if current_runs_dir is not None and not args.skip_fgsm:
        fgsm_out = current_runs_dir / "fgsm"
        fgsm_model = Path("models/global_model.h5")
        if fgsm_model.exists():
            with open(args.config, encoding="utf-8") as f:
                _eval_cfg = yaml.safe_load(f).get("evaluation", {})
            _raw = _eval_cfg.get("ratio_sweep_models")
            ratio_list = [m.strip() for m in _raw] if _raw and isinstance(_raw, list) else []
            fgsm_models = [str(fgsm_model)] + [m for m in ratio_list if m]
            fgsm_models = [p for p in fgsm_models if Path(p).exists()]
            fgsm_cmd = [
                sys.executable, "scripts/run_fgsm.py",
                "--models", *fgsm_models,
                "--config", args.config,
                "--output-dir", str(fgsm_out),
            ]
            success = run_command(fgsm_cmd, "STEP 3b: FGSM (run_dir/fgsm/)")
            step_results.append(("STEP 3b: FGSM", success))
            if not success:
                print("⚠️  FGSM failed. Continuing...")
            all_success = all_success and success
        else:
            print("⚠️  models/global_model.h5 not found. Skipping FGSM.")

    # Step 4: Ratio sweep (100:0 → 0:100) + threshold tuning for one or all models
    eval_report_dir = None
    step4_success = True
    if not args.skip_ratio_sweep:
        last_run_file = Path("data/processed/runs/.last_run_id")
        eval_dir_from_model = get_eval_dir_for_model(args.model) if args.model else None
        if eval_dir_from_model is not None:
            report_dir = eval_dir_from_model
            report_dir.mkdir(parents=True, exist_ok=True)
            eval_report_dir = report_dir
            print(f"  📁 Eval report dir (from --model): {report_dir}")
        elif last_run_file.exists():
            rel_path = last_run_file.read_text(encoding="utf-8").strip()
            report_dir = Path("data/processed/runs") / rel_path / "eval"
            report_dir.mkdir(parents=True, exist_ok=True)
            eval_report_dir = report_dir
        else:
            report_dir = Path("data/processed/eval") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            report_dir.mkdir(parents=True, exist_ok=True)
            eval_report_dir = report_dir
            print(f"  📁 Eval-only report dir: {report_dir}")

        # Build model list: single (--model) or all from config (same as compression_analysis)
        if args.model and args.model.strip():
            ratio_models = [args.model.strip()]
        else:
            with open(args.config, encoding="utf-8") as f:
                _eval_cfg = yaml.safe_load(f).get("evaluation", {})
            _raw = _eval_cfg.get("ratio_sweep_models")
            if _raw and isinstance(_raw, list):
                ratio_models = [m.strip() for m in _raw if m and str(m).strip()]
            else:
                ratio_models = [
                    "models/tflite/saved_model_original.tflite",
                    "models/tflite/saved_model_qat_pruned_float32.tflite",
                    "models/tflite/saved_model_qat_ptq.tflite",
                    "models/tflite/saved_model_no_qat_ptq.tflite",
                    "models/tflite/saved_model_pruned_qat.tflite",
                    "models/tflite/saved_model_pruned_quantized.tflite",
                ]
            if not any(Path(m).exists() for m in ratio_models):
                ratio_models = ["models/tflite/saved_model_original.tflite"]
            if not ratio_models or not any(Path(m).exists() for m in ratio_models):
                ratio_models = ["src/models/global_model.h5"]

        ratio_model_paths = [Path(m) for m in ratio_models if Path(m).exists()]
        report_path = report_dir / "ratio_sweep_report.md"

        # Step 4: one ratio sweep over all models → single comparison report
        if ratio_model_paths:
            sweep_cmd = [
                sys.executable, "scripts/evaluate_ratio_sweep.py",
                "--config", args.config,
                "--models", *[str(p) for p in ratio_model_paths],
                "--report", str(report_path),
            ]
            success = run_command(sweep_cmd, "STEP 4: Ratio Sweep (all models)")
            step4_success = step4_success and success
            if not success:
                print("  ⚠️  Ratio sweep failed. Continuing...")
            all_success = all_success and success

            # Step 4b: threshold tuning per model, append to same report
            if success and report_path.exists():
                for ratio_model_path in ratio_model_paths:
                    stem = ratio_model_path.stem
                    tune_cmd = [
                        sys.executable, "scripts/tune_threshold_all_ratios.py",
                        "--config", args.config,
                        "--model", str(ratio_model_path),
                        "--append-to", str(report_path),
                    ]
                    tune_ok = run_command(tune_cmd, f"STEP 4b: Threshold Tuning — {stem}")
                    step4_success = step4_success and tune_ok
                    if not tune_ok:
                        print(f"  ⚠️  Threshold tuning failed for {stem}. Report is still complete.")
                    all_success = all_success and tune_ok
        step_results.append(("STEP 4: Ratio sweep", step4_success))
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
        step_results.append(("STEP 5: Visualization", success))
        if not success:
            print("⚠️  Visualization failed.")
        all_success = all_success and success
    else:
        print(f"\n{'='*80}")
        print(f"  ⏭️  SKIPPING STEP 5: Visualization")
        print(f"{'='*80}\n")
    
    # Final summary: 완료/실패를 로그에서 정확히 표시
    failed_steps = [name for name, ok in step_results if not ok]
    print(f"\n{'='*80}")
    print(f"  PIPELINE RESULT")
    print(f"{'='*80}")
    if all_success:
        print(f"  ✅ Result: SUCCESS — All steps completed.")
    else:
        print(f"  ❌ Result: FAILED — {len(failed_steps)} step(s) failed.")
        for name in failed_steps:
            print(f"     • {name}")
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
    print("     - models/tflite/saved_model_qat_pruned_float32.tflite (QAT+Prune only)")
    print("     - models/tflite/saved_model_qat_ptq.tflite")
    print("     - models/tflite/saved_model_pruned_quantized.tflite (legacy)")
    print("     - models/tflite/saved_model_no_qat_ptq.tflite (if traditional model was built)")
    print(f"  📊 Analysis ({rel_path}):")
    print(f"     - {_analysis_dir}/compression_analysis.csv")
    print(f"     - {_analysis_dir}/compression_analysis.json")
    print(f"     - {_analysis_dir}/compression_analysis.md")
    print(f"     - {_eval_dir}/ratio_sweep_report.md or ratio_sweep_<model>.md (one per model)")
    _fgsm_dir = f"data/processed/runs/{rel_path}/fgsm"
    if Path(_fgsm_dir).exists():
        print(f"  🎯 FGSM ({rel_path}):")
        print(f"     - {_fgsm_dir}/fgsm_report.md")
        print(f"     - {_fgsm_dir}/fgsm_results.json")
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
