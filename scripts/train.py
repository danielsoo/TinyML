#!/usr/bin/env python3
"""Unified training script that works in both local and Colab environments.

This script provides a Colab notebook-like experience for local execution,
automatically detecting the environment and using appropriate configuration.

Usage:
    # Local execution (auto-detects environment)
    python scripts/train.py

    # Specify config explicitly
    python scripts/train.py --config config/federated_local.yaml

    # Colab execution (auto-detects Colab environment)
    python scripts/train.py
"""
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.utils.env_utils import (
        is_colab,
        is_colab_runtime,
        get_default_config_path,
        get_project_root,
        ensure_dependencies_installed,
        get_default_data_path,
    )
except ImportError:
    # Fallback if env_utils not available
    def is_colab():
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def is_colab_runtime():
        return os.getenv("COLAB_GPU") is not None or os.path.exists("/content")
    
    def get_default_config_path():
        if is_colab() or is_colab_runtime():
            return "config/federated_colab.yaml"
        return "config/federated_local.yaml"
    
    def get_project_root():
        return Path.cwd()
    
    def ensure_dependencies_installed():
        pass
    
    def get_default_data_path():
        if is_colab() or is_colab_runtime():
            return "/content/drive/MyDrive/TinyML_models"
        return "data/raw/Bot-IoT"


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def check_gpu():
    """Check GPU availability."""
    print_section("🔍 GPU Check")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ GPU devices found: {len(gpus)}")
            for gpu in gpus:
                print(f"   - {gpu.name}")
        else:
            print("⚠️  No GPU devices found. Training will use CPU (may be slower).")
            print("   Tip: In Colab, enable GPU via Runtime → Change runtime type → GPU")
        
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check protobuf version
        try:
            import google.protobuf
            print(f"Protobuf version: {google.protobuf.__version__}")
        except:
            pass
            
    except ImportError:
        print("⚠️  TensorFlow not installed. Please install dependencies first.")


def ensure_project_setup():
    """Ensure project is set up correctly."""
    print_section("📁 Project Setup")
    
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    
    # Change to project root
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")
    
    # Check essential directories
    required_dirs = ["src", "config"]
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ directory found")
        else:
            print(f"⚠️  {dir_name}/ directory not found at {dir_path}")


def install_dependencies():
    """Install dependencies if in Colab, or check if installed locally."""
    print_section("📦 Dependencies")
    
    if is_colab() or is_colab_runtime():
        print("🔧 Installing dependencies in Colab environment...")
        
        # Install requirements_colab.txt if exists
        project_root = get_project_root()
        colab_req = project_root / "colab" / "requirements_colab.txt"
        if colab_req.exists():
            print(f"   Installing from {colab_req}...")
            subprocess.run(
                ["pip", "install", "-r", str(colab_req)],
                check=False
            )
        
        # Fix protobuf compatibility
        print("   Fixing protobuf compatibility...")
        subprocess.run(
            ["pip", "install", "--force-reinstall", "protobuf==3.20.3"],
            check=False
        )
        
        # Install Flower if needed
        subprocess.run(
            ["pip", "install", "flwr[simulation]"],
            check=False
        )
        
        print("✅ Dependencies installed")
    else:
        print("ℹ️  Local environment detected. Assuming dependencies are installed.")
        print("   If not, run: pip install -r requirements.txt")


def verify_data_path(config_path: str):
    """Verify that data path from config exists."""
    print_section("📊 Data Verification")
    
    try:
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        data_path = config.get("data", {}).get("path", None)
        if data_path:
            data_path_obj = Path(data_path)
            if data_path_obj.exists():
                # Check for CSV files
                csv_files = list(data_path_obj.glob("*.csv"))
                if csv_files:
                    print(f"✅ Data path found: {data_path}")
                    print(f"   Found {len(csv_files)} CSV file(s)")
                    for csv_file in csv_files[:5]:  # Show first 5
                        print(f"   - {csv_file.name}")
                    if len(csv_files) > 5:
                        print(f"   ... and {len(csv_files) - 5} more")
                else:
                    print(f"⚠️  Data path exists but no CSV files found: {data_path}")
            else:
                print(f"⚠️  Data path does not exist: {data_path}")
                default_path = get_default_data_path()
                if default_path and default_path != data_path:
                    print(f"   Expected default path: {default_path}")
        else:
            print("⚠️  No data path specified in config")
            
    except Exception as e:
        print(f"⚠️  Could not verify data path: {e}")


def run_training(config_path: str, save_model: str = None):
    """Run the federated learning training."""
    print_section("🚀 Starting Training")
    
    # Generate timestamped model filename if not specified
    if save_model is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = get_project_root()
        model_dir = project_root / "src" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        save_model = str(model_dir / f"global_model_{timestamp}.h5")
    
    print(f"Configuration: {config_path}")
    print(f"Model will be saved to: {save_model}")
    print()
    
    # Run training
    cmd = [
        sys.executable,
        "-m", "src.federated.client",
        "--config", config_path,
        "--save-model", save_model,
    ]
    
    result = subprocess.run(cmd, check=False)
    
    # Check if model file was actually created (more reliable than exit code)
    model_path = Path(save_model)
    model_exists = model_path.exists() and model_path.stat().st_size > 0
    
    if result.returncode == 0 or model_exists:
        print()
        print_section("✅ Training Complete")
        print(f"Model saved to: {save_model}")
        
        # Also save as latest for easy access
        project_root = get_project_root()
        latest_path = project_root / "src" / "models" / "global_model.h5"
        if model_exists:
            import shutil
            shutil.copy(save_model, latest_path)
            print(f"Also saved as latest: {latest_path}")
        
        # If exit code was non-zero but model exists, warn but don't fail
        if result.returncode != 0:
            print()
            print("⚠️  Training process returned non-zero exit code, but model was saved successfully.")
            print("   This may indicate warnings or non-critical errors during training.")
    else:
        print()
        print_section("❌ Training Failed")
        print(f"Model file not found at: {save_model}")
        print(f"Exit code: {result.returncode}")
        sys.exit(result.returncode if result.returncode != 0 else 1)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified training script for local and Colab environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect environment and use appropriate config
  python scripts/train.py

  # Specify config explicitly
  python scripts/train.py --config config/federated_local.yaml

  # Skip dependency installation
  python scripts/train.py --skip-deps

  # Skip GPU check
  python scripts/train.py --skip-gpu-check
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: auto-detect based on environment)",
    )
    
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Path to save the trained model (default: timestamped filename)",
    )
    
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Skip GPU availability check",
    )
    
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip dependency installation check",
    )
    
    parser.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip data path verification",
    )
    
    args = parser.parse_args()
    
    # Detect environment
    in_colab = is_colab() or is_colab_runtime()
    env_name = "Colab" if in_colab else "Local"
    print(f"\n🌍 Environment: {env_name}")
    
    # Get config path
    if args.config is None:
        args.config = get_default_config_path()
        print(f"📋 Auto-selected config: {args.config}")
    
    # Run setup steps
    ensure_project_setup()
    
    if not args.skip_gpu_check:
        check_gpu()
    
    if not args.skip_deps:
        install_dependencies()
    
    if not args.skip_data_check:
        verify_data_path(args.config)
    
    # Run training
    run_training(args.config, args.save_model)


if __name__ == "__main__":
    main()

