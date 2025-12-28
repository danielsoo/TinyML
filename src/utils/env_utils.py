"""Environment detection and configuration utilities.

This module provides functions to detect the execution environment
(Colab, local, etc.) and automatically select appropriate configuration.
"""
import os
from pathlib import Path
from typing import Optional


def is_colab() -> bool:
    """Check if running in Google Colab environment.
    
    Returns:
        True if running in Colab, False otherwise.
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_colab_runtime() -> bool:
    """Check if running in Colab by checking environment variables.
    
    Alternative check that doesn't require importing google.colab.
    
    Returns:
        True if COLAB_GPU or similar Colab environment variables are set.
    """
    return os.getenv("COLAB_GPU") is not None or os.path.exists("/content")


def get_default_config_path() -> str:
    """Get the default configuration file path based on environment.
    
    Returns:
        Path to configuration file (federated_colab.yaml or federated_local.yaml).
    """
    if is_colab() or is_colab_runtime():
        return "config/federated_colab.yaml"
    else:
        return "config/federated_local.yaml"


def get_project_root() -> Path:
    """Get the project root directory.
    
    In Colab, this is typically /content/TinyML.
    In local environment, this is the directory containing this file.
    
    Returns:
        Path object pointing to project root.
    """
    if is_colab() or is_colab_runtime():
        # In Colab, project is typically cloned to /content/TinyML
        colab_path = Path("/content/TinyML")
        if colab_path.exists():
            return colab_path
    
    # In local environment, find project root by looking for config directory
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config").exists() and (current / "src").exists():
            return current
        current = current.parent
    
    # Fallback: assume we're already at project root
    return Path.cwd()


def ensure_dependencies_installed() -> None:
    """Ensure required dependencies are installed.
    
    In Colab, this may install additional packages.
    In local environment, assumes dependencies are already installed.
    """
    if is_colab() or is_colab_runtime():
        import subprocess
        
        print("🔧 Checking dependencies in Colab environment...")
        
        # Fix protobuf compatibility issue with TensorFlow
        print("   Fixing protobuf compatibility (TensorFlow requires protobuf==3.20.3)...")
        subprocess.run(
            ["pip", "install", "--force-reinstall", "protobuf==3.20.3"],
            check=False,
            capture_output=True
        )
        print("✅ Dependencies check complete")
    else:
        # In local environment, assume dependencies are managed via requirements.txt
        pass


def get_default_data_path() -> Optional[str]:
    """Get default data path based on environment.
    
    Returns:
        Default data path string, or None if not determinable.
    """
    if is_colab() or is_colab_runtime():
        # In Colab, data is typically in Google Drive
        return "/content/drive/MyDrive/TinyML_models"
    else:
        # In local environment, data is in project directory
        return "data/raw/Bot-IoT"


def get_default_model_save_dir() -> str:
    """Get default directory for saving models.
    
    Returns:
        Path string for model save directory.
    """
    if is_colab() or is_colab_runtime():
        # In Colab, optionally save to Drive as well
        return "src/models"
    else:
        # In local environment, save to project directory
        return "src/models"

