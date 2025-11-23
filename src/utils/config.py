"""
Utility functions for configuration management
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_data_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Get data paths from configuration

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of data paths
    """
    base_dir = Path(__file__).parent.parent

    return {
        "raw": base_dir / config["data"]["raw_dir"],
        "processed": base_dir / config["data"]["processed_dir"],
        "models": base_dir / config["data"]["models_dir"],
    }
