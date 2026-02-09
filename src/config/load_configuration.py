import yaml
from pathlib import Path


def load_config(config_path: str | Path) -> dict:
    """
        Load a YAML configuration file and return its contents as a dictionary.
    Args:
        config_path: Config file path as a string or Path object.

    Returns:
        (dict): Configuration data loaded from the YAML file.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {path}") from e

