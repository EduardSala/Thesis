import yaml
from pathlib import Path
from utils.logger_setup import logger


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
        logger.warning(f"Config file not found: {path}\n")

    try:
        with path.open("r", encoding="utf-8") as f:
            logger.info(f"Config file has been loaded!\n")
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {path}, error: {e}\n")

