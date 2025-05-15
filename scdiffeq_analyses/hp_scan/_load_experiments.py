# -- import packages: ---------------------------------------------------------
import yaml  

# -- set type hints: ----------------------------------------------------------
from typing import Any, Dict, List, Tuple

# -- function: ----------------------------------------------------------------
def load_experiments(config_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load experiment configurations from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Tuple containing (list of experiment configs, common parameters)
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config["experiments"], config["common"]
