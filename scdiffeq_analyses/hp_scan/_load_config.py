# -- import packages: ---------------------------------------------------------
import pathlib
import yaml

# -- run: ---------------------------------------------------------------------
def load_config() -> dict:

    config_path = pathlib.Path(__file__).parents[2].joinpath("assets/experiments.yaml")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
