# -- import packages: ---------------------------------------------------------
import yaml
import os
import json

# -- function: ----------------------------------------------------------------
def save_default_config(output_path: str) -> None:
    """
    Generate and save a default experiment configuration file.

    Args:
        output_path: Path to save the configuration file
    """
    default_config = {
        "experiments": [
            {
                "id": 1,
                "mu_hidden": [32, 32],
                "sigma_hidden": [32, 32],
                "velocity_ratio_target": 2.5,
                "velocity_ratio_enforce": 100,
            },
            # Additional default experiments can be added here
        ],
        "common": {
            "train_epochs": 2500,
            "train_lr": 1e-4,
            "train_step_size": 1500,
            "n_seeds": 5,
            "n_eval": 2000,
            "swa_lrs": 1e-5,
            "mu_dropout": 0,
            "sigma_dropout": 0,
            "batch_size": 2048,
            "potential_type": "fixed",
            "coef_g": 1,
            "velocity_ratio_method": "square",
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False)

    print(f"Default configuration saved to {output_path}")
