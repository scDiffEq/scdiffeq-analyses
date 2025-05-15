# -- import packages: ---------------------------------------------------------
import json

# -- set type hints: ----------------------------------------------------------
from typing import Any, Dict, List


# -- function: ----------------------------------------------------------------
def build_experiment_command(
    experiment: Dict[str, Any], common: Dict[str, Any]
) -> List[str]:
    """
    Build the command-line arguments for running an experiment.

    Args:
        experiment: Dictionary containing experiment-specific parameters
        common: Dictionary containing common parameters

    Returns:
        List of command-line arguments
    """
    # Extract parameters
    mu_hidden = experiment.get("mu_hidden", common.get("mu_hidden", [32, 32]))
    sigma_hidden = experiment.get("sigma_hidden", common.get("sigma_hidden", [32, 32]))
    velocity_ratio_target = experiment.get(
        "velocity_ratio_target", common.get("velocity_ratio_target", 2.5)
    )
    velocity_ratio_enforce = experiment.get(
        "velocity_ratio_enforce", common.get("velocity_ratio_enforce", 100)
    )
    velocity_ratio_method = common.get("velocity_ratio_method", "square")

    # Extract common parameters
    train_epochs = common.get("train_epochs", 2500)
    train_lr = common.get("train_lr", 1e-4)
    train_step_size = common.get("train_step_size", 1500)
    n_seeds = common.get("n_seeds", 5)
    n_eval = common.get("n_eval", 2000)
    swa_lrs = common.get("swa_lrs", 1e-5)
    mu_dropout = common.get("mu_dropout", 0)
    sigma_dropout = common.get("sigma_dropout", 0)
    batch_size = common.get("batch_size", 2048)
    potential_type = common.get("potential_type", "fixed")
    coef_g = common.get("coef_g", 1)

    # Build command
    return [
        "python",
        "run.py",
        f'--mu_hidden="{json.dumps(mu_hidden)}"',
        f'--sigma_hidden="{json.dumps(sigma_hidden)}"',
        f"--velocity_ratio_target={velocity_ratio_target}",
        f"--velocity_ratio_enforce={velocity_ratio_enforce}",
        f"--velocity_ratio_method={velocity_ratio_method}",
        f"--train_epochs={train_epochs}",
        f"--train_lr={train_lr}",
        f"--train_step_size={train_step_size}",
        f"--n_seeds={n_seeds}",
        f"--n_eval={n_eval}",
        f"--swa_lrs={swa_lrs}",
        f"--mu_dropout={mu_dropout}",
        f"--sigma_dropout={sigma_dropout}",
        f"--batch_size={batch_size}",
        f"--potential_type={potential_type}",
        f"--coef_g={coef_g}",
    ]
