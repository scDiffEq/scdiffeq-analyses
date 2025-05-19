# -- import packages: ---------------------------------------------------------
import pathlib

# -- import local dependencies: -----------------------------------------------
from ._model_runner import Runner

# -- import external dependencies: --------------------------------------------
from typing import Dict, List


# -- define function: --------------------------------------------------------
def run_fate_prediction(
    project_name: str,
    h5ad_path: str,
    time_key: str,
    wandb_api_key: str,
    weight_key: str = "W",
    time_point: float = 2,
    ckpt_frequency: int = 1,
    save_last_ckpt: bool = True,
    keep_ckpts: int = 3,
    monitor: str = "epoch_validation_loss",
    n_seeds: int = 1,
    seeds: List[int] = [0],
    n_eval: int = 2000,
    swa_lrs: float = 1e-5,
    mu_hidden: List[int] = [512, 512],
    sigma_hidden: List[int] = [32, 32],
    train_epochs: int = 2500,
    train_lr: float = 4e-5,
    train_step_size: int = 1500,
    mu_dropout: float = 0,
    sigma_dropout: float = 0,
    batch_size: int = 2048,
    coef_diffusion: float = 1,
    diffeq_type: str = "SDE",
    use_key: str = "X_pca",
    potential_type: str = "fixed",
    latent_dim: int = 50,
    velocity_ratio_params: Dict[str, float] = {
        "target": 2.5,
        "enforce": 100,
        "method": "square",
    },
) -> None:
    runner = Runner(
        h5ad_path=h5ad_path,
        project_name=project_name,
        time_key=time_key,
        wandb_api_key=wandb_api_key,
        time_point=time_point,
        weight_key=weight_key,
        ckpt_frequency=ckpt_frequency,
        save_last_ckpt=save_last_ckpt,
        keep_ckpts=keep_ckpts,
        monitor=monitor,
    )
    return runner(
        n_seeds=n_seeds,
        seeds=seeds,
        n_eval=n_eval,
        swa_lrs=swa_lrs,
        mu_hidden=mu_hidden,
        sigma_hidden=sigma_hidden,
        train_epochs=train_epochs,
        train_lr=train_lr,
        train_step_size=train_step_size,
        mu_dropout=mu_dropout,
        sigma_dropout=sigma_dropout,
        batch_size=batch_size,
        potential_type=potential_type,
        coef_diffusion=coef_diffusion,
        diffeq_type=diffeq_type,
        use_key=use_key,
        latent_dim=latent_dim,
        velocity_ratio_params=velocity_ratio_params,
    )
