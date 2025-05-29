# -- import packages: ---------------------------------------------------------
import ABCParse
import anndata
import autodevice
import lightning
import larry
import logging
import pathlib
import scdiffeq as sdq
import shutil
import uuid
import wandb
import os

# -- import local dependencies: -----------------------------------------------
from . import funcs

# -- set type hints: ----------------------------------------------------------
from typing import Any, Dict, List, Literal, Optional, Union, Tuple

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)

# -- cls: ---------------------------------------------------------------------
class Runner(ABCParse.ABCParse):
    _script_name = pathlib.Path(__file__).name

    def __init__(
        self,
        h5ad_path: Union[pathlib.Path, str],
        project_name: str,
        wandb_api_key: str,
        time_key: str = "median_qbin",  # "t",  # "Time point",
        time_point: float = 2,
        weight_key: str = "W",
        ckpt_frequency=1,
        save_last_ckpt=True,
        keep_ckpts=3,
        monitor: str = "epoch_validation_loss",
    ) -> None:
        """
        Args:
            h5ad_path (Union[pathlib.Path, str]): Path to the h5ad file.
            project_name (str): Name of the project.
            time_key (str): Key for the time point.
            weight_key (str): Key for the weight.
            wandb_api_key (str): API key for wandb.
            ckpt_frequency (int): Frequency of saving checkpoints.
            save_last_ckpt (bool): Whether to save the last checkpoint.
            keep_ckpts (int): Number of checkpoints to keep.
            monitor (str): Metric to monitor.
        """
        self.__parse__(locals())

        wandb.login(key=self._wandb_api_key)

    @property
    def adata(self) -> anndata.AnnData:
        if not hasattr(self, "_ADATA"):
            self._ADATA = funcs.load_adata(
                h5ad_path=self._h5ad_path,
                weight_key=self._weight_key,
                time_key="Time point", # real time points
            )
        return self._ADATA

    @property
    def seeds(self) -> List[int]:
        if not hasattr(self, "_seeds_configured"):
            if hasattr(self, "_seeds") and (not self._seeds is None):
                self._seeds_configured = ABCParse.as_list(self._seeds)
            elif hasattr(self, "_n_seeds") and (not self._n_seeds is None):
                self._seeds_configured = range(self._n_seeds)
            else:
                self._seeds_configured = [0]
        return self._seeds_configured

    def _log_script_as_wandb_artifact(self, run_id: str) -> None:

        artifact = wandb.Artifact(
            name=f"{run_id}-{self._script_name}",
            type="run-script.py",
        )
        artifact.add_file(__file__)
        wandb.log_artifact(artifact)

    def _format_ckpt_fname(self, ckpt_fpath: pathlib.Path):
        return ckpt_fpath.name.split(".")[0].replace("=", "_").replace("-", ".")

    def _log_model_ckpt_as_wandb_artifact(self, seed, ckpt_fpath, run_id):

        ckpt_fname = self._format_ckpt_fname(ckpt_fpath)

        artifact = wandb.Artifact(
            name=f"model-ckpt-seed_{seed}-{ckpt_fname}-run_id-{run_id}", type="model"
        )
        artifact.add_file(ckpt_fpath)
        wandb.log_artifact(artifact)

        return ckpt_fname

    def _compose_model_params(self, seed: int) -> Dict[str,Any]:
        return {
            "adata": self.adata,
            "seed": seed,
            "latent_dim": self._latent_dim,
            "time_key": self._time_key,
            "use_key": self._use_key,
            "weight_key": self._weight_key,
            "mu_hidden": self._mu_hidden,
            "sigma_hidden": self._sigma_hidden,
            "mu_dropout": self._mu_dropout,
            "sigma_dropout": self._sigma_dropout,
            "DiffEq_type": self._diffeq_type,
            "batch_size": self._batch_size,
            "potential_type": self._potential_type,
            "coef_diffusion": self._coef_diffusion,
            "train_lr": self._train_lr,
            "train_step_size": self._train_step_size,
            "velocity_ratio_params": self._velocity_ratio_params,
            "monitor_hardware": True,
        }

    @property
    def _LOG_DIR(self):
        if not (hasattr(self, "model") and self.model and 
                hasattr(self.model, "DiffEq") and self.model.DiffEq and 
                hasattr(self.model.DiffEq, "trainer") and self.model.DiffEq.trainer and
                hasattr(self.model.DiffEq.trainer, "logger") and self.model.DiffEq.trainer.logger and
                hasattr(self.model.DiffEq.trainer.logger, "save_dir") and self.model.DiffEq.trainer.logger.save_dir):
            raise AttributeError(
                "Runner.model.DiffEq.trainer.logger.save_dir is not available. "
                "Ensure the model is initialized for the current run."
            )
        return pathlib.Path(self.model.DiffEq.trainer.logger.save_dir).parent

    @property
    def _CKPT_DIR(self) -> pathlib.Path:
        if not (hasattr(self, "model") and self.model and 
                hasattr(self.model, "DiffEq") and self.model.DiffEq and 
                hasattr(self.model.DiffEq, "_ckpt_path") and self.model.DiffEq._ckpt_path):
            raise AttributeError(
                "Runner.model.DiffEq._ckpt_path is not available. "
                "Ensure the model is initialized for the current run before accessing checkpoints."
            )
        return self.model.DiffEq._ckpt_path

    @property
    def _CKPT_PATHS(self):
        # Ensure _CKPT_DIR can be resolved and exists before globbing
        try:
            ckpt_dir = self._CKPT_DIR
            if not ckpt_dir.exists():
                logger.warning(f"Checkpoint directory {ckpt_dir} does not exist.")
                return iter([])  # Return empty iterator
        except AttributeError as e:
            logger.warning(f"Cannot determine checkpoint directory: {e}")
            return iter([]) # Return empty iterator
        return ckpt_dir.glob("*.ckpt")

    def _update_wandb_params(
            self,
            run_id: str,
            params: Dict[str, Any],
            wandb_logger: wandb.sdk.wandb_run.Run,
        ) -> Tuple[wandb.sdk.wandb_run.Run, Dict[str, Any]]:
        """Update the wandb parameters and return the wandb logger and params."""
        # Create a copy of params excluding the adata object to prevent serialization issues
        wandb_params = {k: v for k, v in params.items() if k != "adata"}
        wandb_logger.experiment.config.update(wandb_params)
        wandb_logger.experiment.config.update({"h5ad_path": self._h5ad_path})
        wandb_logger.experiment.config.update({"run_id": run_id})
        params["logger"] = wandb_logger
        return wandb_logger, params

    def _setup_run(self, seed: int) -> Tuple[wandb.sdk.wandb_run.Run, dict, str]:
        run_id = str(uuid.uuid4()).split("-")[0]
        wandb_logger = lightning.pytorch.loggers.WandbLogger(project=self._project_name)
        self._log_script_as_wandb_artifact(run_id=run_id)
        params = self._compose_model_params(seed=seed)
        wandb_logger, params = self._update_wandb_params(run_id=run_id, params=params, wandb_logger=wandb_logger)
        logger.info(f"Model run [run_id: {run_id}, seed: {seed}]: CONSTRUCTED")
        return wandb_logger, params, run_id

    def _wandb_fate_prediction_metrics_logging(
        self, seed: int, run_id: str, ckpt_fpath
    ) -> None:
        logger.info(f"Model run [run_id: {run_id}, seed: {seed}]: SETUP WANDB FATE PREDICTION METRICS LOGGING")
        ckpt_fname = self._format_ckpt_fname(ckpt_fpath)
        artifact = wandb.Artifact(
            name=f"fate-prediction-metrics-seed_{seed}-{ckpt_fname}-run_id_{run_id}",
            type="model-prediction-task",
            metadata={"N": self._n_eval, "seed": seed, "run_id": run_id},
        )
        artifact.add_dir(self._FATE_PREDICTION_CALLBACK._CKPT_METRICS_PATH)
        wandb.log_artifact(artifact)

    def fit_model(self, seed: int, run_id: str, params: Dict[str, Any]):
        logger.info(f"Model run [run_id: {run_id}, seed: {seed}]: INITIALIZED")
        self.model = sdq.scDiffEq(**params)

        SWA_CALLBACK = sdq.callbacks.StochasticWeightAveraging(swa_lrs=self._swa_lrs)
        self._FATE_PREDICTION_CALLBACK = larry.callbacks.FatePredictionCallback(
            self.adata, N=self._n_eval
        )

        self.model.fit(
            train_epochs=self._train_epochs,
            ckpt_frequency=self._ckpt_frequency,
            save_last_ckpt=self._save_last_ckpt,
            keep_ckpts=self._keep_ckpts,
            monitor=self._monitor,
            mode="min",
            train_callbacks=[SWA_CALLBACK, self._FATE_PREDICTION_CALLBACK],
        )

        # -- note: callback automatically performs eval at pl_module.on_train_end()
        # Ensure checkpoint directory exists
        os.makedirs(self._CKPT_DIR, exist_ok=True)

        # Save the checkpoint manually
        ckpt_fname = f"on_train_end.epoch_{self.model.DiffEq.current_epoch}.ckpt"
        ckpt_fpath = self._CKPT_DIR.joinpath(ckpt_fname)
        self.model.DiffEq.trainer.save_checkpoint(ckpt_fpath)
        logger.info(f"Model run [run_id: {run_id}, seed: {seed}]: SAVED CHECKPOINT: {ckpt_fpath}")

        self._log_model_ckpt_as_wandb_artifact(seed=seed, ckpt_fpath=ckpt_fpath, run_id=run_id)
        self._wandb_fate_prediction_metrics_logging(
            seed=seed, run_id=run_id, ckpt_fpath=ckpt_fpath
        )

    def export_model_ckpts(self, seed: int, run_id: str) -> None:
        
        output_dir = pathlib.Path(f"/output/run_id_{run_id}-seed_{seed}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for ckpt_fpath in self._CKPT_PATHS:
            shutil.copy(ckpt_fpath, output_dir)

    def fate_prediction_evaluation(self, seed: int, ckpt_fpath: pathlib.Path, run_id: str) -> None:
        logger.info(f"Model run [run_id: {run_id}, seed: {seed}]: EVALUATING CKPT: {ckpt_fpath}")
        ckpt_fname = self._log_model_ckpt_as_wandb_artifact(
            seed=seed, ckpt_fpath=ckpt_fpath, run_id=run_id
        )

        self.model.DiffEq = sdq.io.load_diffeq(ckpt_fpath)
        self.model.to(autodevice.AutoDevice())
        
        # Use run-specific log directory for fate prediction callback
        # This should be the parent of the wandb logger's save_dir (e.g., wandb/project/run_id/)
        run_specific_log_dir = pathlib.Path(self.model.DiffEq.trainer.logger.save_dir).parent
        
        self._FATE_PREDICTION_CALLBACK.forward(
            pl_module=self.model.DiffEq, ckpt_name=ckpt_fname, log_dir=run_specific_log_dir
        )
        self._wandb_fate_prediction_metrics_logging(seed=seed, run_id=run_id, ckpt_fpath=ckpt_fpath)

    def evaluate_model(self, seed: int, run_id: str) -> None:

        logger.info(f"Model run [run_id: {run_id}, seed: {seed}]: EVALUATING")

        for ckpt_fpath in self._CKPT_PATHS:
            self.fate_prediction_evaluation(
                seed=seed, ckpt_fpath=ckpt_fpath, run_id=run_id
            )
    def forward(self, seed: int) -> None:

        wandb_logger, params, run_id = None, None, None # Initialize to allow access in finally
        try:
            wandb.init(project=self._project_name)
            wandb_logger, params, run_id = self._setup_run(seed=seed)
            self.fit_model(seed=seed, run_id=run_id, params=params)
            # export_model_ckpts is likely redundant if ckpts are logged to wandb directly
            # self.export_model_ckpts(seed=seed, run_id=run_id)
            self.evaluate_model(seed=seed, run_id=run_id)
        except Exception as e:
            logger.error(f"Error during model run [run_id: {run_id}, seed: {seed}]: {e}", exc_info=True)
            # Potentially re-raise the exception if you want the overall process to fail
            # raise e 
        finally:
            # Ensure all checkpoints from the run are logged as artifacts
            if hasattr(self, "model") and self.model and hasattr(self.model, "DiffEq") and self.model.DiffEq and hasattr(self.model.DiffEq, "_ckpt_path") and self.model.DiffEq._ckpt_path:
                final_ckpt_dir = self.model.DiffEq._ckpt_path
                if final_ckpt_dir and final_ckpt_dir.exists() and run_id:
                    logger.info(f"Attempting to log all checkpoints from {final_ckpt_dir} for run_id: {run_id}, seed: {seed}")
                    for ckpt_fpath in final_ckpt_dir.glob("*.ckpt"):
                        try:
                            logger.info(f"Logging checkpoint {ckpt_fpath.name} to wandb artifacts.")
                            self._log_model_ckpt_as_wandb_artifact(seed=seed, ckpt_fpath=ckpt_fpath, run_id=run_id)
                        except Exception as log_e:
                            logger.error(f"Failed to log checkpoint {ckpt_fpath.name} for run_id {run_id}, seed {seed}: {log_e}")
                else:
                    logger.warning(f"Could not log all checkpoints for run_id: {run_id}, seed: {seed}. Checkpoint directory or run_id missing.")
            else:
                logger.warning(f"Model or checkpoint path not available for final artifact logging. run_id: {run_id}, seed: {seed}")

            if wandb.run is not None:
                wandb.finish()
            logger.info(f"Finished wandb run for seed: {seed}, run_id: {run_id}")

    def __call__(
        self,
        n_seeds: Optional[int] = None,
        seeds: Optional[Union[List[int], int]] = [],
        potential_type: Optional[Literal["fixed", "prior"]] = "fixed",
        train_epochs: int = 125,
        train_step_size: int = 25,
        mu_hidden: List[int] = [512, 512],
        sigma_hidden: List[int] = [32, 32],
        mu_dropout: float = 0,
        sigma_dropout: float = 0,
        train_lr=4e-5,
        n_eval: int = 2000,
        swa_lrs: float = 1e-8,
        batch_size: int = 512,
        coef_diffusion: float = 1,
        diffeq_type: str = "SDE",
        use_key: str = "X_pca",
        latent_dim: int = 50,
        velocity_ratio_params: Dict[str, Union[float, str]] = {
            "target": 1,
            "enforce": 0,
            "method": "square",
        },
    ) -> None:

        self.__update__(locals())

        for seed in self.seeds:
            self.forward(seed=seed)
