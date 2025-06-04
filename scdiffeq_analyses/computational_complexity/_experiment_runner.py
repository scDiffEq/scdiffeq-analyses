# -- import packages: ---------------------------------------------------------
import anndata
import ABCParse
import anndata
import lightning
import scdiffeq as sdq
import uuid
import wandb

# -- import local dependencies: -----------------------------------------------
from ._gmm_sampling import GaussianMixtureModelSampling

# -- cls: ---------------------------------------------------------------------
class ExperimentRunner(ABCParse.ABCParse):

    def __init__(
        self,
        wandb_api_key: str,
        project_name: str,
        adata: anndata.AnnData,
        n_components: int = 50,
        n_gmm_components: int = 20,
        random_state: int = 42,
        ckpt_frequency: int = 1,
        save_last_ckpt: bool = True,
        keep_ckpts: int = 3,
        train_epochs: int = 2500,
        monitor: str = "epoch_validation_loss",
        pseudotime_key: str = "ct_pseudotime",
    ) -> None:

        self.__parse__(locals())

        self._run_id = str(uuid.uuid4()).split("-")[0]

        wandb.login(key=self._wandb_api_key)

    def _setup(self, N: int = 20_000) -> anndata.AnnData:
        self._gmm_sampling = GaussianMixtureModelSampling(
            adata=self._adata,
            n_components=self._n_components,
            n_gmm_components=self._n_gmm_components,
            random_state=self._random_state,
        )
        adata_sim = self._gmm_sampling(target_size=N)
        sdq.tl.bin_pseudotime(adata_sim, pseudotime_key = self._pseudotime_key, n_bins=self._n_bins)
        return adata_sim

    def forward(self, N: int = 20_000, seed: int = 0) -> None:
        """ """

        adata_sim = self._setup(N=N)

        wandb.init(project=self._project_name)
        wandb_logger = lightning.pytorch.loggers.WandbLogger(
            project=self._project_name
        )

        MODEL_PARAMS = {
            "adata": adata_sim,
            "use_key": "X",
            "time_key": "t",
            "mu_hidden": [512, 512],
            "sigma_hidden": [32, 32],
            "train_lr": 1e-4,
            "train_step_size": 1500,
            "monitor_hardware": True,
            "dt": self._dt,
            "batch_size": self._batch_size,
            "train_val_split": [0.9, 0.1],
            "seed": seed,
            "latent_dim": 50,
            "velocity_ratio_params": {
                "target": 2.5,
                "enforce": 100,
                "method": "square",
            },
            "potential_type": "fixed",
        }

        wandb_params = {k: v for k, v in MODEL_PARAMS.items() if k != "adata"}
        wandb_logger.experiment.config.update(wandb_params)
        wandb_logger.experiment.config.update(
            {
                "run_id": self._run_id,
                "N": N,
                "n_bins": self._n_bins,
                "distance_mean": self._gmm_sampling._distance_mean,
                "distance_std": self._gmm_sampling._distance_std,
            }
        )
        MODEL_PARAMS["logger"] = wandb_logger

        model = sdq.scDiffEq(**MODEL_PARAMS)

        model.fit(
            train_epochs=self._train_epochs,
            train_callbacks=[sdq.callbacks.StochasticWeightAveraging(swa_lrs=[1e-5])],
            ckpt_frequency=self._ckpt_frequency,
            save_last_ckpt=self._save_last_ckpt,
            keep_ckpts=self._keep_ckpts,
            monitor=self._monitor,
            mode="min",
        )

        wandb.finish()

    def __call__(
        self,
        N: int = 20_000,
        n_bins: int = 10,
        seed: int = 0,
        batch_size: int = 2048,
        dt: float = 0.1,
    ) -> None:
        """ """

        self.__update__(locals())

        self.forward(N=N, seed=seed)
