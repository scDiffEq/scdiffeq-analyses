#!/home/mvinyard/.anaconda3/envs/sdq-dev/bin/python

# pip install scdiffeq
# pip install -U 'wandb>=0.12.10

# -- IMPORT DEPENDENCIES: -----------------------------------------------------
import scdiffeq as sdq
import lightning
import wandb
import torch
import pathlib
import ABCParse
import autodevice
import pandas as pd
import numpy as np
import anndata
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
import ast
import pathlib

import scvelo as scv
import cellrank as cr

print(f"sdq path: {sdq.__path__}")
print(f"sdq version: {sdq.__version__}")


# -- SET TYPING: --------------------------------------------------------------
from typing import Optional, Union, List, Dict


# -- DEFINE VARIABLES: --------------------------------------------------------
PROJECT_NAME = "scDiffEq.pancreas.cytotrace_time.v5" # Method.Dataset.Specification
DIFFEQ_TYPE = "SDE"
USE_KEY = "X_pca"
LATENT_DIM = 50
KEEP_CKPTS = 5


def parse_tuple(string):
    if not string is None:
        try:
            return ast.literal_eval(string)
        except:
            raise argparse.ArgumentTypeError("Argument must be a tuple or list")


# -- DEFINE SWEEP VARIABLES: --------------------------------------------------
parser = argparse.ArgumentParser(description='run_scdiffeq.fate_prediction.argparse')
parser.add_argument(
    '--mu_hidden',
    default=[512, 512],
    type=parse_tuple, # List[int],
    help='(List[int]) Hidden layer sizes for mu (diffusion) network.',
)
parser.add_argument(
    '--sigma_hidden',
    default=[32, 32],
    type=parse_tuple, # List[int],
    help='(List[int]) Hidden layer sizes for sigma (diffusion) network.',
)
parser.add_argument(
    '--train_epochs',
    default=125,
    type=int,
    help='(int) Training epochs.',
)
parser.add_argument(
    '--train_lr',
    default=1e-03,
    type=float,
    help='(float) Initial train learning rate.',
)
parser.add_argument(
    '--train_step_size',
    default=2500,
    type=int,
    help='(int) Learning rate step size during training.',
)
parser.add_argument(
    '--n_seeds',
    default=1,
    type=int,
    help='(int) Number of seeds over which to train / evaluate.',
)
parser.add_argument(
    '--seeds',
    default=None,
    type=str, # Union[List[int], int, str, None],
    help='(Union[List[int], int, str]) Specific seed(s) to use.',
)
# parser.add_argument(
#     '--n_eval',
#     default=200,
#     type=int,
#     help='(int) Number of trajectories to evaluate.',
# )
parser.add_argument(
    '--swa_lrs',
    default=1e-03,
    type=float,
    help='(float) Stochastic weight averaging learning rates.',
)
parser.add_argument(
    '--mu_dropout',
    default=0,
    type=float,
    help='(float) Hidden [mu] dropout.',
)
parser.add_argument(
    '--sigma_dropout',
    default=0,
    type=float,
    help='(float) Hidden [sigma] dropout.',
)
parser.add_argument(
    '--batch_size',
    default=2048,
    type=int,
    help='(int) Batch size.',
)
parser.add_argument(
    '--potential_type',
    default="fixed",
    type=str,
    help='(Union[str, None]) Type of potential to use.',
)
parser.add_argument(
    '--coef_g',
    default=1,
    type=float,
    help='(float) Coefficient of diffusion.',
)

parser.add_argument(
    '--velocity_ratio_target',
    default=2,
    type=float,
    help='(float) Ratio of drift to diffusion.',
)

parser.add_argument(
    '--velocity_ratio_enforce',
    default=100,
    type=float,
    help='(float) Strength (muliplier) at which to enforce the rate of drift to diffusion in the loss function.',
)

parser.add_argument(
    '--velocity_ratio_method',
    default="square",
    type=str,
    help='(float) loss processing method (abs or square).',
)

args = parser.parse_args()


MU_HIDDEN = args.mu_hidden
SIGMA_HIDDEN = args.sigma_hidden
TRAIN_EPOCHS = args.train_epochs
TRAIN_LR = args.train_lr
TRAIN_STEP_SIZE = args.train_step_size
N_SEEDS = args.n_seeds
SEEDS = args.seeds
# N_EVAL = args.n_eval
SWA_LRS = args.swa_lrs
MU_DROPOUT = args.mu_dropout
SIGMA_DROPOUT = args.sigma_dropout
BATCH_SIZE = args.batch_size
POTENTIAL_TYPE = args.potential_type
COEF_DIFFUSION = args.coef_g
VELOCITY_RATIO_PARAMS={
    "target": args.velocity_ratio_target,
    "enforce": args.velocity_ratio_enforce,
    "method": args.velocity_ratio_method,
}

if not SEEDS is None:
    SEEDS = [int(i) for i in SEEDS.strip("[").strip("]").split(",")]

if POTENTIAL_TYPE == "None":
    POTENTIAL_TYPE = None
    
    
# -- Cytotrace time code: -----------------------------------------------------

def assign_bin(value, bins):
    match_idx = np.all([value >= bins[:, 0], value <= bins[:, 1]], axis=0)
    assigned_bin = bins[match_idx].flatten()
    return pd.Series(
        {"ti": assigned_bin[0], "tj": assigned_bin[1], "bin": np.where(match_idx)[0][0]}
    )


# -- OPERATIONAL CODE: --------------------------------------------------------
class Runner(ABCParse.ABCParse):

    _script_name = pathlib.Path(__file__).name

    def __init__(
        self,
        project_name: str,
        time_key: str = "t",
        h5ad_path: Optional[Union[pathlib.Path, str]] = None,
        growth_weights_path: Optional[Union[pathlib.Path, str]] = None,
        weight_key: str = "W",
        ckpt_frequency=1,
        save_last_ckpt=True,
        keep_ckpts=5,
        monitor: str = "epoch_validation_loss",
    ):
        self.__parse__(locals())

        self._load_data()

#     def _annotate_adata_with_growth_weights(self, adata: anndata.AnnData):

#         weight_pt = torch.load(self._growth_weights_path)
#         W = np.ones(len(adata))
#         W[adata.obs["Time point"] == 2] = weight_pt["w"][0]
#         adata.obs[self._weight_key] = W
#         return adata

    def _load_data(self):
        print("Loading adata...")

#         adata = sdq.datasets.pancreas(data_dir = "/home/mvinyard/data/")
        adata = scv.datasets.pancreas()
        print("preprocessing...")
        scv.pp.moments(adata)
        adata.layers["Ms"] = adata.X
        ctk = cr.kernels.CytoTRACEKernel(adata)
        ctk.compute_cytotrace()
        
        print("annotating time...")
        
        bounds = np.linspace(0, 1, 12)
        bins = np.array([[i, j] for i, j in zip(bounds[:-1], bounds[1:])])
        
        time_df = adata.obs["ct_pseudotime"].apply(assign_bin, bins=bins)
        adata.obs = pd.merge(adata.obs, time_df, how="left", on="index")
        adata.obs["bin"] = adata.obs["bin"].astype(int)
        adata.obs = adata.obs.rename({"bin": "t"}, axis=1)
        
#         if not self._growth_weights_path is None:
#             print("Annotating adata with growth weights...")
#             adata = self._annotate_adata_with_growth_weights(adata)
        print(f"\nAnnData prepared:\n{adata}")
        self._ADATA = adata

    @property
    def adata(self):
        if not hasattr(self, "_ADATA"):
            self._load_data()
        return self._ADATA

    def wandb_log_script_as_artifact(self):
        """Should run each iteration/seed, after wandb logger is initialized."""
        artifact = wandb.Artifact(
            name=self._script_name,
            type="run-script.py",
        )
        artifact.add_file(__file__)
        wandb.log_artifact(artifact)

    @property
    def _SWA_CALLBACK(self):
        return lightning.pytorch.callbacks.StochasticWeightAveraging(
            swa_lrs=self._swa_lrs
        )

    @property
    def _TRAIN_CALLBACKS(self):
        return [self._SWA_CALLBACK] # self._FATE_PREDICTION_CALLBACK

    @property
    def _LOG_DIR(self):
        return self.model._metrics_path.parent

    @property
    def _CKPT_DIR(self):
        return self._LOG_DIR.joinpath("checkpoints/")

    @property
    def _CKPT_PATHS(self):
        return self._CKPT_DIR.glob("*.ckpt")

    def _compose_model_params(self, seed):
        return {
            "adata": self.adata,
            "seed": seed,
#             "name": self._project_name,
            "latent_dim": self._latent_dim,
            "time_key": self._time_key,
            "use_key": self._use_key,
#             "weight_key": self._weight_key,
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
        }

    def _setup_model_run(self, seed: int):

        wandb_logger = lightning.pytorch.loggers.WandbLogger(project=self._project_name)
        self.wandb_log_script_as_artifact()
        params = self._compose_model_params(seed)
        wandb_logger.experiment.config.update(params)
        wandb_logger.experiment.config.update({"h5ad_path": self._h5ad_path})
        params["logger"] = wandb_logger
        print(f"Model run (seed: {seed}) initialized.")

        return wandb_logger, params

    def _format_ckpt_fname(self, ckpt_fpath: pathlib.Path):
        return ckpt_fpath.name.split(".")[0].replace("=", "_").replace("-", ".")

    def log_wandb_model_artifact(self, seed, ckpt_fpath):

        ckpt_fname = self._format_ckpt_fname(ckpt_fpath)

        artifact = wandb.Artifact(
            name=f"model-ckpt-seed_{seed}-{ckpt_fname}", type="model"
        )
        artifact.add_file(ckpt_fpath)
        wandb.log_artifact(artifact)

        return ckpt_fname

    def log_wandb_fate_prediction_metrics(self, seed):

        print("logging fate prediction metrics")

        CKPT_METRICS_PATH = self._FATE_PREDICTION_CALLBACK._CKPT_METRICS_PATH

        artifact = wandb.Artifact(
            name=f"fate-prediction-metrics-seed_{seed}",
            type="model-prediction-task",
            metadata={"N": self._n_eval},
        )
        artifact.add_dir(CKPT_METRICS_PATH)
        wandb.log_artifact(artifact)

    def fate_prediction_evaluation(self, seed, ckpt_fpath):

        """"""

        print("running fate prediction evaluation")
        print(f"ckpt_fpath: {ckpt_fpath}")

        ckpt_fname = self.log_wandb_model_artifact(seed=seed, ckpt_fpath=ckpt_fpath)

        self.model.DiffEq = sdq.io.load_diffeq(ckpt_fpath)
        self.model.to(autodevice.AutoDevice())
        self._FATE_PREDICTION_CALLBACK.forward(
            pl_module=self.model.DiffEq, ckpt_name=ckpt_fname, log_dir=self._LOG_DIR
        )
        self.log_wandb_fate_prediction_metrics(seed=seed)

    def forward(self, seed: int):
        
        wandb.init(project=PROJECT_NAME)

        print(f"initializing seed: {seed}")

        wandb_logger, params = self._setup_model_run(seed)

#         self._FATE_PREDICTION_CALLBACK = larry.callbacks.FatePredictionCallback(
#             self.adata, N=self._n_eval
#         )

        # -- model run: -------------------------------------------------------
        print(f"run model - seed: {seed}")
        self.model = sdq.scDiffEq(**params)
        self.model.fit(
            train_epochs=self._train_epochs,
            ckpt_frequency=self._ckpt_frequency,
            save_last_ckpt=self._save_last_ckpt,
            keep_ckpts=self._keep_ckpts,
            monitor=self._monitor,
            mode="min",
            train_callbacks=self._TRAIN_CALLBACKS,
        )

#         self.log_wandb_fate_prediction_metrics(seed=seed)
        # ---------------------------------------------------------------------

        # -- evaluation of saved ckpts: ---------------------------------------
#         print(f"run evaluation - seed: {seed}")
#         for ckpt_fpath in self._CKPT_PATHS:
#             self.fate_prediction_evaluation(seed=seed, ckpt_fpath=ckpt_fpath)
        # ---------------------------------------------------------------------
        wandb.finish()

    def __call__(
        self,
        n_seeds: int,
        seeds: Optional[Union[List[int], int]] = None,
        train_epochs: int = 125,
        train_step_size: int = 25,
        mu_hidden: List[int] = [2048, 2048],
        sigma_hidden: List[int] = [1024, 1024],
        mu_dropout: float = 0,
        sigma_dropout: float = 0,
        train_lr=4e-5,
#         n_eval: int = 2000,
        swa_lrs: float = 1e-8,
        batch_size: int = 512,
        potential_type: Union[None, str] = "fixed",
        coef_diffusion: float = 1,
        diffeq_type: str = "SDE",
        use_key: str = "X_pca",
        latent_dim: int = 50,
        velocity_ratio_params: Dict[str, Union[float, str]] = {
            "target": 1,
            "enforce": 0,
            "method": "square",
        }
    ):

        self.__update__(locals())
        
        self._potential_type = potential_type
        
        
        seeds = range(5)
#         if seeds:
#           seeds = ABCParse.as_list(seeds)
#        else:
#            seeds = range(self._n_seeds)
                

        for seed in seeds:
            self.forward(seed)

# -- RUN MODEL: ---------------------------------------------------------------

runner = Runner(
#     h5ad_path=H5AD_PATH,
    project_name = PROJECT_NAME,
#     growth_weights_path=GROWTH_WEIGHTS_PATH,
#     weight_key=WEIGHT_KEY,
    keep_ckpts=KEEP_CKPTS,
)

print(f"N_SEEDS: {N_SEEDS}")
print(f"SEEDS: {SEEDS}")

runner(
    n_seeds=N_SEEDS,
    seeds=SEEDS,
#     n_eval=N_EVAL,
    swa_lrs=SWA_LRS,
    mu_hidden=MU_HIDDEN,
    sigma_hidden=SIGMA_HIDDEN,
    train_epochs = TRAIN_EPOCHS,
    train_lr=TRAIN_LR,
    train_step_size=TRAIN_STEP_SIZE,
    mu_dropout=MU_DROPOUT,
    sigma_dropout=SIGMA_DROPOUT,
    batch_size=BATCH_SIZE,
    potential_type=POTENTIAL_TYPE,
    coef_diffusion=COEF_DIFFUSION,
    diffeq_type=DIFFEQ_TYPE,
    use_key=USE_KEY,
    latent_dim=LATENT_DIM,
    velocity_ratio_params=VELOCITY_RATIO_PARAMS,
)
