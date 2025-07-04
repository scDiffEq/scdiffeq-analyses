# -- import packages: ---------------------------------------------------------
import autodevice
import gc
import larry
import logging
import os
import scdiffeq as sdq
import scdiffeq_analyses as sdq_an
import sys
import torch
import traceback
import wandb
import yaml

# -- helper code: -------------------------------------------------------------
import pickle
import scdiffeq as sdq
import os
import anndata
import sklearn.preprocessing
import sklearn.decomposition


def get_larry(n_bins: int):
    """Does the fate prediction preprocessing."""

    adata = sdq.datasets.larry(data_dir="/data/")

    scaler = sklearn.preprocessing.StandardScaler()
    PCA = sklearn.decomposition.PCA(n_components=50)

    adata_train = adata[adata.obs["Well"].isin([0, 1])].copy()
    X_train_raw = adata_train.X.toarray()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_train_pca = PCA.fit_transform(X_train_scaled)

    adata.obsm["X_scaled"] = scaler.transform(adata.X.toarray())
    adata.obsm["X_pca"] = PCA.transform(adata.obsm["X_scaled"])

    h5ad_path = "/data/scdiffeq_data/larry/larry.adata.pp.h5ad"

    sdq.tl.bin_pseudotime(adata, pseudotime_key="ct_pseudotime", n_bins=n_bins)

    logger.info(f"Created AnnData object with {adata.obs['t'].nunique()} time points")

    adata.write_h5ad(h5ad_path)

    with open("/data/scdiffeq_data/larry/larry.pca.pkl", "wb") as file:
        pickle.dump(PCA, file)

    with open("/data/scdiffeq_data/larry/larry.scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    return h5ad_path


# -- configure logger: --------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Configure scdiffeq logger to propagate messages to our logger
logging.getLogger("scdiffeq").setLevel(logging.INFO)
logging.getLogger("scdiffeq_analyses").setLevel(logging.INFO)


try:
    logger.info("Starting fate prediction process")
    # Log MPS availability and autodevice selection
    logger.info(f"Torch MPS available: {torch.backends.mps.is_available()}")
    try:
        selected_device = autodevice.AutoDevice()
        logger.info(f"Autodevice selected: {selected_device}")
    except Exception as e:
        logger.error(f"Error getting autodevice: {e}")

    for package in [sdq, sdq_an, larry]:
        logger.info(f"{package.__name__}: {package.__version__}")

    # Log system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")

    # Load configuration from file or use defaults
    config_path = os.environ.get("CONFIG_PATH", "/app/config.yaml")
    config = {}

    if os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
    else:
        logger.info(
            f"Config file not found at {config_path}, using environment variables and defaults"
        )

    # Environment variables take precedence over config file
    # h5ad_path = os.environ.get("H5AD_PATH", config.get("h5ad_path", ""))
    n_bins = config.get("n_bins", 10)
    if n_bins is None:
        raise ValueError("n_bins must be provided via config file")
    h5ad_path = get_larry(n_bins=n_bins)
    wandb_api_key = os.environ.get("WANDB_API_KEY", config.get("wandb_api_key", ""))
    # Allow time_key override from environment
    time_key = os.environ.get("time_key", config.get("time_key", "t"))

    logger.info(f"Using time_key: {time_key}")

    if not h5ad_path:
        raise ValueError(
            "H5AD_PATH must be provided via `get_larry()`"
        )

    if not wandb_api_key:
        raise ValueError(
            "WANDB_API_KEY must be provided via environment variable or config file"
        )

    logger.info(f"Using h5ad file: {h5ad_path}")

    # Run fate prediction with config
    logger.info("Starting fate prediction run")

    # Extract all parameters for clearer logging
    params = {
        "project_name": config.get(
            "project_name", "fate_prediction.testing_docker_osx"
        ),
        "h5ad_path": h5ad_path,
        "wandb_api_key": wandb_api_key,
        "time_key": time_key,
        "n_seeds": config.get("n_seeds", 1),
        "seeds": config.get("seeds", [0]),
        "train_epochs": config.get("train_epochs", 2500),
        "batch_size": config.get("batch_size", 2048),
        "train_lr": config.get("train_lr", 4e-5),
        "n_eval": config.get("n_eval", 2000),
        "swa_lrs": config.get("swa_lrs", 1e-5),
        "mu_hidden": config.get("mu_hidden", [512, 512]),
        "sigma_hidden": config.get("sigma_hidden", [32, 32]),
        "train_step_size": config.get("train_step_size", 1500),
        "mu_dropout": config.get("mu_dropout", 0),
        "sigma_dropout": config.get("sigma_dropout", 0),
        "coef_diffusion": config.get("coef_diffusion", 1),
        "diffeq_type": config.get("diffeq_type", "SDE"),
        "use_key": config.get("use_key", "X_pca"),
        "potential_type": config.get("potential_type", "fixed"),
        "latent_dim": config.get("latent_dim", 50),
        "weight_key": config.get("weight_key", "W"),
        "time_point": config.get("time_point", 2),
        "ckpt_frequency": config.get("ckpt_frequency", 1),
        "save_last_ckpt": config.get("save_last_ckpt", True),
        "keep_ckpts": config.get("keep_ckpts", 3),
        "monitor": config.get("monitor", "epoch_validation_loss"),
    }

    # Log parameters being used
    logger.info("Running with parameters:")
    for key, value in params.items():
        if key != "wandb_api_key":  # Don't log the API key
            logger.info(f"  {key}: {value}")

    # Get velocity_ratio_params separately as it's nested
    velocity_ratio_params = config.get(
        "velocity_ratio_params",
        {
            "target": 2.5,
            "enforce": 100,
            "method": "square",
        },
    )

    logger.info(f"velocity_ratio_params: {velocity_ratio_params}")

    try:
        # Force garbage collection before starting
        gc.collect()
        logger.info("Loading h5ad file...")
        # We're not loading the h5ad file here directly, but it's a good checkpoint

        logger.info("About to start run_fate_prediction function")

        # Add more verbose logging around model construction
        logger.info("Calling run_fate_prediction with all parameters")
        sdq_an.fate_prediction.run_fate_prediction(
            project_name=params["project_name"],
            h5ad_path=params["h5ad_path"],
            wandb_api_key=params["wandb_api_key"],
            time_key=params["time_key"],
            n_seeds=params["n_seeds"],
            seeds=params["seeds"],
            train_epochs=params["train_epochs"],
            batch_size=params["batch_size"],
            train_lr=params["train_lr"],
            n_eval=params["n_eval"],
            swa_lrs=params["swa_lrs"],
            mu_hidden=params["mu_hidden"],
            sigma_hidden=params["sigma_hidden"],
            train_step_size=params["train_step_size"],
            mu_dropout=params["mu_dropout"],
            sigma_dropout=params["sigma_dropout"],
            coef_diffusion=params["coef_diffusion"],
            diffeq_type=params["diffeq_type"],
            use_key=params["use_key"],
            potential_type=params["potential_type"],
            latent_dim=params["latent_dim"],
            weight_key=params["weight_key"],
            time_point=params["time_point"],
            ckpt_frequency=params["ckpt_frequency"],
            save_last_ckpt=params["save_last_ckpt"],
            keep_ckpts=params["keep_ckpts"],
            monitor=params["monitor"],
            velocity_ratio_params=velocity_ratio_params,
        )
        logger.info("Fate prediction completed successfully")
    except Exception as e:
        logger.error(f"Error during run_fate_prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    logger.info("Process completed successfully")

except Exception as e:
    logger.error(f"Fatal error: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)
