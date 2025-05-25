# -- import packages: ---------------------------------------------------------
import autodevice
import gc
import logging
import os
import scdiffeq as sdq
import scdiffeq_analyses as sdq_an
import sys
import torch
import traceback
import wandb
import yaml

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
    logger.info("Launching computational complexity experiment")
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

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError(
            "WANDB_API_KEY must be provided via environment variable or config file"
        )

    params = {
        "N": config.get("N"),
        "seed": config.get("seed"),
        "n_bins": config.get("n_bins"),
    }

    # Log parameters being used
    logger.info("Running with parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")


    # Force garbage collection before starting
    gc.collect()

    adata = sdq.datasets.pancreatic_endocrinogenesis()
    experiment = sdq_an.computational_complexity.ExperimentRunner(
        adata=adata,
        wandb_api_key=wandb_api_key,
        project_name="complexity.testing",
    )
    experiment(
        N=params["N"],
        seed=params["seed"],
        n_bins=params["n_bins"],
    )

except Exception as e:
    logger.error(f"Fatal error: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

    try:

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
