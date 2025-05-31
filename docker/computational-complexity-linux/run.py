# -- import packages: ---------------------------------------------------------
import gc
import logging
import os
import scdiffeq as sdq
import scdiffeq_analyses as sdq_an
import sys
import traceback
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
        "project_name": config.get("project_name"),
        "dt": config.get("dt"),
        "train_epochs": config.get("train_epochs"),
	"batch_size": config.get("batch_size"),
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
        project_name=params["project_name"],
        train_epochs=params["train_epochs"],
    )
    experiment(
        N=params["N"],
        seed=params["seed"],
        n_bins=params["n_bins"],
        dt=params["dt"],
        batch_size=params["batch_size"],
    )

except Exception as e:
    logger.error(f"Fatal error: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)
