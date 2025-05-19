# -- import packages: ---------------------------------------------------------
import scdiffeq as sdq
import scdiffeq_analyses as sdq_an
import larry
import os
import wandb

for package in [sdq, sdq_an, larry]:
    print(f"{package.__name__}: {package.__version__}")

sdq_an.fate_prediction.run_fate_prediction(
    project_name="fate_prediction.testing_docker_osx",
    h5ad_path=os.environ["H5AD_PATH"],
    time_key="t",
    wandb_api_key=os.environ["WANDB_API_KEY"],
)
