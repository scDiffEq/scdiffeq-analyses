# -- import packages: ---------------------------------------------------------
import scdiffeq as sdq
import scdiffeq_analyses as sdq_an
import larry
import os

for package in [sdq, sdq_an, larry]:
    print(f"{package.__name__}: {package.__version__}")

sdq_an.fate_prediction.run_fate_prediction(
    project_name="fate_prediction.testing_docker_linux",
    h5ad_path="/app/static/Weinreb2020_growth-all_kegg.h5ad",
    time_key="t",
)
