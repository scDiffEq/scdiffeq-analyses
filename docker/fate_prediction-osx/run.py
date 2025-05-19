# -- import packages: ---------------------------------------------------------
import scdiffeq as sdq
import scdiffeq_analyses as sdq_an
import larry
import os

for package in [sdq, sdq_an, larry]:
    print(package.__version__)

# STATIC_PATH = "/app/static/Weinreb2020_growth-all_kegg.pt"
# assert os.path.exists(STATIC_PATH), "Missing static Weinreb2020_growth-all_kegg.pt!"
