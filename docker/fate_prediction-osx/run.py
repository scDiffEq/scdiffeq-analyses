# -- import packages: ---------------------------------------------------------
import scdiffeq as sdq
import scdiffeq_analyses as sdq_an
import larry

for package in [sdq, sdq_an, larry]:
    print(package.__version__)
