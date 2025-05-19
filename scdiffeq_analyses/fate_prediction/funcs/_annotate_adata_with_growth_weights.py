# -- import packages: ----------------------------------------------------------
import anndata
import numpy as np
import pathlib
import scdiffeq as sdq
import torch

# -- set type hints: ----------------------------------------------------------
from typing import Union


# -- define function: ----------------------------------------------------------
def annotate_adata_with_growth_weights(
    adata: anndata.AnnData,
    growth_weights_path: Union[pathlib.Path, str],
    weight_key: str = "W",
    time_key: str = "Time point",
    time_point: int = 2,
):

    weight_pt = sdq.datsets.larry_growth_weights()
    # weight_pt = torch.load(growth_weights_path, weights_only=False)
    W = np.ones(len(adata))
    W[adata.obs[time_key] == time_point] = weight_pt["w"][0]
    adata.obs[weight_key] = W
    return adata
