# -- import packages: ----------------------------------------------------------
import anndata
import numpy as np
import pathlib
import scdiffeq as sdq
import torch

# -- set type hints: ----------------------------------------------------------
from typing import Union


# -- define function: ----------------------------------------------------------
def annotate_subset_adata_with_growth_weights(
    adata: anndata.AnnData,
    weight_key: str = "W",
    time_key: str = "Time point",
    time_point: int = 2,
):
    weight_pt = sdq.datasets.larry_kegg_growth_weights()

    obs_df = adata.obs.copy()
    T_MASK = obs_df[time_key] == time_point
    t_cell_idx = obs_df[T_MASK]["idx"].values

    d2_idx = np.load(
        "/Users/mvinyard/scdiffeq_data/larry/d2_idx.larry.npy", allow_pickle=True
    )

    iloc_idx = np.array([np.where(d2_idx == item)[0] for item in t_cell_idx]).flatten()

    W = np.ones(len(adata))

    W[T_MASK] = weight_pt["w"][0][iloc_idx]
    adata.obs[weight_key] = W

    return adata
