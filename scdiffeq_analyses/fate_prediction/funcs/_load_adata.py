# -- import packages: ----------------------------------------------------------
import anndata
import logging
import pathlib

# -- import local dependencies: -----------------------------------------------
from ._annotate_adata_with_growth_weights import annotate_adata_with_growth_weights

# -- set type hints: ----------------------------------------------------------
from typing import Optional, Union

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)

# -- define function: ---------------------------------------------------------
def load_adata(
    h5ad_path: Union[pathlib.Path, str],
    weight_key: str = "W",
    time_key: str = "Time point",
    time_point: int = 2,
) -> None:

    logger.info(f"Loading adata from {h5ad_path}...")
    adata = anndata.read_h5ad(h5ad_path)

    # -- UPDATE TO TRAIN ONLY ON WELLS 0, 1: ----------------------------
    adata.obs["train"] = adata.obs["Well"].isin([0, 1])

    logger.info("Annotating adata with growth weights...")
    adata = annotate_adata_with_growth_weights(
        adata=adata,
        weight_key=weight_key,
        time_key=time_key,
        time_point=time_point,
    )
    logger.info(f"\nAnnData prepared:\n{adata}")
    return adata
