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
    growth_weights_path: Optional[Union[pathlib.Path, str]] = None,
    weight_key: str = "W",
    time_key: str = "Time point",
    time_point: int = 2,
) -> None:
    print("Loading adata...")
    adata = anndata.read_h5ad(h5ad_path, silent=True)

    # -- UPDATE TO TRAIN ONLY ON WELLS 0, 1: ----------------------------
    adata.obs["train"] = adata.obs["Well"].isin([0, 1])

    if not growth_weights_path is None:
        logger.info("Annotating adata with gro    wth weights...")
        adata = annotate_adata_with_growth_weights(
            adata, growth_weights_path, weight_key, time_key, time_point
        )
        logger.info(f"\nAnnData prepared:\n{adata}")
    return adata
