
# -- import packages: ---------------------------------------------------------
import anndata

# -- set typing: --------------------------------------------------------------
from typing import Union, Tuple
# CytoTRACEKernel = cellrank.kernels.CytoTRACEKernel


# -- API-facing function: -----------------------------------------------------
def compute_CytoTRACE_pseudotime(
    adata: anndata.AnnData, return_kernel: bool = False,
): # -> Union[anndata.AnnData, Tuple[anndata.AnnData, CytoTRACEKernel]]:
    """Use cellrank to compute CytoTRACE"""

    import scvelo
    import cellrank
    
    # -- preprocess with scvelo: ----------------------------------------------
    scvelo.pp.moments(adata)

    # -- compute CytoTRACE time with cellrank: --------------------------------
    adata.layers["Ms"] = adata.X
    ctk = cellrank.kernels.CytoTRACEKernel(adata)
    ctk.compute_cytotrace()

    if return_kernel:
        return adata, ctk
    return adata
