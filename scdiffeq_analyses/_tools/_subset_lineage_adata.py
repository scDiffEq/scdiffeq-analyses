# -- import packages: ---------------------------------------------------------
import anndata

# -- function: ----------------------------------------------------------------
def subset_lineage_adata(
        adata: anndata.AnnData,
        cell_idx: int,
        lineage_key: str = "clone_idx",
) -> anndata.AnnData:
    """Subset an AnnData object to a single lineage, based on a given cell index.
    
    Args:
        adata: AnnData object.
        cell_idx: Index of the cell to subset.
        lineage_key: Key in adata.obs that contains the lineage information.
        
    Returns:
        AnnData object containing the lineage.
    """
    clone_idx = adata.obs.loc[str(cell_idx)][lineage_key]
    adata_lin = adata[adata.obs[lineage_key] == clone_idx].copy()
    return adata_lin
