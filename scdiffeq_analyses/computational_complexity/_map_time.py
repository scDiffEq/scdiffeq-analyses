# -- import packages: ---------------------------------------------------------
import anndata
import numpy as np
import scdiffeq as sdq


# -- cls: ---------------------------------------------------------------------
class TimeMapping:
    def __init__(
        self,
        adata: anndata.AnnData,
        n_neighbors: int = 20,
        time_key: str = "ct_pseudotime",
        constant: float = 1e-8,
    ) -> None:
        """TimeMapping cls
        
        Args:
            adata (anndata.AnnData): AnnData object
            n_neighbors (int): Number of neighbors
            time_key (str): Key for the time variable
            constant (float): Constant for the weights
        """
        self._adata = adata
        self._n_cells = adata.shape[0]
        self._n_neighbors = n_neighbors
        self._time_key = time_key
        self._constant = constant
        self._t_sim = []

    @property
    def t_ref(self):
        if not hasattr(self, "_t_ref"):
            self._t_ref = self._adata.obs[self._time_key]
        return self._t_ref

    @property
    def kNN(self):
        if not hasattr(self, "_knn"):
            self._knn = sdq.tl.kNN(
                self._adata, n_neighbors=self._n_neighbors, use_key="X_pca"
            )
        return self._knn

    def _compute_weights(self, nn_dist):
        weights = 1.0 / (nn_dist + self._constant)
        return weights / weights.sum()

    def forward(self, x_nn, x_dist):

        if x_dist.min() < 1e-8:
            return self.t_ref[x_nn[x_dist.argmin()]]
        else:
            weights = self._compute_weights(x_dist)
            return np.sum(weights * self.t_ref[x_nn].values)

    def __call__(self, X_sim):

        X_nn, X_dist = self.kNN.query(X_query=X_sim, include_distances=True)

        for i in range(len(X_sim)):
            self._t_sim.append(self.forward(x_nn=X_nn[i], x_dist=X_dist[i]))
        return np.array(self._t_sim)

# -- function: ----------------------------------------------------------------
def map_time(
    adata: anndata.AnnData,
    X_sim: np.ndarray,
    n_neighbors: int = 20,
    time_key: str = "ct_pseudotime",
    constant: float = 1e-8,
):
    """
    Args:
        adata (anndata.AnnData)
        X_sim (np.ndarray)
        n_neighbors (int) = 20,
        time_key (str) = "ct_pseudotime",
        constant (float) = 1e-8,

    Returns:
        t_sim (np.ndarray)
    """
    time_mapping = TimeMapping(
        adata=adata,
        n_neighbors=n_neighbors,
        time_key=time_key,
        constant=constant,
    )
    return time_mapping(X_sim)
