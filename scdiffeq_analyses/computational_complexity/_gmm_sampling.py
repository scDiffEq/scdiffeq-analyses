# -- import packages: ---------------------------------------------------------
import adata_query
import anndata
import numpy as np
import scipy.sparse
import sklearn.decomposition
import sklearn.mixture
import sklearn.preprocessing

# -- import local dependencies: -----------------------------------------------
from ._map_time import map_time
from ._simulation_visualization import SimulationVisualization

# -- cls: ---------------------------------------------------------------------
class GaussianMixtureModelSampling:
    """
    Fit GMM to original data, then sample from the fitted distribution
    """

    def __init__(
        self,
        adata: anndata.AnnData,
        n_components: int = 50,
        n_gmm_components: int = 20,
        random_state: int = 42,
    ) -> None:

        self._adata = adata
        self._n_components = n_components
        self._n_gmm_components = n_gmm_components
        self._random_state = random_state

        self._preprocess()
        self._fit()
        self._visualization = SimulationVisualization(X_ref=self.X_pca)

    @property
    def n_cells(self) -> int:
        return self._adata.shape[0]

    @property
    def X_raw(self):
        if not hasattr(self, "_X_raw"):
            self._X_raw = adata_query.fetch(self._adata, key="X")
            if scipy.sparse.issparse(self._X_raw):
                if scipy.sparse.isspmatrix_csr(self._X_raw):
                    self._X_raw = self._X_raw.toarray()
                else:
                    self._X_raw = self._X_raw.A
        return self._X_raw

    def _preprocess(self) -> None:
        # -- scaler: ----------------------------------------------------
        self.scaler_model = sklearn.preprocessing.StandardScaler()
        self.X_scaled = self.scaler_model.fit_transform(self.X_raw)
        # -- pca: -------------------------------------------------------
        self.pca_model = sklearn.decomposition.PCA(n_components=self._n_components)
        self.X_pca = self.pca_model.fit_transform(self.X_scaled)

        # -- update reference adata: ------------------------------------
        self._adata.obsm["X_pca"] = self.X_pca

    def _fit(self) -> None:
        self.gmm = sklearn.mixture.GaussianMixture(
            n_components=self._n_gmm_components,
            random_state=self._random_state,
        )
        self.gmm.fit(self.X_pca)

    def _compute_distance(self, X_ref, X_sim) -> None:

        ref_mean, ref_std = X_ref.mean(0), X_ref.std(0)
        sim_mean, sim_std = X_sim.mean(0), X_sim.std(0)

        self._distance_mean = np.abs(ref_mean - sim_mean).max()
        self._distance_std = np.abs(ref_std - sim_std).max()

        print(f"mean ± std difference: {self._distance_mean:.3f}±{self._distance_std:.3f}")

    def __call__(
        self,
        target_size: int,
        report_validation: bool = True,
        plot: bool = False,
        n_neighbors: int = 20,
        time_key: str = "ct_pseudotime",
        constant: float = 1e-8,
    ) -> anndata.AnnData:
        X_sim, _ = self.gmm.sample(target_size)
        self.t_ref = self._adata.obs[time_key]
        self.t_sim = map_time(
            adata=self._adata,
            X_sim=X_sim,
            time_key=time_key,
            constant=constant,
            n_neighbors=n_neighbors,
        )

        if report_validation:
            self._compute_distance(X_ref=self.X_pca, X_sim=X_sim)
        if plot:
            self._visualization.plot(X_sim, t_ref=self.t_ref, t_sim=self.t_sim)

        adata_sim = anndata.AnnData(X_sim)
        adata_sim.obs[time_key] = self.t_sim

        return adata_sim
