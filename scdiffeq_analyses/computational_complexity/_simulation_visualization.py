# -- import packages: ---------------------------------------------------------
import cellplots
import numpy as np
import sklearn.decomposition


# -- cls: ---------------------------------------------------------------------
class SimulationVisualization:
    def __init__(self, X_ref: np.ndarray) -> None:
        """SimulationVisualization cls
        
        Args:
            X_ref (np.ndarray): Reference data
        """

        self._X_ref = X_ref

        # -- fit viz pca on reference cells: ---------------------
        self.viz_pca = sklearn.decomposition.PCA(n_components=2)
        self.ref_2d = self.viz_pca.fit_transform(self._X_ref)

    def plot(self, X_sim: np.ndarray, t_ref: np.ndarray, t_sim: np.ndarray) -> None:

        self.sim_2d = self.viz_pca.transform(X_sim)

        fig, axes = cellplots.plot(2, 2, wspace=0.1)

        pl_data = [self.ref_2d, self.sim_2d]
        color = [t_ref, t_sim]

        for en, ax in enumerate(axes):
            c = color[en]
            c_idx = np.argsort(c)
            c = c[c_idx]
            x, y = pl_data[en][c_idx, 0], pl_data[en][c_idx, 1]
            ax.scatter(x, y, c=c, cmap="plasma_r", ec="None", rasterized=True)
