# -- import packages: ---------------------------------------------------------
import ABCParse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


# -- set typing: --------------------------------------------------------------
from typing import List


# -- operational object class: ------------------------------------------------
class TemporalColorMapFactory(ABCParse.ABCParse):
    def __init__(self, cmap: matplotlib.colors.ListedColormap = cm.plasma_r):
        self.__parse__(locals())

    @property
    def _original_cmap(self):
        return np.array(self._cmap.colors)

    @property
    def cmap(self) -> List[List[float]]:
        _cmap = self._original_cmap[:: self._idx_slice][
            self._pad_left : -self._pad_right
        ]
        return matplotlib.colors.ListedColormap(_cmap, name="TemporalColorMap")

    def __call__(
        self,
        idx_slice: int = 6,
        pad_left: int = 1,
        pad_right: int = 1,
        plot: bool = False,
    ):
        self.__update__(locals())

        return self.cmap


# -- API-facing function: -----------------------------------------------------
def generate_temporal_cmap(
    idx_slice: int = 6,
    pad_left: int = 1,
    pad_right: int = 1,
    plot: bool = False,
    cmap: matplotlib.colors.ListedColormap = cm.plasma_r,
):
    time_cmap = TemporalColorMapFactory()
    return time_cmap(
        idx_slice=idx_slice, pad_left=pad_left, pad_right=pad_right, plot=plot
    )


# -- API-facing object: --------------------------------------------------------
temporal_colormap = generate_temporal_cmap()
