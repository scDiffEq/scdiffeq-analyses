import ABCParse
import cellplots
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from ._box_plot import BoxPlot
from ._foreground_scatter import ForegroundScatter

from typing import List, Optional

class StyledBoxPlot(ABCParse.ABCParse):

    def __init__(
        self,
        colors: Optional[List[str]] = cm.tab20.colors,
        widths: Optional[float] = 0.85,
        scatter_kw={
            "alpha": 0.8,
            "s": 35,
        },
        suppress_scatter: bool = False,
        *args,
        **kwargs,
    ) -> None:

        self.__parse__(locals())

        self.BACKGROUND_KWARGS = {
            "showmeans": True,
            "showfliers": False,
            "meanline": True,
            "zorder": 1,
            "widths": self._widths,
        }
        self.FOREGROUND_KWARGS = {
            "showmeans": False,
            "showfliers": False,
            "meanline": False,
            "zorder": 2,
            "widths": self._widths,
        }

    @property
    def colors(self):
        if not hasattr(self, "_colors") or self._colors is None:
            self._colors = list(cm.tab20.colors)
        return self._colors

    def forward(self) -> None:
        bp = BoxPlot(boxplot_kwargs = self.FOREGROUND_KWARGS, colors=self._colors)
        self.fore_bp = bp(ax=self.ax, data = self._data, mode = "foreground", use_x = self._use_x)
        bp = BoxPlot(boxplot_kwargs = self.BACKGROUND_KWARGS, colors=self._colors)
        self.back_bp = bp(ax=self.ax, data = self._data, mode = "background", use_x = self._use_x)
        if not self._suppress_scatter:
            foreground_scatter = ForegroundScatter(
                colors = self._colors,
                scatter_kw = self._scatter_kw,
            )
            foreground_scatter(
                ax=self.ax, data=self._data, use_x=self._use_x, jitter=self._jitter
            )

        for en, (k, v) in enumerate(self._data.items()):
            if len(v) == 1:
                self.ax.scatter(
                    x = [int(en + 1)] * len(v),
                    y = v,
                    color = self._colors[en],
                    s = self._scatter_kw["s"],
                    alpha = self._scatter_kw["alpha"],
                )

    def __call__(self, data, ax: Optional[plt.Axes] = None, use_x: bool = False, jitter: bool = True) -> None:
        self._data = data
        self._jitter = jitter
        if ax is None:
            fig, axes = cellplots.plot()
            ax = axes[0]
        self.ax = ax
        self._use_x = use_x
        self.forward()


def boxplot(
    data: dict,
    ax: Optional[plt.Axes] = None,
    colors: Optional[List[str]] = cm.tab20.colors,
    widths: Optional[float] = 0.85,
    suppress_scatter: bool = False,
    use_x: bool = False,
    jitter: bool = True,
    scatter_kw={
        "alpha": 0.8,
        "s": 35,
    },
    *args,
    **kwargs,
) -> None:
    """
    Boxplot with foreground scatter points.
    """
    cls = StyledBoxPlot(
        colors = colors,
        widths = widths,
        scatter_kw = scatter_kw,
        suppress_scatter = suppress_scatter,
        *args,
        **kwargs,
    )
    cls(data, ax, use_x=use_x, jitter=jitter)
    return cls
